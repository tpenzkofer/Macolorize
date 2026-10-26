import CoreImage
import CoreML

/// DRUNet denoiser, tiled with feathered stitching, driven by a PER-PIXEL
/// noise-level map. Film grain amplitude in the gamma-encoded pixels the
/// model sees grows strongly with brightness — 5× from shadows to
/// highlights on the issue-1 cinema frame — so any single sigma wipes the
/// shadows flat while leaving grain in the highlights (issue #1 follow-up:
/// "dark areas get completely smoothed over while lighter areas retain the
/// grain"). The stage measures the image's own grain-vs-brightness curve
/// and feeds DRUNet the matching sigma at every pixel; the strength slider
/// scales that measured grain.
public struct DenoiseStage: ProcessingStage {
    public let id = StageID.denoise
    let settings: DenoiseSettings

    public init(settings: DenoiseSettings) {
        self.settings = settings
    }

    /// Strength → multiple of the measured grain std. Film grain is
    /// spatially correlated (clumps span several pixels), which a denoiser
    /// trained on white noise partly reads as texture — it needs a sigma
    /// well above the measured std: 1/3 = the measured level, 100% = 3×.
    static func grainFactor(strength: Double) -> Double {
        3 * strength.clamped(to: 0...1)
    }

    /// DRUNet is bias-free, which generalizes past its trained sigma 0–50.
    static let maxSigma: Double = 150

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        guard settings.strength > 0 else { return input }
        // The grain curve is measured on the FULL-RESOLUTION rendition so
        // preview and export use the same curve (proxies soften grain).
        let profile = GrainProfile.measure(input.statsSource, ciContext: context.ciContext)
        let factor = Self.grainFactor(strength: settings.strength)

        let result: CIImage
        if ModelStore.isAvailable(.drunetMap) {
            result = try await denoiseWithMap(input.image, profile: profile,
                                              factor: factor, context: context)
        } else {
            // Legacy scalar model (installs from before v1.3): one sigma at
            // the frame's typical grain level until the update is installed.
            result = try await denoiseScalar(input.image,
                                             sigma: min(profile.typicalSigma * factor, Self.maxSigma),
                                             context: context)
        }

        var output = input
        output.image = result
        return output
    }

    private func denoiseWithMap(_ image: CIImage, profile: GrainProfile,
                                factor: Double, context: RenderContext) async throws -> CIImage {
        let model = try await ModelStore.shared.model(.drunetMap)
        let map = SendableImage(image: profile.noiseMap(for: image, factor: factor,
                                                        maxSigma: Self.maxSigma)
                                    .clampedToExtent())
        let tileSize = 512
        let runner = TiledModelRunner(tileSize: tileSize, overlap: 128, outputScale: 1,
                                      inputName: "image", outputName: "denoised")
        return try await runner.run(model: model, over: image,
                                    ciContext: context.ciContext) { buffer, source in
            let mapBuffer = try InpaintStage.makeGrayPixelBuffer(size: tileSize)
            // Values are sigma/255 in the working space; render them raw
            // (no color space) so each byte holds sigma on the 0-255 scale.
            RenderService.context.render(
                map.image.cropped(to: source)
                    .transformed(by: .init(translationX: -source.minX, y: -source.minY)),
                to: mapBuffer,
                bounds: CGRect(x: 0, y: 0, width: tileSize, height: tileSize),
                colorSpace: nil)
            return try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffer),
                "noise_map": MLFeatureValue(pixelBuffer: mapBuffer),
            ])
        }
    }

    private func denoiseScalar(_ image: CIImage, sigma: Double,
                               context: RenderContext) async throws -> CIImage {
        let model = try await ModelStore.shared.model(.drunet)
        let sigmaValue = Float(sigma)
        let runner = TiledModelRunner(tileSize: 512, overlap: 128, outputScale: 1,
                                      inputName: "image", outputName: "denoised")
        return try await runner.run(model: model, over: image,
                                    ciContext: context.ciContext) { buffer, _ in
            let sigmaArray = try MLMultiArray(shape: [1], dataType: .float32)
            sigmaArray[0] = NSNumber(value: sigmaValue)
            return try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffer),
                "sigma": MLFeatureValue(multiArray: sigmaArray),
            ])
        }
    }
}

/// CIImage is immutable; lets the noise map cross into the tile builder.
private struct SendableImage: @unchecked Sendable {
    let image: CIImage
}

/// The image's grain level as a function of encoded brightness, measured
/// from the residual against a 3×3 median in flat 16×16 blocks.
struct GrainProfile: Sendable {
    /// Sigma (0–255 scale) at `knotCount` evenly spaced encoded-brightness
    /// knots over 0…1.
    let knots: [Double]
    /// Pixel-weighted typical sigma (for the scalar legacy model).
    let typicalSigma: Double

    static let knotCount = 16
    static let blockSize = 16
    /// Residual std vs a ~5×5 median under-reads white noise by ~1%.
    static let medianBias = 1.01
    /// Larger frames are measured on a centered crop of this many pixels
    /// per side — grain-vs-brightness holds frame-wide, memory does not.
    static let maxAnalysisSide: CGFloat = 4096

    /// Encoded (sRGB tone curve) luminance — the domain DRUNet sees.
    static func encodedLuminance(_ image: CIImage) -> CIImage {
        LabColor.luminanceOnly(image)
            .applyingFilter("CILinearToSRGBToneCurve")
    }

    /// Two 3×3 passes ≈ a 5×5 median: a single 3×3 tracks the grain
    /// clumps and under-reads the grain level by ~15%.
    static func despeckled(_ image: CIImage) -> CIImage {
        image.applyingFilter("CIMedianFilter").applyingFilter("CIMedianFilter")
    }

    /// |image − despeckled(image)|, both polarities (blend modes may clamp
    /// at 0). The grain estimator's residual; GrainMatch normalizes its
    /// synthetic grain with the same metric.
    static func residualMagnitude(of image: CIImage) -> CIImage {
        let median = despeckled(image)
        let up = image.applyingFilter("CISubtractBlendMode",
                                      parameters: [kCIInputBackgroundImageKey: median])
        let down = median.applyingFilter("CISubtractBlendMode",
                                         parameters: [kCIInputBackgroundImageKey: image])
        return up.applyingFilter("CIMaximumCompositing",
                                 parameters: [kCIInputBackgroundImageKey: down])
    }

    static func measure(_ image: CIImage, ciContext: CIContext) -> GrainProfile {
        var extent = image.extent
        if extent.width > maxAnalysisSide || extent.height > maxAnalysisSide {
            let side = CGSize(width: min(extent.width, maxAnalysisSide),
                              height: min(extent.height, maxAnalysisSide))
            extent = CGRect(x: extent.midX - side.width / 2, y: extent.midY - side.height / 2,
                            width: side.width, height: side.height).integral
        }
        let width = Int(extent.width), height = Int(extent.height)
        guard width >= blockSize * 4, height >= blockSize * 4 else {
            return GrainProfile(knots: Array(repeating: 10, count: knotCount), typicalSigma: 10)
        }

        let luminance = encodedLuminance(image.clampedToExtent())
        let median = despeckled(luminance)
        let residual = residualMagnitude(of: luminance)

        func render(_ source: CIImage) -> [Float] {
            var pixels = [Float](repeating: 0, count: width * height)
            pixels.withUnsafeMutableBytes { raw in
                ciContext.render(source, toBitmap: raw.baseAddress!, rowBytes: width * 4,
                                 bounds: extent, format: .Rf, colorSpace: nil)
            }
            return pixels
        }
        let brightness = render(median)
        let magnitude = render(residual)

        // Per block: mean brightness and residual variance.
        let bandCount = knotCount
        var bands = [[Float]](repeating: [], count: bandCount)
        var pixelsPerBand = [Int](repeating: 0, count: bandCount)
        let blocksX = width / blockSize, blocksY = height / blockSize
        for by in 0..<blocksY {
            for bx in 0..<blocksX {
                var sum: Float = 0, sumSquares: Float = 0
                for y in (by * blockSize)..<((by + 1) * blockSize) {
                    let row = y * width
                    for x in (bx * blockSize)..<((bx + 1) * blockSize) {
                        sum += brightness[row + x]
                        let r = magnitude[row + x]
                        sumSquares += r * r
                    }
                }
                let count = Float(blockSize * blockSize)
                let mean = sum / count
                let band = min(max(Int(mean * Float(bandCount)), 0), bandCount - 1)
                bands[band].append(sumSquares / count)
                pixelsPerBand[band] += blockSize * blockSize
            }
        }

        // Noise per band: a low percentile of block variances — the
        // flattest blocks are grain alone, textured/edge blocks sit above.
        var centers: [Double] = [], sigmas: [Double] = [], weights: [Double] = []
        for band in 0..<bandCount where bands[band].count >= 24 {
            let sorted = bands[band].sorted()
            let variance = Double(sorted[sorted.count * 3 / 10])
            centers.append((Double(band) + 0.5) / Double(bandCount))
            sigmas.append(variance.squareRoot() * 255 * medianBias)
            weights.append(Double(pixelsPerBand[band]))
        }
        guard !centers.isEmpty else {
            return GrainProfile(knots: Array(repeating: 10, count: knotCount), typicalSigma: 10)
        }

        // Piecewise-linear through the measured bands, flat beyond them.
        let knots = (0..<knotCount).map { index -> Double in
            let t = Double(index) / Double(knotCount - 1)
            if t <= centers[0] { return sigmas[0] }
            if t >= centers[centers.count - 1] { return sigmas[sigmas.count - 1] }
            let upper = centers.firstIndex { $0 >= t }!
            let lower = upper - 1
            let f = (t - centers[lower]) / (centers[upper] - centers[lower])
            return sigmas[lower] + f * (sigmas[upper] - sigmas[lower])
        }
        let typical = zip(sigmas, weights).map(*).reduce(0, +) / weights.reduce(0, +)
        if ProcessInfo.processInfo.environment["MACOLORIZE_DEBUG_DENOISE"] != nil {
            let bands = zip(centers, sigmas).map { String(format: "%.2f:%.1f", $0, $1) }
            FileHandle.standardError.write(Data(
                "grain profile \(bands.joined(separator: " ")) typical \(typical)\n".utf8))
        }
        return GrainProfile(knots: knots, typicalSigma: typical)
    }

    /// Per-pixel sigma/255 for `image`: the curve evaluated at each pixel's
    /// (slightly smoothed) encoded brightness, times `factor`.
    func noiseMap(for image: CIImage, factor: Double, maxSigma: Double) -> CIImage {
        let brightness = Self.encodedLuminance(image.clampedToExtent())
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 2.0])
        let scaled = knots.map { Float(min($0 * factor, maxSigma) / 255) }
        func vector(_ offset: Int) -> CIVector {
            CIVector(x: CGFloat(scaled[offset]), y: CGFloat(scaled[offset + 1]),
                     z: CGFloat(scaled[offset + 2]), w: CGFloat(scaled[offset + 3]))
        }
        guard let kernel = GrainCurveKernel.kernel,
              let map = kernel.apply(extent: image.extent,
                                     arguments: [brightness, vector(0), vector(4),
                                                 vector(8), vector(12)])
        else {
            return CIImage(color: CIColor(red: 0.04, green: 0.04, blue: 0.04))
                .cropped(to: image.extent)
        }
        return map
    }
}

/// Evaluates the 16-knot grain curve at each pixel's brightness.
enum GrainCurveKernel {
    private static let source = """
    #include <CoreImage/CoreImage.h>
    using namespace metal;

    [[stitchable]] float4 grainCurve(coreimage::sample_t brightness,
                                     float4 k0, float4 k1, float4 k2, float4 k3) {
        float knots[16] = { k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w,
                            k2.x, k2.y, k2.z, k2.w, k3.x, k3.y, k3.z, k3.w };
        float t = clamp(brightness.r, 0.0, 1.0) * 15.0;
        int i = min(int(floor(t)), 14);
        float sigma = mix(knots[i], knots[i + 1], t - float(i));
        return float4(sigma, sigma, sigma, 1.0);
    }
    """

    static let kernel: CIColorKernel? = {
        (try? CIKernel.kernels(withMetalString: source))?
            .compactMap { $0 as? CIColorKernel }.first
    }()
}
