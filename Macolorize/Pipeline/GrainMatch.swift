import CoreImage
import CoreImage.CIFilterBuiltins

/// Re-grains healed regions. LaMa fills masked areas with smooth, grain-free
/// pixels, so on a grainy scan every heal reads as a soft smudge (issue #1:
/// "dust fixes continue to blur the image"). This adds synthetic grain
/// matched to the scan's own grain-vs-brightness profile, inside the mask
/// only, so heals blend into the surrounding film texture.
enum GrainMatch {
    static func apply(healed: CIImage, original: CIImage, mask: CIImage,
                      ciContext: CIContext) -> CIImage {
        guard let kernel = GrainAddKernel.kernel else { return healed }
        let extent = healed.extent
        let profile = GrainProfile.measure(original, ciContext: ciContext)
        let sigmaMap = profile.noiseMap(for: healed, factor: 1, maxSigma: 255)

        // White noise: blurring it to mimic grain correlation shifts energy
        // to coarse scales the fine-scale estimator does not see, so the
        // normalization below inflated it into visible mottling.
        let tile = CIFilter.affineTile()
        tile.inputImage = noiseTile
        tile.transform = .identity
        let noise = tile.outputImage!.cropped(to: extent)

        // Scale the noise so the grain estimator measures exactly the
        // profile's sigma in it — same metric, same look.
        let unit = 1 / max(residualRMS(of: noise, ciContext: ciContext), 1e-4)

        let encoded = healed.applyingFilter("CILinearToSRGBToneCurve")
        guard let grained = kernel.apply(extent: extent,
                                         arguments: [encoded, noise, sigmaMap, Float(unit)])
        else { return healed }
        let grainedLinear = grained.applyingFilter("CISRGBToneCurveToLinear")

        let feathered = mask
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 1.5])
            .cropped(to: extent)
        return grainedLinear.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: healed,
            kCIInputMaskImageKey: feathered,
        ]).cropped(to: extent)
    }

    static let tileSize = 512

    /// Seeded Gaussian noise (0.5 ± 0.1, opaque, single channel): fixed, so
    /// preview and export get identical grain. Generated on the CPU —
    /// CIRandomGenerator's random alpha corrupts every statistic taken on it.
    static let noiseTile: CIImage = {
        var state: UInt64 = 0x5EED_6A41
        func uniform() -> Double {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
        }
        var pixels = [Float](repeating: 0, count: tileSize * tileSize * 4)
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let u1 = max(uniform(), .leastNonzeroMagnitude)
            let gaussian = (-2 * log(u1)).squareRoot() * cos(2 * .pi * uniform())
            let value = Float(0.5 + 0.1 * gaussian)
            pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value
            pixels[i + 3] = 1
        }
        let data = pixels.withUnsafeBytes { Data($0) }
        return CIImage(bitmapData: data, bytesPerRow: tileSize * 16,
                       size: CGSize(width: tileSize, height: tileSize),
                       format: .RGBAf, colorSpace: nil)
    }()

    /// RMS of the grain-estimator residual over a 512² patch of `noise`.
    static func residualRMS(of noise: CIImage, ciContext: CIContext) -> Double {
        let patch = CGRect(x: noise.extent.minX, y: noise.extent.minY,
                           width: min(noise.extent.width, 512),
                           height: min(noise.extent.height, 512))
        let residual = GrainProfile.residualMagnitude(of: noise.clampedToExtent())
            .cropped(to: patch)
        let stats = CleanStage.meanAndSigma(of: residual, ciContext: ciContext)
        return (stats.sigma * stats.sigma + stats.mean * stats.mean).squareRoot()
    }
}

/// encoded.rgb + (noise − 0.5) · unit · sigma (sigma map holds sigma/255).
enum GrainAddKernel {
    private static let source = """
    #include <CoreImage/CoreImage.h>
    using namespace metal;

    [[stitchable]] float4 grainAdd(coreimage::sample_t encoded,
                                   coreimage::sample_t noise,
                                   coreimage::sample_t sigma,
                                   float unit) {
        return float4(encoded.rgb + (noise.r - 0.5) * unit * sigma.r, encoded.a);
    }
    """

    static let kernel: CIColorKernel? = {
        (try? CIKernel.kernels(withMetalString: source))?
            .compactMap { $0 as? CIColorKernel }.first
    }()
}
