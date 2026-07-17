import CoreImage
import CoreML

/// Automatic dust & scratch removal. Detection is classical (no pretrained
/// dust U-Net is publicly available): small outliers versus an iterated
/// 3×3 median — catching both bright dust (reflective scans) and dark dust
/// (transparency scans) — dilated into a mask, then healed by LaMa exactly
/// like manual retouch strokes.
public struct CleanStage: ProcessingStage {
    public let id = StageID.clean
    let settings: CleanSettings

    public init(settings: CleanSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        var mask = Self.dustMask(for: input.image, sensitivity: settings.sensitivity)

        // Long structured scratches: ML detector (classical median-diff
        // misses anything wider than a few pixels).
        if settings.detectScratches, ModelStore.isAvailable(.scratchnet) {
            let scratchMask = try await Self.scratchMask(for: input.image, context: context)
            mask = mask.applyingFilter("CIMaximumCompositing", parameters: [
                kCIInputBackgroundImageKey: scratchMask,
            ]).cropped(to: input.image.extent)
        }

        var output = input
        output.image = try await InpaintStage.apply(to: input.image,
                                                    mask: mask,
                                                    ciContext: context.ciContext)
        return output
    }

    /// Tiled BOPBTL scratch U-Net over the image luminance; tile masks are
    /// merged by per-pixel max.
    static func scratchMask(for image: CIImage, context: RenderContext) async throws -> CIImage {
        let model = try await ModelStore.shared.model(.scratchnet)
        let tileSize = 1024
        // The detector was trained on gamma-encoded photos; the working
        // space is linear, so encode before rendering the gray tiles.
        let gray = LabColor.luminanceOnly(image.clampedToExtent())
            .applyingFilter("CIGammaAdjust", parameters: ["inputPower": 1.0 / 2.2])

        let grid = TileGrid(extent: image.extent, tileSize: tileSize, overlap: 128)
        var merged = CIImage(color: .black).cropped(to: image.extent)
        for tile in grid.tiles {
            try Task.checkCancellation()
            let buffer = try InpaintStage.makeGrayPixelBuffer(size: tileSize)
            let tileImage = gray
                .cropped(to: tile.source)
                .transformed(by: .init(translationX: -tile.source.minX, y: -tile.source.minY))
            context.ciContext.render(tileImage, to: buffer,
                                     bounds: CGRect(x: 0, y: 0,
                                                    width: CGFloat(tileSize),
                                                    height: CGFloat(tileSize)),
                                     colorSpace: nil)
            let prediction = try await model.mlModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: buffer),
                ]))
            guard let outBuffer = prediction.featureValue(for: "mask")?.imageBufferValue else {
                throw TiledModelRunnerError.unexpectedOutput("mask")
            }
            let piece = CIImage(cvPixelBuffer: outBuffer)
                .transformed(by: .init(translationX: tile.source.minX, y: tile.source.minY))
                .cropped(to: tile.core)
            merged = piece.applyingFilter("CIMaximumCompositing", parameters: [
                kCIInputBackgroundImageKey: merged,
            ]).cropped(to: image.extent)
        }

        // Binarize at 40% probability and grow slightly for LaMa context.
        return merged
            .applyingFilter("CIColorThreshold", parameters: ["inputThreshold": 0.4])
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 3])
            .cropped(to: image.extent)
    }

    /// White-on-black mask of probable dust/scratch pixels.
    /// sensitivity 0 → only extreme outliers; 1 → aggressive.
    public static func dustMask(for image: CIImage, sensitivity: Double) -> CIImage {
        // Clamp so the median filter sees sane values past the image edges
        // (otherwise the border rows light up as false positives).
        let luminance = LabColor.luminanceOnly(image.clampedToExtent())

        // Iterated 3×3 median ≈ despeckled reference (removes up to ~4 px specks).
        var median = luminance
        for _ in 0..<4 {
            median = median.applyingFilter("CIMedianFilter")
        }

        // |luminance − median| in linear light, both polarities.
        let bright = luminance.applyingFilter("CISubtractBlendMode", parameters: [
            kCIInputBackgroundImageKey: median,
        ])
        let dark = median.applyingFilter("CISubtractBlendMode", parameters: [
            kCIInputBackgroundImageKey: luminance,
        ])
        let difference = bright.applyingFilter("CIMaximumCompositing", parameters: [
            kCIInputBackgroundImageKey: dark,
        ])

        // Linear-light threshold: 0.30 (conservative) … 0.04 (aggressive).
        let threshold = 0.30 - 0.26 * sensitivity.clamped(to: 0...1)
        let binary = difference.applyingFilter("CIColorThreshold", parameters: [
            "inputThreshold": threshold,
        ])

        // Grow the mask a touch so LaMa sees past speck edges.
        return binary
            .applyingFilter("CIMorphologyMaximum", parameters: [kCIInputRadiusKey: 3])
            .cropped(to: image.extent)
    }
}
