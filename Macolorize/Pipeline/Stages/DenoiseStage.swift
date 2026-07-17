import CoreImage
import CoreML

/// DRUNet blind denoiser, tiled. The strength slider maps to the model's
/// noise-level input (sigma 0–25 covers film grain), so one model covers
/// gentle grain smoothing through heavy cleanup.
public struct DenoiseStage: ProcessingStage {
    public let id = StageID.denoise
    let settings: DenoiseSettings

    static let tileSize = 512
    static let overlap = 96

    public init(settings: DenoiseSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        guard settings.strength > 0 else { return input }
        let model = try await ModelStore.shared.model(.drunet)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let image = input.image
        let clamped = image.clampedToExtent()

        let sigma = try MLMultiArray(shape: [1], dataType: .float32)
        sigma[0] = NSNumber(value: Float(settings.strength * 25.0))

        let grid = TileGrid(extent: image.extent, tileSize: Self.tileSize, overlap: Self.overlap)
        var stitched = CIImage(color: .black).cropped(to: image.extent)
        for tile in grid.tiles {
            try Task.checkCancellation()
            let buffer = try TiledModelRunner.makePixelBuffer(size: Self.tileSize)
            let tileImage = clamped
                .cropped(to: tile.source)
                .transformed(by: .init(translationX: -tile.source.minX, y: -tile.source.minY))
            context.ciContext.render(tileImage, to: buffer,
                                     bounds: CGRect(x: 0, y: 0,
                                                    width: CGFloat(Self.tileSize),
                                                    height: CGFloat(Self.tileSize)),
                                     colorSpace: srgb)
            let prediction = try await model.mlModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: buffer),
                    "sigma": MLFeatureValue(multiArray: sigma),
                ]))
            guard let outBuffer = prediction.featureValue(for: "denoised")?.imageBufferValue else {
                throw TiledModelRunnerError.unexpectedOutput("denoised")
            }
            let piece = CIImage(cvPixelBuffer: outBuffer, options: [.colorSpace: srgb])
                .transformed(by: .init(translationX: tile.source.minX, y: tile.source.minY))
                .cropped(to: tile.core)
            stitched = piece.composited(over: stitched)
        }

        var output = input
        output.image = stitched.cropped(to: image.extent)
        return output
    }
}
