import CoreImage
import CoreML

/// Real-ESRGAN super-resolution, tiled. Runs at native resolution on export.
/// In preview mode the upscaled result is scaled back to the input size so the
/// sharpening character shows without changing canvas geometry.
public struct SharpenStage: ProcessingStage {
    public let id = StageID.sharpen
    let settings: SharpenSettings

    public init(settings: SharpenSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        let modelID: ModelID = settings.model == .x2plus ? .realesrganX2 : .realesrganX4
        let outputScale = settings.model == .x2plus ? 2 : 4
        let model = try await ModelStore.shared.model(modelID)

        let runner = TiledModelRunner(tileSize: 512, overlap: 128, outputScale: outputScale)
        var result = try await runner.run(model: model,
                                          over: input.image,
                                          ciContext: context.ciContext)

        if context.mode == .preview {
            // Keep preview geometry stable: show the effect at input size.
            let inverse = 1.0 / CGFloat(outputScale)
            result = result
                .transformed(by: .init(scaleX: inverse, y: inverse))
                .cropped(to: input.image.extent)
        }

        var output = input
        output.image = result
        return output
    }
}
