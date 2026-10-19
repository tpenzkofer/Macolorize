import CoreImage
import CoreML

/// DRUNet blind denoiser, tiled with feathered stitching. The strength
/// slider maps to the model's noise-level input across its full trained
/// range (sigma 0–50): heavy archive grain genuinely needs the upper
/// half — at 25 the smoothing barely registered on a grainy cinema scan
/// (issue #1 follow-up), and the old pipeline masked that because LaMa
/// was accidentally repainting the grain fields.
public struct DenoiseStage: ProcessingStage {
    public let id = StageID.denoise
    let settings: DenoiseSettings

    public init(settings: DenoiseSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        guard settings.strength > 0 else { return input }
        let model = try await ModelStore.shared.model(.drunet)

        let sigmaValue = Float(settings.strength * 50.0)
        let runner = TiledModelRunner(tileSize: 512, overlap: 128, outputScale: 1,
                                      inputName: "image", outputName: "denoised")
        let result = try await runner.run(model: model,
                                          over: input.image,
                                          ciContext: context.ciContext) { buffer in
            let sigma = try MLMultiArray(shape: [1], dataType: .float32)
            sigma[0] = NSNumber(value: sigmaValue)
            return try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffer),
                "sigma": MLFeatureValue(multiArray: sigma),
            ])
        }

        var output = input
        output.image = result
        return output
    }
}
