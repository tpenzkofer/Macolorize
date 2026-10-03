import CoreImage

/// Color-negative inversion with orange-mask removal. The film base (orange
/// mask + fog) is estimated as the per-channel bright percentile — unexposed
/// film is the brightest thing on a negative — each channel is normalized by
/// its base, then inverted in linear light:  out = 1 − x/base.
/// Combine with Restore for black/white points and final color balance.
public struct NegativeStage: ProcessingStage {
    public let id = StageID.negative
    let settings: NegativeSettings

    public init(settings: NegativeSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        // Full-resolution statistics so preview and export invert with the
        // same film-base estimate (see PipelineImage.statsSource).
        let stats = try RestoreStage.analyze(input.statsSource, ciContext: context.ciContext)
        let base = (r: max(stats.percentile(99.5, of: stats.samples.r), 1e-3),
                    g: max(stats.percentile(99.5, of: stats.samples.g), 1e-3),
                    b: max(stats.percentile(99.5, of: stats.samples.b), 1e-3))

        func invert(_ source: CIImage) -> CIImage {
            source.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: CGFloat(-1 / base.r), y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: CGFloat(-1 / base.g), z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(-1 / base.b), w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: 1, y: 1, z: 1, w: 0),
            ]).applyingFilter("CIColorClamp", parameters: [
                "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1),
            ])
        }

        var output = input
        output.image = invert(input.image)
        output.statsSource = invert(input.statsSource)
        return output
    }
}
