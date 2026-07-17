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
        let stats = try RestoreStage.analyze(input.image, ciContext: context.ciContext)
        let base = (r: max(stats.percentile(99.5, of: stats.samples.r), 1e-3),
                    g: max(stats.percentile(99.5, of: stats.samples.g), 1e-3),
                    b: max(stats.percentile(99.5, of: stats.samples.b), 1e-3))

        var inverted = input.image.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: CGFloat(-1 / base.r), y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: CGFloat(-1 / base.g), z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(-1 / base.b), w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputBiasVector": CIVector(x: 1, y: 1, z: 1, w: 0),
        ])
        inverted = inverted.applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1),
        ])

        var output = input
        output.image = inverted
        return output
    }
}
