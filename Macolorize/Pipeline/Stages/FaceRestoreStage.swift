import CoreImage
import CoreML

/// GFPGAN face restoration. Faces are found with the Vision framework,
/// similarity-aligned into 512×512 crops, restored, and feather-blended back
/// — the rest of the image is untouched.
public struct FaceRestoreStage: ProcessingStage {
    public let id = StageID.faces
    let settings: FaceSettings

    public init(settings: FaceSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        let faces = try FaceAlign.detectFaces(in: input.image)
        guard !faces.isEmpty else { return input }

        let model = try await ModelStore.shared.model(.gfpgan)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let cropRect = CGRect(x: 0, y: 0, width: FaceAlign.cropSize, height: FaceAlign.cropSize)

        var result = input.image
        for face in faces {
            try Task.checkCancellation()

            let crop = result
                .clampedToExtent()
                .transformed(by: face.toCrop)
                .cropped(to: cropRect)

            let buffer = try TiledModelRunner.makePixelBuffer(size: Int(FaceAlign.cropSize))
            context.ciContext.render(crop, to: buffer, bounds: cropRect, colorSpace: srgb)

            let prediction = try await model.mlModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "face": MLFeatureValue(pixelBuffer: buffer),
                ]))
            guard let outBuffer = prediction.featureValue(for: "restored")?.imageBufferValue else {
                throw TiledModelRunnerError.unexpectedOutput("restored")
            }

            var restored = CIImage(cvPixelBuffer: outBuffer, options: [.colorSpace: srgb])
            if settings.strength < 1 {
                restored = restored.applyingFilter("CIDissolveTransition", parameters: [
                    kCIInputImageKey: crop,
                    kCIInputTargetImageKey: restored,
                    kCIInputTimeKey: settings.strength,
                ]).cropped(to: cropRect)
            }

            // Feathered circular mask in crop space, then everything back to
            // image space through the inverse alignment.
            let mask = Self.featherMask
            let back = face.toCrop.inverted()
            result = restored
                .transformed(by: back)
                .applyingFilter("CIBlendWithMask", parameters: [
                    kCIInputBackgroundImageKey: result,
                    kCIInputMaskImageKey: mask.transformed(by: back),
                ])
                .cropped(to: input.image.extent)
        }

        var output = input
        output.image = result
        return output
    }

    /// White disc (r≈200) blurred 40 px, centered in the 512 crop — covers
    /// the face while fading out before hair/background seams can show.
    nonisolated(unsafe) static let featherMask: CIImage = {
        CIImage(color: .white)
            .cropped(to: CGRect(x: 96, y: 76, width: 320, height: 380))
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 42])
            .cropped(to: CGRect(x: 0, y: 0, width: 512, height: 512))
    }()
}
