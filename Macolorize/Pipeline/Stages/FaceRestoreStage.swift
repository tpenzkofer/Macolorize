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

    /// Elliptical gradient mask covering the face region of the 512 crop.
    /// Deliberately not a blurred rectangle: straight mask edges ghost as
    /// faint horizontal/vertical lines in flat backgrounds; a curved falloff
    /// doesn't read as a line.
    nonisolated(unsafe) static let featherMask: CIImage = {
        let circle = CIImage(color: .white).applyingFilter("CIRadialGradient", parameters: [
            "inputCenter": CIVector(x: 256, y: 256),
            "inputRadius0": 130,
            "inputRadius1": 245,
            "inputColor0": CIColor.white,
            "inputColor1": CIColor.black,
        ])
        // Stretch vertically into a face-shaped ellipse, recenter on the
        // eyes/mouth region of the FFHQ template.
        return circle
            .transformed(by: CGAffineTransform(translationX: -256, y: -256)
                .concatenating(.init(scaleX: 1.0, y: 1.25))
                .concatenating(.init(translationX: 256, y: 266)))
            .cropped(to: CGRect(x: 0, y: 0, width: 512, height: 512))
    }()
}
