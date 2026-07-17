import CoreImage
import CoreML

/// DDColor AI colorization. The model sees a 512×512 gray version and
/// predicts a/b chroma; the full-resolution luminance is preserved exactly by
/// merging in Lab space — no tiling, no detail loss.
public struct ColorizeStage: ProcessingStage {
    public let id = StageID.colorize
    let settings: ColorizeSettings

    static let modelInputSize = 512

    public init(settings: ColorizeSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        let modelID: ModelID = settings.variant == .artistic ? .ddcolorArtistic : .ddcolor
        let model = try await ModelStore.shared.model(modelID)
        let extent = input.image.extent
        let size = Self.modelInputSize

        // Gray 512×512 input, aspect squashed exactly like the official pipeline.
        let gray = LabColor.luminanceOnly(input.image)
        let toModel = CGAffineTransform(a: CGFloat(size) / extent.width, b: 0,
                                        c: 0, d: CGFloat(size) / extent.height,
                                        tx: -extent.minX * CGFloat(size) / extent.width,
                                        ty: -extent.minY * CGFloat(size) / extent.height)
        let small = gray.transformed(by: toModel)

        let buffer = try TiledModelRunner.makePixelBuffer(size: size)
        context.ciContext.render(small, to: buffer,
                                 bounds: CGRect(x: 0, y: 0, width: CGFloat(size), height: CGFloat(size)),
                                 colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)

        try Task.checkCancellation()
        let prediction = try await model.mlModel.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: buffer),
            ]))
        guard let abArray = prediction.featureValue(for: "ab")?.multiArrayValue else {
            throw TiledModelRunnerError.unexpectedOutput("ab")
        }

        // ab tensor (1,2,512,512) → RGBAf CIImage with R=a, G=b, upsampled to full res.
        let abSmall = try Self.abImage(from: abArray, size: size)
        let fromModel = CGAffineTransform(a: extent.width / CGFloat(size), b: 0,
                                          c: 0, d: extent.height / CGFloat(size),
                                          tx: extent.minX, ty: extent.minY)
        var abFull = abSmall.transformed(by: fromModel)

        // User-painted color hints override the prediction locally.
        abFull = ColorHintRenderer.apply(hints: context.colorHints,
                                         to: abFull,
                                         imageExtent: extent)

        var colorized = try LabColor.merge(original: input.image,
                                           ab: abFull,
                                           chromaScale: settings.saturation)
        if settings.strength < 1 {
            colorized = colorized.applyingFilter("CIDissolveTransition", parameters: [
                kCIInputImageKey: input.image,
                kCIInputTargetImageKey: colorized,
                kCIInputTimeKey: settings.strength,
            ]).cropped(to: extent)
        }

        var output = input
        output.image = colorized
        return output
    }

    /// Packs the (1,2,H,W) ab tensor into an RGBAf bitmap (R=a, G=b).
    static func abImage(from array: MLMultiArray, size: Int) throws -> CIImage {
        let pixelCount = size * size
        var rgba = [Float](repeating: 0, count: pixelCount * 4)

        switch array.dataType {
        case .float32:
            array.withUnsafeBufferPointer(ofType: Float.self) { source in
                for i in 0..<pixelCount {
                    rgba[i * 4] = source[i]                  // a
                    rgba[i * 4 + 1] = source[pixelCount + i] // b
                    rgba[i * 4 + 3] = 1
                }
            }
        case .float16:
            array.withUnsafeBufferPointer(ofType: Float16.self) { source in
                for i in 0..<pixelCount {
                    rgba[i * 4] = Float(source[i])
                    rgba[i * 4 + 1] = Float(source[pixelCount + i])
                    rgba[i * 4 + 3] = 1
                }
            }
        default:
            throw TiledModelRunnerError.unexpectedOutput("ab (unsupported dtype)")
        }

        let data = rgba.withUnsafeBufferPointer { Data(buffer: $0) }
        // colorSpace nil: raw Lab-unit values, no color management.
        // Row order: model output row 0 = image top, matching CIImage's
        // bitmapData convention (top-to-bottom) — no flip needed; verified
        // against a scene with known sky/ground layout.
        return CIImage(bitmapData: data,
                       bytesPerRow: size * 4 * MemoryLayout<Float>.size,
                       size: CGSize(width: size, height: size),
                       format: .RGBAf,
                       colorSpace: nil)
    }
}
