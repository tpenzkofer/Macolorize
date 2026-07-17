import CoreImage
import CoreML

/// LaMa inpainting over masked regions. Only 800×800 tiles covering the
/// mask's bounding box are processed (never the whole image); the inpainted
/// pixels are composited back strictly inside the mask.
public enum InpaintStage {
    public static let tileSize = 800
    static let tileOverlap = 200

    public static func apply(to image: CIImage,
                             mask: CIImage,
                             ciContext: CIContext) async throws -> CIImage {
        // Mask rendered at whatever resolution — scale it to the image.
        let scaleX = image.extent.width / mask.extent.width
        let scaleY = image.extent.height / mask.extent.height
        let scaledMask = mask
            .transformed(by: .init(scaleX: scaleX, y: scaleY))
            .transformed(by: .init(translationX: image.extent.minX - mask.extent.minX * scaleX,
                                   y: image.extent.minY - mask.extent.minY * scaleY))

        guard let bbox = maskBoundingBox(scaledMask, ciContext: ciContext) else {
            return image
        }

        let model = try await ModelStore.shared.model(.lama)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        let clampedImage = image.clampedToExtent()
        let clampedMask = scaledMask.clampedToExtent()

        // Tile region: the bbox grown to at least one tile, clamped roughly
        // to the image but allowed to overhang (sampling is clamped).
        let region = tileRegion(for: bbox, imageExtent: image.extent)
        let grid = TileGrid(extent: region, tileSize: tileSize, overlap: tileOverlap)

        var result = image
        for tile in grid.tiles {
            try Task.checkCancellation()

            // Skip tiles whose core contains no mask.
            guard let tileMaskBounds = maskBoundingBox(
                scaledMask.cropped(to: tile.core.intersection(image.extent)),
                ciContext: ciContext), !tileMaskBounds.isEmpty else { continue }

            let toOrigin = CGAffineTransform(translationX: -tile.source.minX,
                                             y: -tile.source.minY)
            let imageBuffer = try TiledModelRunner.makePixelBuffer(size: tileSize)
            ciContext.render(clampedImage.cropped(to: tile.source).transformed(by: toOrigin),
                             to: imageBuffer,
                             bounds: CGRect(x: 0, y: 0, width: 800, height: 800),
                             colorSpace: srgb)
            let maskBuffer = try makeGrayPixelBuffer(size: tileSize)
            // Gray renders must not specify a color space (DeviceGray is not
            // a valid CI render target space and yields zeros).
            ciContext.render(clampedMask.cropped(to: tile.source).transformed(by: toOrigin),
                             to: maskBuffer,
                             bounds: CGRect(x: 0, y: 0, width: 800, height: 800),
                             colorSpace: nil)

            let prediction = try await model.mlModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "image": MLFeatureValue(pixelBuffer: imageBuffer),
                    "mask": MLFeatureValue(pixelBuffer: maskBuffer),
                ]))
            guard let outBuffer = prediction.featureValue(for: "output")?.imageBufferValue else {
                throw TiledModelRunnerError.unexpectedOutput("output")
            }

            let inpainted = CIImage(cvPixelBuffer: outBuffer, options: [.colorSpace: srgb])
                .transformed(by: .init(translationX: tile.source.minX, y: tile.source.minY))
                .cropped(to: tile.core)

            // Composite only where the mask is set (slightly feathered).
            let compositeMask = scaledMask
                .cropped(to: tile.core)
                .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 1.5])
                .cropped(to: tile.core)
            result = inpainted.applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: result,
                kCIInputMaskImageKey: compositeMask,
            ]).cropped(to: image.extent)
        }
        return result
    }

    /// Expands the mask bbox to at least one tile, preferring to stay inside
    /// the image; falls back to overhang for images smaller than a tile.
    static func tileRegion(for bbox: CGRect, imageExtent: CGRect) -> CGRect {
        let margin: CGFloat = 32
        var region = bbox.insetBy(dx: -margin, dy: -margin)
        let side = CGFloat(tileSize)

        if region.width < side {
            region.origin.x -= (side - region.width) / 2
            region.size.width = side
        }
        if region.height < side {
            region.origin.y -= (side - region.height) / 2
            region.size.height = side
        }
        // Shift back inside the image where possible.
        if region.maxX > imageExtent.maxX { region.origin.x = imageExtent.maxX - region.width }
        if region.minX < imageExtent.minX { region.origin.x = imageExtent.minX }
        if region.maxY > imageExtent.maxY { region.origin.y = imageExtent.maxY - region.height }
        if region.minY < imageExtent.minY { region.origin.y = imageExtent.minY }
        return region
    }

    /// Bounding box of nonzero mask pixels, or nil if the mask is empty.
    /// Scans a downsampled render (≤512 px long edge).
    static func maskBoundingBox(_ mask: CIImage, ciContext: CIContext) -> CGRect? {
        let extent = mask.extent
        guard extent.width >= 1, extent.height >= 1, !extent.isInfinite else { return nil }
        let downScale = min(1, 512 / max(extent.width, extent.height))
        let width = max(Int(extent.width * downScale), 1)
        let height = max(Int(extent.height * downScale), 1)

        var pixels = [UInt8](repeating: 0, count: width * height)
        pixels.withUnsafeMutableBytes { raw in
            ciContext.render(mask.transformed(by: .init(scaleX: downScale, y: downScale)),
                             toBitmap: raw.baseAddress!,
                             rowBytes: width,
                             bounds: CGRect(x: extent.minX * downScale,
                                            y: extent.minY * downScale,
                                            width: CGFloat(width), height: CGFloat(height)),
                             format: .R8,
                             colorSpace: nil)
        }

        var minX = width, maxX = -1, minY = height, maxY = -1
        for y in 0..<height {
            for x in 0..<width where pixels[y * width + x] > 8 {
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        guard maxX >= 0 else { return nil }

        // Bitmap row 0 is the render bounds' TOP; convert back to CI coords.
        let inv = 1 / downScale
        let rect = CGRect(x: extent.minX + CGFloat(minX) * inv,
                          y: extent.minY + CGFloat(height - 1 - maxY) * inv,
                          width: CGFloat(maxX - minX + 1) * inv,
                          height: CGFloat(maxY - minY + 1) * inv)
        return rect.insetBy(dx: -inv, dy: -inv).intersection(extent)
    }

    static func makeGrayPixelBuffer(size: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                         kCVPixelFormatType_OneComponent8,
                                         [kCVPixelBufferCGImageCompatibilityKey: true,
                                          kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary,
                                         &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw TiledModelRunnerError.pixelBufferAllocation
        }
        return buffer
    }
}
