import CoreImage

/// Materializes a lazy Core Image graph into memory so repeated renders
/// don't recompute it. Used to cache the expensive pipeline prefix
/// (negative/clean/heals): rendered once in linear half-float, then every
/// preview update and the export read the identical pixels.
public enum ImageRasterizer {
    public static func materialize(_ image: CIImage, ciContext: CIContext) -> CIImage? {
        let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
        guard let cgImage = ciContext.createCGImage(image,
                                                    from: image.extent,
                                                    format: .RGBAh,
                                                    colorSpace: linear) else {
            return nil
        }
        // The CGImage carries the linear color space, so values round-trip
        // exactly into the (linear) working space.
        return CIImage(cgImage: cgImage)
            .transformed(by: .init(translationX: image.extent.minX,
                                   y: image.extent.minY))
    }

    /// Prefiltered proxy of a (materialized) image for interactive preview.
    public static func proxy(of image: CIImage, maxDimension: CGFloat) -> CIImage {
        let extent = image.extent
        let scale = min(1, maxDimension / max(extent.width, extent.height))
        guard scale < 1 else { return image }
        return image
            .clampedToExtent()
            .applyingFilter("CILanczosScaleTransform", parameters: [
                kCIInputScaleKey: scale,
                kCIInputAspectRatioKey: 1.0,
            ])
            .cropped(to: CGRect(x: extent.minX * scale,
                                y: extent.minY * scale,
                                width: (extent.width * scale).rounded(.down),
                                height: (extent.height * scale).rounded(.down)))
    }
}
