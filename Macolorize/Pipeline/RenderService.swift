import CoreImage
import Metal

/// The app's single shared Metal device and CIContext.
/// CIContext is documented thread-safe; never create per-frame contexts.
public enum RenderService {
    public static let device: MTLDevice = MTLCreateSystemDefaultDevice()!

    public static let context: CIContext = CIContext(
        mtlDevice: device,
        options: [
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
            .workingFormat: CIFormat.RGBAh,
            .cacheIntermediates: false,
            .name: "MacolorizeCI",
        ])
}
