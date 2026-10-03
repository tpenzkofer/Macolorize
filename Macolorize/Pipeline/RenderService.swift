import CoreImage
import Metal

/// The app's single shared Metal device and CIContext.
/// CIContext is documented thread-safe; never create per-frame contexts.
public enum RenderService {
    public static let device: MTLDevice = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Macolorize requires Metal (no GPU device available — e.g. a sandboxed or headless environment).")
        }
        return device
    }()

    public static let context: CIContext = CIContext(
        mtlDevice: device,
        options: [
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
            .workingFormat: CIFormat.RGBAh,
            .cacheIntermediates: false,
            .name: "MacolorizeCI",
        ])
}
