import CoreImage

/// Lab helpers implementing skimage-exact math (sRGB, D65, L ∈ [0,100]) —
/// what DDColor was trained on. Core Image's built-in CIConvertRGBtoLab uses
/// a different convention (validated: up to ~11 units off in chroma), so the
/// merge runs through a custom CIColorKernel compiled at runtime.
public enum LabColor {
    /// Rec.709 relative luminance in linear working space, replicated to RGB.
    /// Equivalent to DDColor's official preprocessing (lab2rgb(L,0,0)): a gray
    /// image with the source's luminance.
    public static func luminanceOnly(_ image: CIImage) -> CIImage {
        image.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputGVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputBVector": CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
        ])
    }

    /// Metal CI kernel: keeps the original's L (from linear luminance), takes
    /// a/b from the model, converts Lab→XYZ(D65)→linear sRGB. Samples arrive
    /// in the linear working space; output is linear as well.
    private static let mergeKernelSource = """
    #include <CoreImage/CoreImage.h>
    using namespace metal;

    [[stitchable]] float4 labMerge(coreimage::sample_t original,
                                   coreimage::sample_t ab,
                                   float chromaScale) {
        const float eps = 216.0 / 24389.0;   // 0.008856
        const float kappa = 24389.0 / 27.0;  // 903.3
        // D65 reference white
        const float Xn = 0.95047, Yn = 1.0, Zn = 1.08883;

        float Y = dot(original.rgb, float3(0.2126, 0.7152, 0.0722));
        Y = max(Y / Yn, 0.0);
        float fy = (Y > eps) ? pow(Y, 1.0/3.0) : (kappa * Y + 16.0) / 116.0;
        float L = 116.0 * fy - 16.0;

        float a = ab.r * chromaScale;
        float b = ab.g * chromaScale;

        float fx = fy + a / 500.0;
        float fz = fy - b / 200.0;
        float fx3 = fx * fx * fx;
        float fz3 = fz * fz * fz;
        float xr = (fx3 > eps) ? fx3 : (116.0 * fx - 16.0) / kappa;
        float yr = (L > kappa * eps) ? pow((L + 16.0) / 116.0, 3.0) : L / kappa;
        float zr = (fz3 > eps) ? fz3 : (116.0 * fz - 16.0) / kappa;

        float X = xr * Xn, Y2 = yr * Yn, Z = zr * Zn;
        // XYZ (D65) -> linear sRGB
        float3 rgb;
        rgb.r =  3.2404542 * X - 1.5371385 * Y2 - 0.4985314 * Z;
        rgb.g = -0.9692660 * X + 1.8760108 * Y2 + 0.0415560 * Z;
        rgb.b =  0.0556434 * X - 0.2040259 * Y2 + 1.0572252 * Z;
        return float4(clamp(rgb, 0.0, 1.0), original.a);
    }
    """

    nonisolated(unsafe) private static let mergeKernel: CIColorKernel? = {
        do {
            let kernels = try CIKernel.kernels(withMetalString: mergeKernelSource)
            return kernels.compactMap { $0 as? CIColorKernel }.first
        } catch {
            assertionFailure("Lab merge kernel failed to compile: \(error)")
            return nil
        }
    }()

    public enum LabError: Error {
        case kernelUnavailable
    }

    /// Combines the original's luminance with predicted a/b chroma.
    /// - Parameters:
    ///   - original: full-resolution source (linear working space)
    ///   - ab: image whose R/G channels carry a/b in Lab units (upsampled to
    ///     the original's extent)
    ///   - chromaScale: saturation multiplier on a/b
    public static func merge(original: CIImage,
                             ab: CIImage,
                             chromaScale: Double = 1.0) throws -> CIImage {
        guard let kernel = mergeKernel else { throw LabError.kernelUnavailable }
        guard let output = kernel.apply(extent: original.extent,
                                        arguments: [original, ab, Float(chromaScale)]) else {
            throw LabError.kernelUnavailable
        }
        return output
    }
}
