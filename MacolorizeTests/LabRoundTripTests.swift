import Testing
import CoreImage
@testable import MacolorizeCore

/// Validates the custom Lab merge kernel against skimage golden values
/// (skimage.color.rgb2lab, sRGB, D65) — the convention DDColor was trained on.
/// Note: Core Image's built-in CIConvertRGBtoLab was tested and rejected
/// (up to ~11 Lab units off in chroma; different white-point convention).
struct LabRoundTripTests {
    static let golden: [(rgb: (Double, Double, Double), lab: (Double, Double, Double))] = [
        (rgb: (1.0, 0.0, 0.0), lab: (53.241, 80.092, 67.203)),
        (rgb: (0.0, 1.0, 0.0), lab: (87.735, -86.183, 83.180)),
        (rgb: (0.0, 0.0, 1.0), lab: (32.296, 79.186, -107.857)),
        (rgb: (0.5, 0.5, 0.5), lab: (53.389, -0.001, 0.003)),
        (rgb: (1.0, 1.0, 1.0), lab: (100.000, -0.002, 0.005)),
        (rgb: (0.8, 0.6, 0.2), lab: (66.414, 9.497, 57.944)),
        (rgb: (0.2, 0.4, 0.9), lab: (46.828, 28.177, -69.451)),
        (rgb: (0.6, 0.2, 0.5), lab: (39.048, 50.832, -21.280)),
    ]

    private func makeColor(_ rgb: (Double, Double, Double)) -> CIImage {
        let color = CIColor(red: rgb.0, green: rgb.1, blue: rgb.2,
                            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)!
        return CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
    }

    private func makeAB(_ a: Double, _ b: Double) -> CIImage {
        var rgba = [Float](repeating: 0, count: 4 * 4)
        for i in 0..<4 {
            rgba[i * 4] = Float(a)
            rgba[i * 4 + 1] = Float(b)
            rgba[i * 4 + 3] = 1
        }
        let data = rgba.withUnsafeBufferPointer { Data(buffer: $0) }
        return CIImage(bitmapData: data, bytesPerRow: 32,
                       size: CGSize(width: 2, height: 2), format: .RGBAf, colorSpace: nil)
    }

    private func renderSRGB(_ image: CIImage) -> [Float] {
        var pixel = [Float](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { raw in
            RenderService.context.render(image,
                                         toBitmap: raw.baseAddress!,
                                         rowBytes: 16,
                                         bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                                         format: .RGBAf,
                                         colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        }
        return pixel
    }

    /// merge(color, golden ab) must reconstruct the color: the kernel derives
    /// L from the color's luminance; with the color's own a/b the Lab→RGB
    /// inverse must land back on the source color. Exercises the whole chain.
    @Test func mergeReconstructsGoldenColors() throws {
        for entry in Self.golden {
            let merged = try LabColor.merge(original: makeColor(entry.rgb),
                                            ab: makeAB(entry.lab.1, entry.lab.2))
            let pixel = renderSRGB(merged)
            #expect(abs(Double(pixel[0]) - entry.rgb.0) < 0.015,
                    "R for \(entry.rgb): \(pixel[0])")
            #expect(abs(Double(pixel[1]) - entry.rgb.1) < 0.015,
                    "G for \(entry.rgb): \(pixel[1])")
            #expect(abs(Double(pixel[2]) - entry.rgb.2) < 0.015,
                    "B for \(entry.rgb): \(pixel[2])")
        }
    }

    /// Merging with zero chroma must produce the pure luminance gray.
    @Test func zeroChromaGivesNeutralGray() throws {
        let merged = try LabColor.merge(original: makeColor((0.8, 0.3, 0.5)),
                                        ab: makeAB(0, 0))
        let pixel = renderSRGB(merged)
        #expect(abs(pixel[0] - pixel[1]) < 0.005)
        #expect(abs(pixel[1] - pixel[2]) < 0.005)
    }

    /// The merge must preserve the original's luminance exactly (full-detail
    /// L channel is the whole point of the Lab merge).
    @Test func luminancePreserved() throws {
        let original = makeColor((0.35, 0.55, 0.25))
        let merged = try LabColor.merge(original: original, ab: makeAB(40, -30))
        func luminance(_ image: CIImage) -> Float {
            let gray = LabColor.luminanceOnly(image)
            return renderSRGB(gray)[0]
        }
        #expect(abs(luminance(merged) - luminance(original)) < 0.01)
    }
}
