import Testing
import CoreImage
@testable import MacolorizeCore

/// The dust detector must fire on sparse extreme outliers (dust) and stay
/// quiet on film grain — on a grainy scan a fixed threshold flagged ~half
/// the frame as "dust" and LaMa repainted it into smears (issue #1).
struct CleanStageTests {
    private static let size = 800
    /// Dust speck centers (x, y from top), placed away from each other.
    private static let specks: [(Int, Int)] = [
        (100, 120), (300, 80), (650, 200), (180, 400),
        (500, 450), (720, 600), (90, 700), (400, 650),
    ]

    /// Mid-gray image with ±5% linear grain and 3×3 white dust specks.
    private func makeGrainyDustyImage() -> CIImage {
        let size = Self.size
        var seed: UInt64 = 0xDEADBEEFCAFE1234
        func rand() -> Float {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Float(seed % 10_000) / 10_000
        }
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let i = (y * size + x) * 4
                let value = UInt8(max(0, min(255, (0.45 + (rand() - 0.5) * 0.1) * 255)))
                pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value
                pixels[i + 3] = 255
            }
        }
        for (cx, cy) in Self.specks {
            for dy in -1...1 {
                for dx in -1...1 {
                    let i = ((cy + dy) * size + (cx + dx)) * 4
                    pixels[i] = 255; pixels[i + 1] = 255; pixels[i + 2] = 255
                }
            }
        }
        return CIImage(bitmapData: Data(pixels), bytesPerRow: size * 4,
                       size: CGSize(width: size, height: size),
                       format: .RGBA8,
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    @Test func detectsDustButNotGrain() throws {
        let image = makeGrainyDustyImage()
        let mask = CleanStage.dustMask(for: image, sensitivity: 0.5)

        let size = Self.size
        var pixels = [UInt8](repeating: 0, count: size * size)
        pixels.withUnsafeMutableBytes { raw in
            RenderService.context.render(mask, toBitmap: raw.baseAddress!,
                                         rowBytes: size,
                                         bounds: CGRect(x: 0, y: 0, width: size, height: size),
                                         format: .R8,
                                         colorSpace: nil)
        }

        // Every dust speck is flagged (bitmap row 0 = image top after render).
        for (cx, cy) in Self.specks {
            #expect(pixels[cy * size + cx] > 128, "speck at (\(cx), \(cy)) missed")
        }

        // The grain is NOT carpet-flagged: the specks (3×3, dilated by 3 →
        // ≤ ~13×13 each) bound the legitimate coverage well under 2%.
        let coverage = Double(pixels.count { $0 > 128 }) / Double(size * size)
        #expect(coverage < 0.02, "mask covers \(coverage * 100)% — detector is flagging grain")
    }
}
