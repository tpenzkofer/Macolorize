import Testing
import CoreImage
@testable import MacolorizeCore

struct DustDetectorTests {
    /// Smooth gradient with known bright and dark specks.
    private func makeDustyImage(size: Int, specks: [(x: Int, y: Int, bright: Bool)]) -> CIImage {
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let i = (y * size + x) * 4
                let value = UInt8(60 + (x * 120) / size)
                pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value
                pixels[i + 3] = 255
            }
        }
        for speck in specks {
            for dy in -1...1 {
                for dx in -1...1 {
                    let i = ((speck.y + dy) * size + (speck.x + dx)) * 4
                    let v: UInt8 = speck.bright ? 255 : 5
                    pixels[i] = v; pixels[i + 1] = v; pixels[i + 2] = v
                }
            }
        }
        return CIImage(bitmapData: Data(pixels),
                       bytesPerRow: size * 4,
                       size: CGSize(width: size, height: size),
                       format: .RGBA8,
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    private func maskValues(_ mask: CIImage, size: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: size * size)
        pixels.withUnsafeMutableBytes { raw in
            RenderService.context.render(mask,
                                         toBitmap: raw.baseAddress!,
                                         rowBytes: size,
                                         bounds: CGRect(x: 0, y: 0,
                                                        width: CGFloat(size), height: CGFloat(size)),
                                         format: .R8,
                                         colorSpace: nil)
        }
        return pixels
    }

    @Test func detectsBrightAndDarkSpecks() {
        let size = 256
        // Bitmap y (top-down) — specks at known positions.
        let specks = [(x: 64, y: 64, bright: true), (x: 192, y: 160, bright: false)]
        let image = makeDustyImage(size: size, specks: specks)
        let mask = CleanStage.dustMask(for: image, sensitivity: 0.5)
        let values = maskValues(mask, size: size)

        for speck in specks {
            #expect(values[speck.y * size + speck.x] > 128,
                    "speck at \(speck.x),\(speck.y) not detected")
        }
        // Clean areas stay unmasked (sample a few far-away points).
        #expect(values[16 * size + 16] < 16)
        #expect(values[200 * size + 40] < 16)
        // Overall coverage is small (no runaway false positives).
        let coverage = Double(values.filter { $0 > 128 }.count) / Double(size * size)
        #expect(coverage < 0.02)
    }

    @Test func lowSensitivityIgnoresGradients() {
        let size = 256
        let image = makeDustyImage(size: size, specks: [])
        let mask = CleanStage.dustMask(for: image, sensitivity: 0.2)
        let values = maskValues(mask, size: size)
        #expect(values.allSatisfy { $0 < 16 })
    }
}
