import Testing
import CoreImage
@testable import MacolorizeCore

struct ColorHintTests {
    @Test func srgbToLabMatchesGoldenValues() {
        // skimage golden: pure red → (53.241, 80.092, 67.203)
        let red = ColorHint.srgbToLab(red: 1, green: 0, blue: 0)
        #expect(abs(red.l - 53.241) < 0.05)
        #expect(abs(red.ab.a - 80.092) < 0.05)
        #expect(abs(red.ab.b - 67.203) < 0.05)

        let gray = ColorHint.srgbToLab(red: 0.5, green: 0.5, blue: 0.5)
        #expect(abs(gray.ab.a) < 0.05)
        #expect(abs(gray.ab.b) < 0.05)
    }

    @Test func hintOverridesPredictionInsideStrokeOnly() {
        let size = 128
        // Neutral prediction: ab = (0, 0) everywhere.
        var rgba = [Float](repeating: 0, count: size * size * 4)
        for i in 0..<(size * size) { rgba[i * 4 + 3] = 1 }
        let data = rgba.withUnsafeBufferPointer { Data(buffer: $0) }
        let predicted = CIImage(bitmapData: data,
                                bytesPerRow: size * 4 * MemoryLayout<Float>.size,
                                size: CGSize(width: size, height: size),
                                format: .RGBAf, colorSpace: nil)

        let hint = ColorHint(
            stroke: BrushStroke(points: [CGPoint(x: 0.5, y: 0.5)], radius: 0.1),
            red: 1, green: 0, blue: 0)
        let result = ColorHintRenderer.apply(hints: [hint],
                                             to: predicted,
                                             imageExtent: predicted.extent)

        func sample(_ x: Int, _ y: Int) -> (a: Float, b: Float) {
            var pixel = [Float](repeating: 0, count: 4)
            pixel.withUnsafeMutableBytes { raw in
                RenderService.context.render(result,
                                             toBitmap: raw.baseAddress!,
                                             rowBytes: 16,
                                             bounds: CGRect(x: CGFloat(x), y: CGFloat(y),
                                                            width: 1, height: 1),
                                             format: .RGBAf,
                                             colorSpace: nil)
            }
            return (pixel[0], pixel[1])
        }

        // Center of the stroke: the hint's red chroma.
        let center = sample(size / 2, size / 2)
        #expect(abs(center.a - 80.09) < 4.0, "a at center: \(center.a)")
        #expect(abs(center.b - 67.20) < 4.0, "b at center: \(center.b)")
        // Far corner: untouched prediction.
        let corner = sample(4, 4)
        #expect(abs(corner.a) < 1.0)
        #expect(abs(corner.b) < 1.0)
    }
}
