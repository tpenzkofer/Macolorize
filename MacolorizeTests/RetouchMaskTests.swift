import Testing
import CoreGraphics
import CoreImage
@testable import MacolorizeCore

struct RetouchMaskTests {
    private func maskPixels(_ image: CIImage, size: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: size * size)
        pixels.withUnsafeMutableBytes { raw in
            RenderService.context.render(image,
                                         toBitmap: raw.baseAddress!,
                                         rowBytes: size,
                                         bounds: CGRect(x: 0, y: 0,
                                                        width: CGFloat(size), height: CGFloat(size)),
                                         format: .R8,
                                         colorSpace: nil)
        }
        return pixels
    }

    @Test func strokePaintsWhereExpected() throws {
        // Horizontal stroke across the vertical middle.
        let stroke = BrushStroke(points: [CGPoint(x: 0.1, y: 0.5), CGPoint(x: 0.9, y: 0.5)],
                                 radius: 0.05)
        let size = 100
        let mask = try #require(RetouchMask.rasterize(
            strokes: [stroke], size: CGSize(width: size, height: size)))
        let pixels = maskPixels(mask, size: size)

        // Bitmap row 0 = top. Center row is painted, top/bottom rows are not.
        #expect(pixels[50 * size + 50] > 200)
        #expect(pixels[5 * size + 50] == 0)
        #expect(pixels[95 * size + 50] == 0)
        // Left/right margins beyond the stroke are unpainted.
        #expect(pixels[50 * size + 2] == 0)
        #expect(pixels[50 * size + 97] == 0)
    }

    @Test func normalizedYIsTopDown() throws {
        // A dot near the TOP of the image (y ≈ 0.1).
        let stroke = BrushStroke(points: [CGPoint(x: 0.5, y: 0.1)], radius: 0.04)
        let size = 100
        let mask = try #require(RetouchMask.rasterize(
            strokes: [stroke], size: CGSize(width: size, height: size)))
        let pixels = maskPixels(mask, size: size)
        #expect(pixels[10 * size + 50] > 200)   // top of the bitmap
        #expect(pixels[90 * size + 50] == 0)    // bottom untouched
    }

    @Test func emptyStrokesGiveNilMask() {
        #expect(RetouchMask.rasterize(strokes: [], size: CGSize(width: 10, height: 10)) == nil)
    }

    @Test func inpaintTileRegionCoversSmallDefect() {
        let bbox = CGRect(x: 1500, y: 900, width: 40, height: 30)
        let region = InpaintStage.tileRegion(for: bbox,
                                             imageExtent: CGRect(x: 0, y: 0, width: 3000, height: 2000))
        #expect(region.width >= 800)
        #expect(region.height >= 800)
        #expect(region.contains(bbox))
        #expect(CGRect(x: 0, y: 0, width: 3000, height: 2000).contains(region))
    }
}
