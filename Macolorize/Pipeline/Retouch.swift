import CoreGraphics
import CoreImage

/// One brush stroke, in resolution-independent coordinates:
/// points normalized to [0,1] (x right, y down from the image top),
/// radius normalized to image width. Rasterizable at any output size.
public struct BrushStroke: Codable, Sendable, Equatable {
    public var points: [CGPoint]
    public var radius: CGFloat

    public init(points: [CGPoint], radius: CGFloat) {
        self.points = points
        self.radius = radius
    }
}

public enum RetouchMask {
    /// Rasterizes strokes into a white-on-black mask of the given pixel size.
    public static func rasterize(strokes: [BrushStroke], size: CGSize) -> CIImage? {
        guard !strokes.isEmpty, size.width >= 1, size.height >= 1 else { return nil }
        let width = Int(size.width.rounded())
        let height = Int(size.height.rounded())
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width,
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }

        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setStrokeColor(gray: 1, alpha: 1)
        context.setLineCap(.round)
        context.setLineJoin(.round)

        for stroke in strokes {
            guard let first = stroke.points.first else { continue }
            context.setLineWidth(max(stroke.radius * 2 * size.width, 1))
            // Normalized y is top-down; CGContext origin is bottom-left.
            func devicePoint(_ p: CGPoint) -> CGPoint {
                CGPoint(x: p.x * size.width, y: (1 - p.y) * size.height)
            }
            context.beginPath()
            context.move(to: devicePoint(first))
            if stroke.points.count == 1 {
                // A dot: tiny segment so the round cap paints a disc.
                context.addLine(to: devicePoint(first).applying(.init(translationX: 0.01, y: 0)))
            } else {
                for point in stroke.points.dropFirst() {
                    context.addLine(to: devicePoint(point))
                }
            }
            context.strokePath()
        }

        guard let cgImage = context.makeImage() else { return nil }
        return CIImage(cgImage: cgImage)
    }
}
