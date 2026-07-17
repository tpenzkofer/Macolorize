import CoreImage

public struct HistogramData: Equatable, Sendable {
    public var red: [Float]
    public var green: [Float]
    public var blue: [Float]
    public var binCount: Int { red.count }
}

public enum Histogram {
    /// Per-channel histogram of a small render of the image (display-referred
    /// sRGB), normalized so the tallest bin is 1.
    public static func compute(from image: CIImage,
                               ciContext: CIContext,
                               bins: Int = 64) -> HistogramData {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else {
            return HistogramData(red: [], green: [], blue: [])
        }
        let scale = min(1, 160 / max(extent.width, extent.height))
        let width = max(Int(extent.width * scale), 1)
        let height = max(Int(extent.height * scale), 1)

        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes { raw in
            ciContext.render(image.transformed(by: .init(scaleX: scale, y: scale)),
                             toBitmap: raw.baseAddress!,
                             rowBytes: width * 4,
                             bounds: CGRect(x: extent.minX * scale, y: extent.minY * scale,
                                            width: CGFloat(width), height: CGFloat(height)),
                             format: .RGBA8,
                             colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        }

        var red = [Float](repeating: 0, count: bins)
        var green = [Float](repeating: 0, count: bins)
        var blue = [Float](repeating: 0, count: bins)
        let divisor = 256 / bins
        for i in 0..<(width * height) {
            red[Int(pixels[i * 4]) / divisor] += 1
            green[Int(pixels[i * 4 + 1]) / divisor] += 1
            blue[Int(pixels[i * 4 + 2]) / divisor] += 1
        }
        let peak = max(red.max() ?? 1, green.max() ?? 1, blue.max() ?? 1, 1)
        return HistogramData(red: red.map { $0 / peak },
                             green: green.map { $0 / peak },
                             blue: blue.map { $0 / peak })
    }
}
