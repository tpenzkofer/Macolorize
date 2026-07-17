import Testing
import CoreImage
import ImageIO
import UniformTypeIdentifiers
@testable import MacolorizeCore

/// Phase 0 acceptance: a 16-bit TIFF with metadata survives
/// load → render → export with depth and EXIF/TIFF metadata intact.
struct ImageIORoundTripTests {
    @Test func sixteenBitTIFFRoundTrip() throws {
        let fixtureURL = try Self.makeFixtureTIFF()
        defer { try? FileManager.default.removeItem(at: fixtureURL) }

        let loaded = try ImageLoader.load(fixtureURL)
        #expect(loaded.pixelSize == CGSize(width: 64, height: 48))

        let outputURL = fixtureURL.deletingLastPathComponent()
            .appendingPathComponent("roundtrip_out.tif")
        defer { try? FileManager.default.removeItem(at: outputURL) }
        try ImageExporter.export(loaded.fullResolution, from: fixtureURL, to: outputURL)

        let source = try #require(CGImageSourceCreateWithURL(outputURL as CFURL, nil))
        let props = try #require(
            CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])

        #expect(props[kCGImagePropertyDepth] as? Int == 16)
        #expect(props[kCGImagePropertyPixelWidth] as? Int == 64)
        #expect(props[kCGImagePropertyPixelHeight] as? Int == 48)

        let tiff = try #require(props[kCGImagePropertyTIFFDictionary] as? [CFString: Any])
        #expect(tiff[kCGImagePropertyTIFFModel] as? String == "TestScanner 9000")
    }

    @Test func proxyStaysSmall() throws {
        let fixtureURL = try Self.makeFixtureTIFF(width: 64, height: 48)
        defer { try? FileManager.default.removeItem(at: fixtureURL) }
        let loaded = try ImageLoader.load(fixtureURL)
        // Small images are used as-is.
        #expect(loaded.proxy.extent.size == loaded.pixelSize)
    }

    @Test func exportNamingAvoidsCollisions() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let source = dir.appendingPathComponent("slide.tif")
        let first = ExportNaming.destination(for: source)
        #expect(first.lastPathComponent == "slide_restored.tif")
        #expect(first.deletingLastPathComponent().lastPathComponent == "Macolorized")

        try FileManager.default.createDirectory(
            at: first.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: first)
        let second = ExportNaming.destination(for: source)
        #expect(second.lastPathComponent == "slide_restored-2.tif")
    }

    /// Builds a 16-bit RGBA TIFF gradient with TIFF/EXIF metadata in a temp dir.
    static func makeFixtureTIFF(width: Int = 64, height: Int = 48) throws -> URL {
        var pixels = [UInt16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                pixels[i] = UInt16((x * 65535) / max(width - 1, 1))
                pixels[i + 1] = UInt16((y * 65535) / max(height - 1, 1))
                pixels[i + 2] = 32768
                pixels[i + 3] = 65535
            }
        }
        let context = try #require(pixels.withUnsafeMutableBytes { buffer in
            CGContext(data: buffer.baseAddress,
                      width: width,
                      height: height,
                      bitsPerComponent: 16,
                      bytesPerRow: width * 8,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                          | CGBitmapInfo.byteOrder16Little.rawValue)
        })
        let cgImage = try #require(context.makeImage())

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fixture_\(UUID().uuidString).tif")
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil))
        let metadata: [CFString: Any] = [
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFModel: "TestScanner 9000",
                kCGImagePropertyTIFFMake: "Macolorize",
            ],
            kCGImagePropertyExifDictionary: [
                kCGImagePropertyExifUserComment: "fixture",
            ],
        ]
        CGImageDestinationAddImage(destination, cgImage, metadata as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }
}
