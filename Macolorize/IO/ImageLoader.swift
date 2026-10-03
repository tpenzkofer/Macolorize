import CoreImage
import ImageIO
import UniformTypeIdentifiers

/// A decoded scan: full-resolution image plus a screen-sized proxy for live preview.
/// CIImage is immutable; the struct is safe to pass across concurrency domains.
public struct LoadedImage: @unchecked Sendable {
    public let url: URL
    public let fullResolution: CIImage
    public let proxy: CIImage
    public let pixelSize: CGSize

    public init(url: URL, fullResolution: CIImage, proxy: CIImage, pixelSize: CGSize) {
        self.url = url
        self.fullResolution = fullResolution
        self.proxy = proxy
        self.pixelSize = pixelSize
    }
}

public enum ImageLoaderError: Error, LocalizedError {
    case unreadable(URL)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let url): "Could not read image at \(url.path)"
        }
    }
}

public enum ImageLoader {
    /// Long-edge size of the interactive preview proxy. 4096 keeps typical
    /// 4K archive scans at full resolution in the preview (issue #1: a 2048
    /// proxy showed ~half resolution and made preview diverge from export);
    /// larger scans still use a proxy and re-render fully on export.
    public static let proxyMaxDimension: CGFloat = 4096

    public static func load(_ url: URL) throws -> LoadedImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) > 0,
              let full = CIImage(contentsOf: url, options: [.applyOrientationProperty: true])
        else {
            throw ImageLoaderError.unreadable(url)
        }
        // Normalize to a zero origin so downstream geometry can assume it.
        let normalized = full.transformed(by: .init(translationX: -full.extent.minX,
                                                    y: -full.extent.minY))

        let proxy: CIImage
        if max(normalized.extent.width, normalized.extent.height) <= proxyMaxDimension {
            proxy = normalized
        } else if let thumb = CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: proxyMaxDimension,
        ] as CFDictionary) {
            proxy = CIImage(cgImage: thumb)
        } else {
            proxy = normalized
        }

        return LoadedImage(url: url,
                           fullResolution: normalized,
                           proxy: proxy,
                           pixelSize: normalized.extent.size)
    }

    public static func thumbnail(for url: URL, maxPixel: CGFloat = 256) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(source, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ] as CFDictionary)
    }

    /// Image files in a folder, sorted like Finder.
    public static func imageFiles(in folder: URL) -> [URL] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles])) ?? []
        return contents
            .filter { url in
                guard let type = UTType(filenameExtension: url.pathExtension) else { return false }
                return type.conforms(to: .image)
            }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }
}
