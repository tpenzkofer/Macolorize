import CoreImage
import ImageIO
import UniformTypeIdentifiers

public enum ExportFormat: String, Sendable, CaseIterable, Codable {
    case tiff16
    case png
    case jpeg

    public var utType: UTType {
        switch self {
        case .tiff16: .tiff
        case .png: .png
        case .jpeg: .jpeg
        }
    }

    public var fileExtension: String {
        switch self {
        case .tiff16: "tif"
        case .png: "png"
        case .jpeg: "jpg"
        }
    }
}

public enum ImageExporterError: Error, LocalizedError {
    case renderFailed
    case writeFailed(URL)

    public var errorDescription: String? {
        switch self {
        case .renderFailed: "Could not render the processed image."
        case .writeFailed(let url): "Could not write \(url.lastPathComponent)."
        }
    }
}

public enum ImageExporter {
    /// Renders `image` and writes it to `destinationURL`, carrying over the
    /// source file's EXIF/IPTC/TIFF metadata. Originals are never modified.
    public static func export(_ image: CIImage,
                              from sourceURL: URL,
                              to destinationURL: URL,
                              format: ExportFormat = .tiff16,
                              context: CIContext = RenderService.context) throws {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let ciFormat: CIFormat = format == .tiff16 ? .RGBA16 : .RGBA8
        guard let cgImage = context.createCGImage(image,
                                                  from: image.extent,
                                                  format: ciFormat,
                                                  colorSpace: colorSpace)
        else { throw ImageExporterError.renderFailed }

        try FileManager.default.createDirectory(at: destinationURL.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)

        var properties = sourceMetadata(of: sourceURL)
        // Geometry and orientation describe the *source* pixels; drop them.
        properties.removeValue(forKey: kCGImagePropertyPixelWidth)
        properties.removeValue(forKey: kCGImagePropertyPixelHeight)
        properties.removeValue(forKey: kCGImagePropertyOrientation)
        properties.removeValue(forKey: kCGImagePropertyDepth)
        if var tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any] {
            tiff.removeValue(forKey: kCGImagePropertyTIFFOrientation)
            properties[kCGImagePropertyTIFFDictionary] = tiff
        }
        if format == .jpeg {
            properties[kCGImageDestinationLossyCompressionQuality] = 0.92
        }

        guard let destination = CGImageDestinationCreateWithURL(destinationURL as CFURL,
                                                                format.utType.identifier as CFString,
                                                                1, nil)
        else { throw ImageExporterError.writeFailed(destinationURL) }

        CGImageDestinationAddImage(destination, cgImage, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ImageExporterError.writeFailed(destinationURL)
        }
    }

    private static func sourceMetadata(of url: URL) -> [CFString: Any] {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        else { return [:] }
        return props
    }
}

public enum ExportNaming {
    /// Default non-destructive output path: `<sourceFolder>/Macolorized/<name>_restored.<ext>`,
    /// with `-2`, `-3`… appended on collision.
    public static func destination(for source: URL, format: ExportFormat = .tiff16) -> URL {
        let folder = source.deletingLastPathComponent()
            .appendingPathComponent("Macolorized", isDirectory: true)
        let base = source.deletingPathExtension().lastPathComponent + "_restored"
        var candidate = folder.appendingPathComponent(base)
            .appendingPathExtension(format.fileExtension)
        var counter = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = folder.appendingPathComponent("\(base)-\(counter)")
                .appendingPathExtension(format.fileExtension)
            counter += 1
        }
        return candidate
    }
}
