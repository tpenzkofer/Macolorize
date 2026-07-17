import Foundation

public enum ModelInstallerError: Error, LocalizedError {
    case unreadable
    case extractionFailed
    case nothingFound

    public var errorDescription: String? {
        switch self {
        case .unreadable: "Could not read the selected file."
        case .extractionFailed: "Could not unpack the models archive."
        case .nothingFound: "No .mlpackage models found in the archive."
        }
    }
}

/// Imports a Macolorize-Models.zip (or loose .mlpackage bundles) into
/// Application Support. The zip is copied into our container first so the
/// extraction subprocess needs no sandbox extension of its own.
public enum ModelInstaller {
    /// Returns the number of models installed. Synchronous (extraction can
    /// take ~30 s for the full archive) — call from a background task.
    public static func install(from pickedURL: URL) throws -> Int {
        let accessing = pickedURL.startAccessingSecurityScopedResource()
        defer { if accessing { pickedURL.stopAccessingSecurityScopedResource() } }

        let staging = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-import-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        var searchRoot = staging
        if pickedURL.pathExtension.lowercased() == "mlpackage" {
            // A single model picked directly.
            let copy = staging.appendingPathComponent(pickedURL.lastPathComponent)
            try FileManager.default.copyItem(at: pickedURL, to: copy)
        } else {
            let zipCopy = staging.appendingPathComponent("models.zip")
            do {
                try FileManager.default.copyItem(at: pickedURL, to: zipCopy)
            } catch {
                throw ModelInstallerError.unreadable
            }
            let extracted = staging.appendingPathComponent("extracted", isDirectory: true)
            try FileManager.default.createDirectory(at: extracted, withIntermediateDirectories: true)

            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
            process.arguments = ["-x", "-k", zipCopy.path, extracted.path]
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                throw ModelInstallerError.extractionFailed
            }
            searchRoot = extracted
        }

        // Find .mlpackage bundles anywhere in the extracted tree.
        var found: [URL] = []
        if let enumerator = FileManager.default.enumerator(
            at: searchRoot, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator where url.pathExtension == "mlpackage" {
                found.append(url)
                enumerator.skipDescendants()
            }
        }
        guard !found.isEmpty else { throw ModelInstallerError.nothingFound }

        let destination = ModelStore.installDirectory
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for model in found {
            let target = destination.appendingPathComponent(model.lastPathComponent)
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: model, to: target)
        }
        return found.count
    }
}
