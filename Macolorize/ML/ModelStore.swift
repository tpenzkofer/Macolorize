import CoreML
import Foundation

public enum ModelID: String, CaseIterable, Sendable {
    case realesrganX2 = "RealESRGAN_x2plus"
    case realesrganX4 = "RealESRGAN_general_x4v3"
    case ddcolor = "DDColor"
    case ddcolorArtistic = "DDColor_artistic"
    case lama = "LaMa"
    case dustnet = "DustNet"
    case gfpgan = "GFPGAN"
    case drunet = "DRUNet"
    case scratchnet = "ScratchNet"

    /// LaMa's Fourier convolutions can't run on the ANE; GFPGAN's StyleGAN
    /// decoder needs FP32 (FP16 overflows), which also means GPU.
    var computeUnits: MLComputeUnits {
        switch self {
        case .lama, .gfpgan: .cpuAndGPU
        default: .all
        }
    }
}

public enum ModelStoreError: Error, LocalizedError {
    case notFound(ModelID)

    public var errorDescription: String? {
        switch self {
        case .notFound(let id):
            "Model \(id.rawValue) is not installed. Run `make models` in the project folder."
        }
    }
}

/// MLModel prediction is documented thread-safe; the wrapper lets loaded
/// models cross actor boundaries.
public struct SendableModel: @unchecked Sendable {
    public let mlModel: MLModel
}

/// Lazy-loading cache for Core ML models. Looks in the app bundle first
/// (compiled .mlmodelc resources), then $MACOLORIZE_MODELS_DIR, then ./Models
/// (CLI/test runs from the repo); .mlpackage files found on disk are compiled
/// once and cached in ~/Library/Caches.
public actor ModelStore {
    public static let shared = ModelStore()

    private var loaded: [ModelID: SendableModel] = [:]

    public func model(_ id: ModelID) async throws -> SendableModel {
        if let cached = loaded[id] { return cached }
        guard let url = Self.locate(id) else { throw ModelStoreError.notFound(id) }

        let compiledURL: URL
        if url.pathExtension == "mlmodelc" {
            compiledURL = url
        } else {
            compiledURL = try await Self.compiledURL(for: url)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = id.computeUnits
        let model = SendableModel(mlModel: try MLModel(contentsOf: compiledURL,
                                                       configuration: configuration))
        loaded[id] = model
        return model
    }

    public static func isAvailable(_ id: ModelID) -> Bool {
        locate(id) != nil
    }

    static func locate(_ id: ModelID) -> URL? {
        var candidates: [URL] = []
        if let bundled = Bundle.main.url(forResource: id.rawValue, withExtension: "mlmodelc") {
            candidates.append(bundled)
        }
        if let bundled = Bundle.main.url(forResource: id.rawValue, withExtension: "mlpackage") {
            candidates.append(bundled)
        }
        var directories: [URL] = []
        if let custom = ProcessInfo.processInfo.environment["MACOLORIZE_MODELS_DIR"] {
            directories.append(URL(fileURLWithPath: custom, isDirectory: true))
        }
        directories.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Models", isDirectory: true))
        for directory in directories {
            candidates.append(directory.appendingPathComponent("\(id.rawValue).mlpackage"))
            candidates.append(directory.appendingPathComponent("\(id.rawValue).mlmodelc"))
        }
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Compiles an .mlpackage, caching by name + modification date.
    private static func compiledURL(for packageURL: URL) async throws -> URL {
        let cacheRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Macolorize/CompiledModels", isDirectory: true)
        let values = try? packageURL.resourceValues(forKeys: [.contentModificationDateKey])
        let modified = values?.contentModificationDate?.timeIntervalSince1970 ?? 0
        let cacheName = packageURL.deletingPathExtension().lastPathComponent
            + "-\(Int(modified)).mlmodelc"
        let cached = cacheRoot.appendingPathComponent(cacheName)
        if FileManager.default.fileExists(atPath: cached.path) {
            return cached
        }
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let compiled = try await MLModel.compileModel(at: packageURL)
        try? FileManager.default.removeItem(at: cached)
        try FileManager.default.moveItem(at: compiled, to: cached)
        return cached
    }
}
