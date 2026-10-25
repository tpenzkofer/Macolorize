import Foundation
import Observation
import MacolorizeCore

/// Downloads model archives from the GitHub releases and installs them.
/// Only archives that provide a missing model are fetched, so an existing
/// install picks up a model update without re-downloading the 1.4 GB base.
@MainActor @Observable
final class ModelDownloader: NSObject {
    enum Phase: Equatable {
        case idle
        case downloading(Double)   // 0…1 across all needed archives
        case installing
        case done(Int)             // models installed
        case failed(String)
    }

    private(set) var phase: Phase = .idle

    struct Archive: Sendable {
        let url: URL
        let models: Set<ModelID>
        /// Approximate download size, for the inspector copy.
        let megabytes: Int
    }

    /// Pinned to the release tags that carry each archive; add an archive
    /// (rather than re-uploading the base) when models change. Overridable
    /// for testing.
    static var archives: [Archive] {
        let environment = ProcessInfo.processInfo.environment
        func url(_ key: String, _ fallback: String) -> URL {
            environment[key].flatMap(URL.init(string:)) ?? URL(string: fallback)!
        }
        return [
            Archive(url: url("MACOLORIZE_MODELS_URL",
                             "https://github.com/tpenzkofer/Macolorize/releases/download/v1.1.0/Macolorize-Models.zip"),
                    models: [.lama, .gfpgan, .ddcolor, .ddcolorArtistic,
                             .realesrganX2, .realesrganX4, .drunet, .scratchnet],
                    megabytes: 1400),
            Archive(url: url("MACOLORIZE_MODELS_UPDATE_URL",
                             "https://github.com/tpenzkofer/Macolorize/releases/download/v1.3.0/Macolorize-Models-Update-1.3.zip"),
                    models: [.drunetMap],
                    megabytes: 60),
        ]
    }

    /// The archives that provide at least one missing model.
    static func neededArchives(missing: Set<ModelID>) -> [Archive] {
        archives.filter { !$0.models.isDisjoint(with: missing) }
    }

    var isBusy: Bool {
        switch phase {
        case .downloading, .installing: true
        default: false
        }
    }

    private var progressObservation: NSKeyValueObservation?

    private enum DownloadOutcome: Sendable {
        case success(URL)
        case failure(String)
    }

    func start(library: ModelLibrary) {
        guard !isBusy else { return }
        let queue = Self.neededArchives(missing: library.missing)
        guard !queue.isEmpty else { return }
        phase = .downloading(0)
        download(queue, index: 0, installed: 0, library: library)
    }

    private func download(_ queue: [Archive], index: Int, installed: Int,
                          library: ModelLibrary) {
        let archive = queue[index]
        let task = URLSession.shared.downloadTask(with: archive.url) { [weak self] location, response, error in
            let outcome: DownloadOutcome
            if let error {
                outcome = .failure(error.localizedDescription)
            } else if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                outcome = .failure("Not available for download yet — use Install from Zip… with a downloaded models archive instead.")
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                outcome = .failure("Download failed (HTTP \(http.statusCode)).")
            } else if let location {
                // Move out of the system temp slot before the handler returns.
                let kept = FileManager.default.temporaryDirectory
                    .appendingPathComponent("Macolorize-Models-\(UUID().uuidString).zip")
                do {
                    try FileManager.default.moveItem(at: location, to: kept)
                    outcome = .success(kept)
                } catch {
                    outcome = .failure(error.localizedDescription)
                }
            } else {
                outcome = .failure("Download failed.")
            }
            Task { @MainActor [weak self] in
                self?.finish(outcome, queue: queue, index: index,
                             installed: installed, library: library)
            }
        }
        let total = Double(queue.count)
        progressObservation = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            let fraction = (Double(index) + progress.fractionCompleted) / total
            Task { @MainActor [weak self] in
                if case .downloading = self?.phase {
                    self?.phase = .downloading(fraction)
                }
            }
        }
        task.resume()
    }

    private func finish(_ outcome: DownloadOutcome, queue: [Archive], index: Int,
                        installed: Int, library: ModelLibrary) {
        progressObservation = nil
        switch outcome {
        case .failure(let message):
            library.refresh()
            phase = .failed(message)
        case .success(let zipURL):
            phase = .installing
            Task {
                do {
                    let count = try await Task.detached(priority: .userInitiated) {
                        defer { try? FileManager.default.removeItem(at: zipURL) }
                        return try ModelInstaller.install(from: zipURL)
                    }.value
                    library.refresh()
                    if index + 1 < queue.count {
                        phase = .downloading(Double(index + 1) / Double(queue.count))
                        download(queue, index: index + 1, installed: installed + count,
                                 library: library)
                    } else {
                        phase = .done(installed + count)
                    }
                } catch {
                    phase = .failed(error.localizedDescription)
                }
            }
        }
    }
}
