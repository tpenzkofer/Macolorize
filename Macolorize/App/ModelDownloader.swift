import Foundation
import Observation
import MacolorizeCore

/// Downloads the models archive from the GitHub release and installs it.
/// While the repository is private, GitHub answers 404 for unauthenticated
/// asset downloads — surfaced as a friendly hint to use manual import.
@MainActor @Observable
final class ModelDownloader: NSObject {
    enum Phase: Equatable {
        case idle
        case downloading(Double)   // 0…1
        case installing
        case done(Int)             // models installed
        case failed(String)
    }

    private(set) var phase: Phase = .idle

    /// Pinned to the release tag that carries the current models archive;
    /// bump when the models change. Overridable for testing.
    static var modelsURL: URL {
        if let override = ProcessInfo.processInfo.environment["MACOLORIZE_MODELS_URL"],
           let url = URL(string: override) {
            return url
        }
        return URL(string:
            "https://github.com/tpenzkofer/Macolorize/releases/download/v1.1.0/Macolorize-Models.zip")!
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
        phase = .downloading(0)

        let task = URLSession.shared.downloadTask(with: Self.modelsURL) { [weak self] location, response, error in
            let outcome: DownloadOutcome
            if let error {
                outcome = .failure(error.localizedDescription)
            } else if let http = response as? HTTPURLResponse, http.statusCode == 404 {
                outcome = .failure("Not available for download yet — use Install Models… with a downloaded Macolorize-Models.zip instead.")
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
                self?.finishDownload(outcome, library: library)
            }
        }
        progressObservation = task.progress.observe(\.fractionCompleted) { [weak self] progress, _ in
            let fraction = progress.fractionCompleted
            Task { @MainActor [weak self] in
                if case .downloading = self?.phase {
                    self?.phase = .downloading(fraction)
                }
            }
        }
        task.resume()
    }

    private func finishDownload(_ outcome: DownloadOutcome, library: ModelLibrary) {
        progressObservation = nil
        switch outcome {
        case .failure(let message):
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
                    phase = .done(count)
                } catch {
                    phase = .failed(error.localizedDescription)
                }
            }
        }
    }
}
