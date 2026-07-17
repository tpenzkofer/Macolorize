import Foundation
import CoreImage

public enum BatchEvent: Sendable {
    case started(URL)
    case finished(source: URL, output: URL)
    case failed(source: URL, message: String)
}

/// Runs a recipe over many files with bounded concurrency (full-res float
/// pipelines are memory-hungry; two in flight is plenty on Apple Silicon).
public enum BatchRunner {
    public static func run(urls: [URL],
                           recipe: Recipe,
                           format: ExportFormat = .tiff16,
                           maxConcurrent: Int = 2) -> AsyncStream<BatchEvent> {
        AsyncStream { continuation in
            let task = Task {
                await withTaskGroup(of: Void.self) { group in
                    var pending = urls.makeIterator()

                    @discardableResult
                    func addNext() -> Bool {
                        guard let url = pending.next() else { return false }
                        group.addTask {
                            continuation.yield(.started(url))
                            do {
                                try Task.checkCancellation()
                                let loaded = try ImageLoader.load(url)
                                let result = try await ProcessingEngine.shared.run(
                                    recipe: recipe,
                                    on: PipelineImage(image: loaded.fullResolution),
                                    mode: .export)
                                let destination = ExportNaming.destination(for: url, format: format)
                                try ImageExporter.export(result.image,
                                                         from: url,
                                                         to: destination,
                                                         format: format)
                                continuation.yield(.finished(source: url, output: destination))
                            } catch is CancellationError {
                                continuation.yield(.failed(source: url, message: "Cancelled"))
                            } catch {
                                continuation.yield(.failed(source: url,
                                                           message: error.localizedDescription))
                            }
                        }
                        return true
                    }

                    for _ in 0..<max(maxConcurrent, 1) { addNext() }
                    while await group.next() != nil {
                        if Task.isCancelled {
                            group.cancelAll()
                        } else {
                            addNext()
                        }
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
