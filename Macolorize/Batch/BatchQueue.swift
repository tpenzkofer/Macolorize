import Foundation
import Observation

/// Main-actor view model over a BatchRunner stream.
@MainActor @Observable
public final class BatchQueue {
    public enum RowStatus: Equatable, Sendable {
        case pending
        case running
        case finished(URL)
        case failed(String)
    }

    public struct Row: Identifiable, Sendable {
        public let url: URL
        public var status: RowStatus

        public init(url: URL, status: RowStatus = .pending) {
            self.url = url
            self.status = status
        }

        public var id: URL { url }
        public var name: String { url.lastPathComponent }
    }

    public private(set) var rows: [Row] = []
    public private(set) var isRunning = false
    private var runTask: Task<Void, Never>?

    public init() {}

    public var completedCount: Int {
        rows.filter {
            if case .finished = $0.status { return true }
            if case .failed = $0.status { return true }
            return false
        }.count
    }

    public var progress: Double {
        rows.isEmpty ? 0 : Double(completedCount) / Double(rows.count)
    }

    public func start(urls: [URL], recipe: Recipe, format: ExportFormat) {
        guard !isRunning, !urls.isEmpty else { return }
        rows = urls.map { Row(url: $0) }
        isRunning = true
        runTask = Task { [weak self] in
            for await event in BatchRunner.run(urls: urls, recipe: recipe, format: format) {
                guard let self else { return }
                switch event {
                case .started(let url):
                    self.update(url) { $0.status = .running }
                case .finished(let source, let output):
                    self.update(source) { $0.status = .finished(output) }
                case .failed(let source, let message):
                    self.update(source) { $0.status = .failed(message) }
                }
            }
            self?.isRunning = false
        }
    }

    public func cancel() {
        runTask?.cancel()
    }

    private func update(_ url: URL, _ mutate: (inout Row) -> Void) {
        guard let index = rows.firstIndex(where: { $0.url == url }) else { return }
        mutate(&rows[index])
    }
}
