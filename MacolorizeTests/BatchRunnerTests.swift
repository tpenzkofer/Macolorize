import Testing
import Foundation
@testable import MacolorizeCore

struct BatchRunnerTests {
    @Test func processesAllFilesInFolder() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        for name in ["a", "b", "c"] {
            let fixture = try ImageIORoundTripTests.makeFixtureTIFF()
            try FileManager.default.moveItem(
                at: fixture, to: dir.appendingPathComponent("\(name).tif"))
        }

        var recipe = Recipe()
        recipe.restore.enabled = true

        var finished: [URL] = []
        var failed: [String] = []
        let urls = ImageLoader.imageFiles(in: dir)
        #expect(urls.count == 3)

        for await event in BatchRunner.run(urls: urls, recipe: recipe, format: .tiff16) {
            switch event {
            case .started: break
            case .finished(_, let output): finished.append(output)
            case .failed(_, let message): failed.append(message)
            }
        }

        #expect(failed.isEmpty)
        #expect(finished.count == 3)
        for output in finished {
            #expect(FileManager.default.fileExists(atPath: output.path))
            #expect(output.deletingLastPathComponent().lastPathComponent == "Macolorized")
        }
    }
}
