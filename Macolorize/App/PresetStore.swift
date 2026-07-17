import Foundation
import Observation
import MacolorizeCore

/// Named recipes persisted as JSON in Application Support.
@MainActor @Observable
final class PresetStore {
    struct Preset: Identifiable, Sendable {
        let name: String
        let recipe: Recipe
        var id: String { name }
    }

    private(set) var presets: [Preset] = []

    private static var directory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask).first!
        return base.appendingPathComponent("Macolorize/Presets", isDirectory: true)
    }

    init() {
        reload()
    }

    func reload() {
        let decoder = JSONDecoder()
        let files = (try? FileManager.default.contentsOfDirectory(
            at: Self.directory, includingPropertiesForKeys: nil)) ?? []
        presets = files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url),
                      let recipe = try? decoder.decode(Recipe.self, from: data) else { return nil }
                return Preset(name: url.deletingPathExtension().lastPathComponent, recipe: recipe)
            }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    func save(name: String, recipe: Recipe) throws {
        try FileManager.default.createDirectory(at: Self.directory,
                                                withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let sanitized = name.replacingOccurrences(of: "/", with: "-")
        let url = Self.directory.appendingPathComponent(sanitized).appendingPathExtension("json")
        try encoder.encode(recipe).write(to: url)
        reload()
    }

    func delete(_ preset: Preset) {
        let url = Self.directory.appendingPathComponent(preset.name).appendingPathExtension("json")
        try? FileManager.default.removeItem(at: url)
        reload()
    }
}
