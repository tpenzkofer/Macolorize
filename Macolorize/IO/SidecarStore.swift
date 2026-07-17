import Foundation

/// Per-image session state persisted next to the source scans, so reopening
/// a slide restores its dialed-in recipe and retouch strokes. Originals are
/// never modified; sidecars live in a hidden `.macolorize` subfolder.
public struct SessionSidecar: Codable, Sendable, Equatable {
    public var recipe: Recipe
    public var strokes: [BrushStroke]
    public var colorHints: [ColorHint]
    public var rotation: Int

    public init(recipe: Recipe, strokes: [BrushStroke],
                colorHints: [ColorHint] = [], rotation: Int = 0) {
        self.recipe = recipe
        self.strokes = strokes
        self.colorHints = colorHints
        self.rotation = rotation
    }

    private enum CodingKeys: String, CodingKey {
        case recipe, strokes, colorHints, rotation
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        recipe = try container.decodeIfPresent(Recipe.self, forKey: .recipe) ?? Recipe()
        strokes = try container.decodeIfPresent([BrushStroke].self, forKey: .strokes) ?? []
        colorHints = try container.decodeIfPresent([ColorHint].self, forKey: .colorHints) ?? []
        rotation = try container.decodeIfPresent(Int.self, forKey: .rotation) ?? 0
    }
}

public enum SidecarStore {
    static func url(for source: URL) -> URL {
        source.deletingLastPathComponent()
            .appendingPathComponent(".macolorize", isDirectory: true)
            .appendingPathComponent(source.lastPathComponent)
            .appendingPathExtension("json")
    }

    public static func load(for source: URL) -> SessionSidecar? {
        guard let data = try? Data(contentsOf: url(for: source)) else { return nil }
        return try? JSONDecoder().decode(SessionSidecar.self, from: data)
    }

    public static func save(_ sidecar: SessionSidecar, for source: URL) {
        let destination = url(for: source)
        do {
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(sidecar).write(to: destination)
        } catch {
            // Sidecars are a convenience; failing to save one is not fatal.
        }
    }
}
