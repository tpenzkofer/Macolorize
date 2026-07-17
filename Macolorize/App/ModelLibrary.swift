import Foundation
import Observation
import MacolorizeCore

/// Live view of which Core ML models are installed; refreshed after imports
/// so the UI unlocks without a relaunch.
@MainActor @Observable
final class ModelLibrary {
    private(set) var installed: Set<ModelID> = []

    /// Models the app actually ships/needs (dustnet has a classical fallback
    /// and never exists as a file).
    static let required: [ModelID] = [
        .lama, .gfpgan, .ddcolor, .ddcolorArtistic,
        .realesrganX2, .realesrganX4, .drunet, .scratchnet,
    ]

    init() {
        refresh()
    }

    func refresh() {
        installed = Set(ModelID.allCases.filter { ModelStore.isAvailable($0) })
    }

    func has(_ id: ModelID) -> Bool {
        installed.contains(id)
    }

    var missingCount: Int {
        Self.required.filter { !installed.contains($0) }.count
    }

    var allInstalled: Bool { missingCount == 0 }
}
