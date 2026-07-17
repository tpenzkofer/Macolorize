import SwiftUI
import MacolorizeCore

struct SlideFile: Identifiable, Hashable, Sendable {
    let url: URL
    var id: URL { url }
    var name: String { url.lastPathComponent }
}

@MainActor @Observable
final class AppState {
    var folderURL: URL?
    var files: [SlideFile] = []
    var session: EditSession?
    var showFolderPicker = false
    var compareEnabled = false
    var showBatchSheet = false
    var retouchMode = false
    /// Carried across image switches so a dialed-in look sticks.
    private var lastRecipe = Recipe()

    var selection: URL? {
        didSet { openSelection() }
    }

    init() {
        if let restored = FolderAccess.restore() {
            setFolder(restored)
        }
    }

    func setFolder(_ url: URL) {
        folderURL = url
        FolderAccess.persist(url)
        files = ImageLoader.imageFiles(in: url).map { SlideFile(url: $0) }
        selection = files.first?.id
    }

    private func openSelection() {
        if let current = session {
            lastRecipe = current.recipe
        }
        guard let selection,
              let file = files.first(where: { $0.id == selection }) else {
            session = nil
            return
        }
        guard session?.file != file else { return }
        let newSession = EditSession(file: file, recipe: lastRecipe)
        session = newSession
        Task { await newSession.load() }
    }
}
