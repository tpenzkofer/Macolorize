import SwiftUI
import MacolorizeCore

struct SidebarView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        @Bindable var appState = appState
        List(appState.files, selection: $appState.selection) { file in
            HStack(spacing: 8) {
                ThumbnailView(url: file.url)
                Text(file.name)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .tag(file.id)
        }
        .overlay {
            if appState.files.isEmpty {
                ContentUnavailableView(
                    "No Scans",
                    systemImage: "photo.on.rectangle.angled",
                    description: Text("Open a folder of scanned slides to get started."))
            }
        }
        .safeAreaInset(edge: .bottom) {
            Button {
                appState.showFolderPicker = true
            } label: {
                Label("Open Folder…", systemImage: "folder")
                    .frame(maxWidth: .infinity)
            }
            .controlSize(.large)
            .padding(10)
        }
        .fileImporter(isPresented: $appState.showFolderPicker,
                      allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                guard url.startAccessingSecurityScopedResource() else { return }
                appState.setFolder(url)
            }
        }
    }
}

/// Wrapper so an immutable CGImage can cross a task boundary.
private struct SendableCGImage: @unchecked Sendable {
    let image: CGImage
}

struct ThumbnailView: View {
    let url: URL
    @State private var thumbnail: CGImage?

    var body: some View {
        Group {
            if let thumbnail {
                Image(decorative: thumbnail, scale: 2)
                    .resizable()
                    .scaledToFill()
            } else {
                Color.secondary.opacity(0.15)
            }
        }
        .frame(width: 44, height: 44)
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .task(id: url) {
            let url = url
            let wrapped = await Task.detached(priority: .utility) {
                ImageLoader.thumbnail(for: url, maxPixel: 128).map(SendableCGImage.init)
            }.value
            thumbnail = wrapped?.image
        }
    }
}
