import SwiftUI

struct MainWindow: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            DetailView()
        }
        .navigationTitle(appState.session?.file.name ?? "Macolorize")
    }
}
