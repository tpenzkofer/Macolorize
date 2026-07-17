import SwiftUI

@main
struct MacolorizeApp: App {
    @State private var appState = AppState()
    @State private var modelLibrary = ModelLibrary()

    var body: some Scene {
        WindowGroup {
            MainWindow()
                .environment(appState)
                .environment(modelLibrary)
        }
        .defaultLaunchBehavior(.presented)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Folder…") {
                    appState.showFolderPicker = true
                }
                .keyboardShortcut("o")

                Button("Process Folder…") {
                    appState.showBatchSheet = true
                }
                .keyboardShortcut("b")
                .disabled(appState.session == nil || appState.files.isEmpty)
            }
            CommandMenu("Adjustments") {
                Button("Toggle Restore Faded Color") {
                    appState.session?.recipe.restore.enabled.toggle()
                }
                .keyboardShortcut("r")
                .disabled(appState.session == nil)

                Button("Toggle Colorize B&W") {
                    appState.session?.recipe.colorize.enabled.toggle()
                }
                .keyboardShortcut("k")
                .disabled(appState.session == nil)

                Button("Toggle Restore Faces") {
                    appState.session?.recipe.faces.enabled.toggle()
                }
                .keyboardShortcut("f")
                .disabled(appState.session == nil)

                Button("Compare Before/After") {
                    appState.compareEnabled.toggle()
                }
                .keyboardShortcut("\\")
                .disabled(appState.session == nil)

                Button("Retouch Mode") {
                    appState.retouchMode.toggle()
                }
                .keyboardShortcut("t")
                .disabled(appState.session == nil)

                Divider()

                Button("Rotate Left") {
                    if let session = appState.session {
                        session.rotation = (session.rotation + 3) & 3
                    }
                }
                .keyboardShortcut("[")
                .disabled(appState.session == nil)

                Button("Rotate Right") {
                    if let session = appState.session {
                        session.rotation = (session.rotation + 1) & 3
                    }
                }
                .keyboardShortcut("]")
                .disabled(appState.session == nil)
            }
        }
    }
}
