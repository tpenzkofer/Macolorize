import SwiftUI
import MacolorizeCore

/// Sheet that applies the current recipe to every scan in the open folder.
struct BatchSheet: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss

    @State private var queue = BatchQueue()
    @State private var format: ExportFormat = .tiff16

    private var urls: [URL] { appState.files.map(\.url) }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Batch Process \(urls.count) Scans")
                        .font(.headline)
                    Text("Applies the current adjustments; results go to a Macolorized subfolder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Picker("Format", selection: $format) {
                    Text("16-bit TIFF").tag(ExportFormat.tiff16)
                    Text("PNG").tag(ExportFormat.png)
                    Text("JPEG").tag(ExportFormat.jpeg)
                }
                .fixedSize()
                .disabled(queue.isRunning)
            }
            .padding()

            Table(queue.rows.isEmpty ? urls.map { BatchQueue.Row(url: $0) } : queue.rows) {
                TableColumn("File") { row in
                    Text(row.name)
                }
                TableColumn("Status") { row in
                    switch row.status {
                    case .pending:
                        Text("Waiting").foregroundStyle(.secondary)
                    case .running:
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("Processing…")
                        }
                    case .finished(let url):
                        Label(url.lastPathComponent, systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    case .failed(let message):
                        Label(message, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                    }
                }
                .width(min: 180)
            }
            .frame(minHeight: 260)

            Divider()

            HStack {
                if queue.isRunning || queue.completedCount > 0 {
                    ProgressView(value: queue.progress)
                        .frame(maxWidth: 220)
                    Text("\(queue.completedCount)/\(queue.rows.count)")
                        .font(.caption)
                        .monospacedDigit()
                }
                Spacer()
                Button(queue.isRunning ? "Cancel" : "Close") {
                    if queue.isRunning {
                        queue.cancel()
                    } else {
                        dismiss()
                    }
                }
                Button("Start") {
                    guard let recipe = appState.session?.recipe else { return }
                    queue.start(urls: urls, recipe: recipe, format: format)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(queue.isRunning || urls.isEmpty || appState.session == nil)
            }
            .padding()
        }
        .frame(minWidth: 560)
    }
}
