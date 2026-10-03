import SwiftUI
import MacolorizeCore

struct DetailView: View {
    @Environment(AppState.self) private var appState

    @State private var canvasTransform = CanvasTransform()
    @State private var dragStartPan: CGSize?
    @State private var zoomStart: CGFloat?
    @State private var wipe: CGFloat = 0.5
    enum BrushTool: String, CaseIterable, Identifiable {
        case heal
        case color
        var id: String { rawValue }
        var label: String { self == .heal ? "Heal" : "Color" }
        var icon: String { self == .heal ? "bandage" : "paintbrush.pointed" }
    }

    @State private var brushSize: CGFloat = 24
    @State private var brushTool: BrushTool = .heal
    @State private var hintColor: Color = .blue
    @State private var activeStrokePoints: [CGPoint] = []
    @State private var canvasSize: CGSize = .zero

    private var retouchMode: Bool { appState.retouchMode }

    private var compareEnabled: Bool { appState.compareEnabled }

    @Environment(ModelLibrary.self) private var modelLibrary
    private var retouchAvailable: Bool { modelLibrary.has(.lama) }
    @State private var showInspector = true
    @State private var exportMessage: String?
    @State private var exportError: String?
    @State private var isExporting = false

    var body: some View {
        Group {
            if let session = appState.session {
                canvas(for: session)
            } else {
                ContentUnavailableView(
                    "No Slide Selected",
                    systemImage: "sparkles.rectangle.stack",
                    description: Text("Select a scan in the sidebar."))
            }
        }
        .toolbar { toolbarContent }
        .sheet(isPresented: Binding(
            get: { appState.showBatchSheet },
            set: { appState.showBatchSheet = $0 })
        ) {
            BatchSheet()
        }
        .inspector(isPresented: $showInspector) {
            if let session = appState.session {
                InspectorView(session: session)
                    .inspectorColumnWidth(min: 240, ideal: 280)
            }
        }
        .alert("Export Failed", isPresented: .init(
            get: { exportError != nil },
            set: { if !$0 { exportError = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(exportError ?? "")
        }
    }

    @ViewBuilder
    private func canvas(for session: EditSession) -> some View {
        GeometryReader { geometry in
            ZStack {
                CanvasView(image: session.displayImage,
                           comparisonImage: compareEnabled ? session.originalImage : nil,
                           maskOverlay: session.showDetectionMask ? session.maskOverlay : nil,
                           wipe: wipe,
                           transform: canvasTransform)
                    .onAppear { canvasSize = geometry.size }
                    .onChange(of: geometry.size) { _, newSize in canvasSize = newSize }
                if compareEnabled {
                    WipeDivider(wipe: $wipe, size: geometry.size)
                }
                if retouchMode {
                    brushLayer(for: session, viewSize: geometry.size)
                }
                if session.isLoading {
                    ProgressView()
                }
                if let message = exportMessage {
                    VStack {
                        Spacer()
                        Text(message)
                            .font(.callout)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.regularMaterial, in: Capsule())
                            .padding(.bottom, 16)
                    }
                    .transition(.opacity)
                }
            }
        }
        .gesture(retouchMode ? nil : dragGesture)
        .gesture(magnifyGesture)
        .onTapGesture(count: 2) {
            withAnimation(.snappy) { canvasTransform = CanvasTransform() }
        }
        .onChange(of: session.file) {
            canvasTransform = CanvasTransform()
            appState.retouchMode = false
        }
    }

    /// Maps a view point to normalized image coordinates (y down); nil when
    /// the point misses the image.
    private func normalizedImagePoint(_ point: CGPoint,
                                      viewSize: CGSize,
                                      session: EditSession) -> CGPoint? {
        guard let extent = session.originalImage?.extent,
              extent.width > 0, extent.height > 0 else { return nil }
        let fit = min(viewSize.width / extent.width, viewSize.height / extent.height)
        let scale = fit * canvasTransform.zoom
        let x0 = (viewSize.width - extent.width * scale) / 2 + canvasTransform.pan.width
        let y0 = (viewSize.height - extent.height * scale) / 2 + canvasTransform.pan.height
        let normalized = CGPoint(x: (point.x - x0) / (extent.width * scale),
                                 y: (point.y - y0) / (extent.height * scale))
        guard (0...1).contains(normalized.x), (0...1).contains(normalized.y) else { return nil }
        return normalized
    }

    @ViewBuilder
    private func brushLayer(for session: EditSession, viewSize: CGSize) -> some View {
        let imageWidth = session.originalImage?.extent.width ?? 1
        let fit = min(viewSize.width / max(imageWidth, 1),
                      viewSize.height / max(session.originalImage?.extent.height ?? 1, 1))
        let displayScale = fit * canvasTransform.zoom

        Canvas { context, _ in
            guard activeStrokePoints.count > 1 else { return }
            var path = Path()
            // Points are normalized; convert back to view space for feedback.
            let extent = session.originalImage?.extent ?? .zero
            let scale = displayScale
            let x0 = (viewSize.width - extent.width * scale) / 2 + canvasTransform.pan.width
            let y0 = (viewSize.height - extent.height * scale) / 2 + canvasTransform.pan.height
            let viewPoints = activeStrokePoints.map {
                CGPoint(x: x0 + $0.x * extent.width * scale,
                        y: y0 + $0.y * extent.height * scale)
            }
            path.addLines(viewPoints)
            let strokeColor: Color = brushTool == .heal ? .red : hintColor
            context.stroke(path,
                           with: .color(strokeColor.opacity(0.55)),
                           style: StrokeStyle(lineWidth: brushSize, lineCap: .round, lineJoin: .round))
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    if let p = normalizedImagePoint(value.location,
                                                    viewSize: viewSize,
                                                    session: session) {
                        activeStrokePoints.append(p)
                    }
                }
                .onEnded { _ in
                    guard !activeStrokePoints.isEmpty else { return }
                    // Brush radius: view points → normalized image width.
                    let radius = (brushSize / 2) / max(imageWidth * displayScale, 1)
                    let stroke = BrushStroke(points: activeStrokePoints, radius: radius)
                    switch brushTool {
                    case .heal:
                        session.strokes.append(stroke)
                    case .color:
                        let rgb = NSColor(hintColor).usingColorSpace(.sRGB) ?? .blue
                        session.colorHints.append(ColorHint(stroke: stroke,
                                                            red: rgb.redComponent,
                                                            green: rgb.greenComponent,
                                                            blue: rgb.blueComponent))
                        // Hints only show through the colorizer.
                        session.recipe.colorize.enabled = true
                    }
                    activeStrokePoints = []
                }
        )
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                if dragStartPan == nil { dragStartPan = canvasTransform.pan }
                canvasTransform.pan = CGSize(
                    width: (dragStartPan?.width ?? 0) + value.translation.width,
                    height: (dragStartPan?.height ?? 0) + value.translation.height)
            }
            .onEnded { _ in dragStartPan = nil }
    }

    private var magnifyGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                if zoomStart == nil { zoomStart = canvasTransform.zoom }
                canvasTransform.zoom = min(max((zoomStart ?? 1) * value.magnification, 0.2), 64)
            }
            .onEnded { _ in zoomStart = nil }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            if appState.session?.isProcessing == true {
                ProgressView()
                    .controlSize(.small)
            }

            if retouchMode {
                Picker("Tool", selection: $brushTool) {
                    ForEach(BrushTool.allCases) { tool in
                        Label(tool.label, systemImage: tool.icon).tag(tool)
                    }
                }
                .pickerStyle(.segmented)
                .help("Heal removes defects; Color paints colorization hints")

                if brushTool == .color {
                    ColorPicker("Hint Color", selection: $hintColor, supportsOpacity: false)
                        .labelsHidden()
                        .help("The color this region should get")
                }

                Slider(value: $brushSize, in: 6...80) {
                    Text("Brush Size")
                }
                .frame(width: 100)
                .help("Brush size")

                Button {
                    guard let session = appState.session else { return }
                    switch brushTool {
                    case .heal where !session.strokes.isEmpty:
                        session.strokes.removeLast()
                    case .color where !session.colorHints.isEmpty:
                        session.colorHints.removeLast()
                    default:
                        break
                    }
                } label: {
                    Label("Undo Stroke", systemImage: "arrow.uturn.backward")
                }
                .disabled({
                    guard let session = appState.session else { return true }
                    return brushTool == .heal
                        ? session.strokes.isEmpty
                        : session.colorHints.isEmpty
                }())
            }

            Toggle(isOn: Binding(
                get: { appState.retouchMode },
                set: { appState.retouchMode = $0 })
            ) {
                Label("Retouch", systemImage: "bandage")
            }
            .disabled(appState.session?.loaded == nil || !retouchAvailable)
            .help(retouchAvailable
                  ? "Paint over dust or scratches to heal them (⌘T)"
                  : "LaMa model not installed — run `make models`")

            Menu {
                Button("Zoom to Fit") {
                    withAnimation(.snappy) { canvasTransform = CanvasTransform() }
                }
                Button("Actual Pixels (100%)") {
                    guard let session = appState.session,
                          let extent = session.originalImage?.extent,
                          canvasSize.width > 0, extent.width > 0 else { return }
                    let fit = min(canvasSize.width / extent.width,
                                  canvasSize.height / extent.height)
                    let screenScale = NSScreen.main?.backingScaleFactor ?? 2
                    // One FULL-RESOLUTION pixel per device pixel — the canvas
                    // holds the proxy, so scale up by the proxy ratio.
                    let ratio = session.fullResolutionPerProxyPixel
                    canvasTransform = CanvasTransform(zoom: ratio / (fit * screenScale),
                                                      pan: .zero)
                }
            } label: {
                Label("Zoom", systemImage: "plus.magnifyingglass")
            }
            .disabled(appState.session?.loaded == nil)

            Toggle(isOn: Binding(
                get: { appState.session?.showDetectionMask ?? false },
                set: { appState.session?.showDetectionMask = $0 })
            ) {
                Label("Show Mask", systemImage: "eye.trianglebadge.exclamationmark")
            }
            .disabled(appState.session?.recipe.clean.enabled != true)
            .help("Highlight what dust removal will heal")

            Toggle(isOn: Binding(
                get: { appState.compareEnabled },
                set: { appState.compareEnabled = $0 })
            ) {
                Label("Compare", systemImage: "rectangle.split.2x1")
            }
            .disabled(appState.session?.loaded == nil)
            .help("Show original left of the divider (⌘\\)")

            Button {
                exportCurrent()
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .disabled(appState.session?.loaded == nil || isExporting)
            .help("Export to a Macolorized folder next to the original")

            Button {
                appState.showBatchSheet = true
            } label: {
                Label("Batch", systemImage: "square.stack.3d.down.right")
            }
            .disabled(appState.session == nil || appState.files.isEmpty)
            .help("Apply the current adjustments to every scan in the folder")

            Button {
                showInspector.toggle()
            } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
            }
        }
    }

    private func exportCurrent() {
        guard let session = appState.session else { return }
        isExporting = true
        Task {
            defer { isExporting = false }
            do {
                let url = try await session.export()
                withAnimation { exportMessage = "Exported \(url.lastPathComponent)" }
                try? await Task.sleep(for: .seconds(3))
                withAnimation { exportMessage = nil }
            } catch {
                exportError = error.localizedDescription
            }
        }
    }
}

/// Draggable vertical divider for the before/after wipe.
private struct WipeDivider: View {
    @Binding var wipe: CGFloat
    let size: CGSize

    var body: some View {
        let x = wipe * size.width
        ZStack {
            Rectangle()
                .fill(.white.opacity(0.9))
                .frame(width: 1.5)
                .shadow(radius: 2)
            Image(systemName: "arrow.left.and.right.circle.fill")
                .font(.title2)
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(.white)
                .background(Circle().fill(.black.opacity(0.5)))
        }
        .frame(maxHeight: .infinity)
        .position(x: x, y: size.height / 2)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    wipe = min(max(value.location.x / max(size.width, 1), 0.02), 0.98)
                }
        )
    }
}
