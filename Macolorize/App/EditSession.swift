import SwiftUI
import CoreImage
import MacolorizeCore

/// Per-image editing state: the loaded scan, the current Recipe, and the live
/// preview. Parameter changes are debounced and processed latest-wins.
@MainActor @Observable
final class EditSession {
    let file: SlideFile

    private(set) var loaded: LoadedImage?
    private(set) var isLoading = false
    private(set) var isProcessing = false
    private(set) var loadError: String?

    var recipe = Recipe() {
        didSet { if recipe != oldValue { schedulePreview(); scheduleSidecarSave() } }
    }

    /// Manual retouch strokes (normalized coordinates); healed via LaMa.
    var strokes: [BrushStroke] = [] {
        didSet { if strokes != oldValue { schedulePreview(); scheduleSidecarSave() } }
    }

    /// Painted colorization hints (only used while Colorize is enabled).
    var colorHints: [ColorHint] = [] {
        didSet { if colorHints != oldValue { schedulePreview(); scheduleSidecarSave() } }
    }

    /// Quarter-turns clockwise (0–3), applied before everything else.
    var rotation: Int = 0 {
        didSet { if rotation != oldValue { schedulePreview(); scheduleSidecarSave() } }
    }

    /// Show the dust/scratch detection mask as a red overlay.
    var showDetectionMask = false {
        didSet { if showDetectionMask != oldValue { schedulePreview() } }
    }

    private(set) var histogram: HistogramData?
    private(set) var maskOverlay: CIImage?

    private func oriented(_ image: CIImage) -> CIImage {
        switch rotation & 3 {
        case 1: image.oriented(.right)
        case 2: image.oriented(.down)
        case 3: image.oriented(.left)
        default: image
        }
    }

    private var sidecarTask: Task<Void, Never>?

    private func scheduleSidecarSave() {
        sidecarTask?.cancel()
        let sidecar = SessionSidecar(recipe: recipe, strokes: strokes,
                                     colorHints: colorHints, rotation: rotation)
        let source = file.url
        sidecarTask = Task.detached(priority: .utility) {
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            SidecarStore.save(sidecar, for: source)
        }
    }

    private(set) var processedProxy: CIImage?
    private var previewTask: Task<Void, Never>?

    /// Untouched proxy, for the compare wipe.
    var originalImage: CIImage? { loaded.map { oriented($0.proxy) } }

    /// How many full-resolution pixels one proxy pixel spans (≥ 1). Used so
    /// "Actual Pixels" zoom means real image pixels, not proxy pixels.
    var fullResolutionPerProxyPixel: CGFloat {
        guard let loaded else { return 1 }
        return max(loaded.pixelSize.width / max(loaded.proxy.extent.width, 1), 1)
    }
    /// What the canvas shows.
    var displayImage: CIImage? { processedProxy ?? originalImage }

    init(file: SlideFile, recipe: Recipe = Recipe()) {
        self.file = file
        self.recipe = recipe
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        let url = file.url
        do {
            // A sidecar from a previous session wins over the carried recipe.
            if let sidecar = SidecarStore.load(for: url) {
                recipe = sidecar.recipe
                strokes = sidecar.strokes
                colorHints = sidecar.colorHints
                rotation = sidecar.rotation
            }
            loaded = try await Task.detached(priority: .userInitiated) {
                try ImageLoader.load(url)
            }.value
            schedulePreview()
        } catch {
            loadError = error.localizedDescription
        }
    }

    private func schedulePreview() {
        guard let loaded else { return }
        previewTask?.cancel()
        let recipe = recipe
        let proxy = oriented(loaded.proxy)
        let input = PipelineImage(
            image: proxy,
            originalExtent: proxy.extent,
            scale: proxy.extent.width / max(loaded.pixelSize.width, 1),
            // Analyzing stages must see full resolution even in preview, so
            // preview tone curves match the export exactly (issue #1).
            statsSource: oriented(loaded.fullResolution))
        let mask = strokes.isEmpty ? nil : RetouchMask.rasterize(
            strokes: strokes, size: proxy.extent.size)
        let hints = colorHints
        let wantsMaskOverlay = showDetectionMask && recipe.clean.enabled
        previewTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled, let self else { return }
            self.isProcessing = true
            defer { self.isProcessing = false }
            do {
                let result = try await ProcessingEngine.shared.run(
                    recipe: recipe, on: input, mode: .preview,
                    retouchMask: mask, colorHints: hints)
                guard !Task.isCancelled else { return }
                self.processedProxy = result.image
                self.histogram = Histogram.compute(from: result.image,
                                                   ciContext: RenderService.context)
                self.maskOverlay = wantsMaskOverlay
                    ? CleanStage.dustMask(for: proxy, sensitivity: recipe.clean.sensitivity)
                    : nil
            } catch {
                // Cancelled or failed preview: keep showing the last good frame.
            }
        }
    }

    /// Re-runs the full pipeline at native resolution and writes the result
    /// next to the source (`<folder>/Macolorized/<name>_restored.tif`).
    func export(format: ExportFormat = .tiff16) async throws -> URL {
        guard let loaded else { throw ImageLoaderError.unreadable(file.url) }
        let destination = ExportNaming.destination(for: file.url, format: format)
        let source = file.url
        let recipe = recipe
        let fullRes = oriented(loaded.fullResolution)
        let input = PipelineImage(image: fullRes)
        let mask = strokes.isEmpty ? nil : RetouchMask.rasterize(
            strokes: strokes, size: fullRes.extent.size)
        let result = try await ProcessingEngine.shared.run(
            recipe: recipe, on: input, mode: .export,
            retouchMask: mask, colorHints: colorHints)
        try await Task.detached(priority: .userInitiated) {
            try ImageExporter.export(result.image,
                                     from: source,
                                     to: destination,
                                     format: format)
        }.value
        return destination
    }
}
