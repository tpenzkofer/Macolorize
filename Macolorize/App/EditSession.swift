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
    private(set) var loadError: String?

    /// True once processing has run long enough that the frame on screen is
    /// meaningfully stale. Drives a visible canvas hint — a slow clean pass
    /// otherwise left users comparing (and exporting) against an outdated
    /// preview without noticing (issue #1).
    private(set) var previewIsStale = false
    private var staleHintTask: Task<Void, Never>?
    private(set) var isProcessing = false {
        didSet {
            staleHintTask?.cancel()
            if isProcessing {
                staleHintTask = Task { [weak self] in
                    try? await Task.sleep(for: .milliseconds(400))
                    guard !Task.isCancelled else { return }
                    self?.previewIsStale = true
                }
            } else {
                previewIsStale = false
            }
        }
    }

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

    // MARK: Prefix cache (issue #1 follow-up)

    /// Inputs that determine the pipeline prefix (negative → heals → clean).
    private struct PrefixKey: Equatable {
        var rotation: Int
        var strokes: [BrushStroke]
        var negative: NegativeSettings
        var clean: CleanSettings
    }

    private struct PrefixCache {
        let key: PrefixKey
        /// Materialized full-resolution prefix output (what export continues from).
        let fullRes: PipelineImage
        /// Materialized proxy of `fullRes` (what preview continues from);
        /// its `statsSource` is `fullRes.image`.
        let proxy: PipelineImage
    }

    private var prefixCache: PrefixCache?
    private var prefixTask: (key: PrefixKey, task: Task<PrefixCache, Error>)?

    private var currentPrefixKey: PrefixKey {
        PrefixKey(rotation: rotation, strokes: strokes,
                  negative: recipe.negative, clean: recipe.clean)
    }

    /// The pipeline prefix, computed ONCE at full resolution and cached.
    /// Preview and export both continue from these identical pixels, so the
    /// heals the user sees are exactly the heals the export gets (detection
    /// and inpainting are not scale-stable — running them per rendition made
    /// the export visibly diverge from the preview on grainy scans). Caching
    /// also makes tuning-slider tweaks cheap: only the suffix re-runs.
    private func prefix() async throws -> PrefixCache? {
        guard let loaded else { return nil }
        let key = currentPrefixKey
        if let cache = prefixCache, cache.key == key { return cache }
        if let inflight = prefixTask {
            if inflight.key == key { return try await inflight.task.value }
            inflight.task.cancel()
            prefixTask = nil
        }

        let fullRes = oriented(loaded.fullResolution)
        if !ProcessingEngine.prefixIsActive(recipe: recipe, hasRetouch: !strokes.isEmpty) {
            // Nothing to compute: continue from the source renditions.
            let cache = PrefixCache(
                key: key,
                fullRes: PipelineImage(image: fullRes),
                proxy: PipelineImage(
                    image: oriented(loaded.proxy),
                    scale: loaded.proxy.extent.width / max(loaded.pixelSize.width, 1),
                    statsSource: fullRes))
            prefixCache = cache
            return cache
        }

        let recipe = recipe
        let mask = strokes.isEmpty ? nil : RetouchMask.rasterize(
            strokes: strokes, size: fullRes.extent.size)
        let task = Task<PrefixCache, Error> {
            let (full, proxy) = try await ProcessingEngine.shared.materializedPrefix(
                recipe: recipe,
                on: PipelineImage(image: fullRes),
                retouchMask: mask,
                proxyMaxDimension: ImageLoader.proxyMaxDimension)
            return PrefixCache(key: key, fullRes: full, proxy: proxy)
        }
        prefixTask = (key, task)
        defer { if prefixTask?.key == key { prefixTask = nil } }
        let cache = try await task.value
        prefixCache = cache
        return cache
    }

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
            prefixCache = nil
            schedulePreview()
        } catch {
            loadError = error.localizedDescription
        }
    }

    private func schedulePreview() {
        guard loaded != nil else { return }
        previewTask?.cancel()
        let recipe = recipe
        let hints = colorHints
        let wantsMaskOverlay = showDetectionMask && recipe.clean.enabled
        previewTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled, let self else { return }
            self.isProcessing = true
            defer { self.isProcessing = false }
            do {
                // The cached full-res prefix: heals shown are the heals exported.
                guard let cache = try await self.prefix() else { return }
                try Task.checkCancellation()
                let result = try await ProcessingEngine.shared.run(
                    recipe: recipe, on: cache.proxy, mode: .preview,
                    phase: .suffix, colorHints: hints)
                guard !Task.isCancelled else { return }
                self.processedProxy = result.image
                self.histogram = Histogram.compute(from: result.image,
                                                   ciContext: RenderService.context)
                self.maskOverlay = wantsMaskOverlay ? self.detectionOverlay() : nil
            } catch {
                // Cancelled or failed preview: keep showing the last good frame.
            }
        }
    }

    /// Red-overlay mask of what dust removal detects — computed on the FULL
    /// resolution image (what the pipeline actually heals), then proxied to
    /// the canvas size. The proxy's softened grain hides most real
    /// detections, which is exactly what misled preview before.
    private func detectionOverlay() -> CIImage? {
        guard let loaded else { return nil }
        let mask = CleanStage.dustMask(for: oriented(loaded.fullResolution),
                                       sensitivity: recipe.clean.sensitivity)
        return ImageRasterizer.proxy(of: mask,
                                     maxDimension: ImageLoader.proxyMaxDimension)
    }

    /// Re-runs the full pipeline at native resolution and writes the result
    /// next to the source (`<folder>/Macolorized/<name>_restored.tif`).
    func export(format: ExportFormat = .tiff16) async throws -> URL {
        guard loaded != nil else { throw ImageLoaderError.unreadable(file.url) }
        let destination = ExportNaming.destination(for: file.url, format: format)
        let source = file.url
        let recipe = recipe
        // Continue from the SAME cached full-res prefix the preview used —
        // the retouch mask and heals are already applied in it.
        guard let cache = try await prefix() else {
            throw ImageLoaderError.unreadable(file.url)
        }
        let result = try await ProcessingEngine.shared.run(
            recipe: recipe, on: cache.fullRes, mode: .export,
            phase: .suffix, colorHints: colorHints)
        try await Task.detached(priority: .userInitiated) {
            try ImageExporter.export(result.image,
                                     from: source,
                                     to: destination,
                                     format: format)
        }.value
        return destination
    }
}
