import CoreImage

/// Runs a Recipe's enabled stages, in fixed pipeline order, off the main actor.
public actor ProcessingEngine {
    public static let shared = ProcessingEngine()

    public init() {}

    /// Which part of the pipeline to run. The PREFIX (negative inversion,
    /// manual heals, dust/scratch clean) is expensive, independent of the
    /// tuning sliders, and NOT scale-stable (LaMa fills differ between proxy
    /// and full resolution — issue #1 follow-up). The app computes it once
    /// at full resolution, caches the rasterized result, and runs only the
    /// SUFFIX per preview update.
    public enum Phase: Sendable {
        case all
        case prefix
        case suffix
    }

    /// - Parameters:
    ///   - retouchMask: optional manual inpainting mask (white = heal);
    ///     applied before all recipe stages, like the Clean stage.
    ///   - colorHints: user-painted colorization hints, honored by the
    ///     Colorize stage.
    public func run(recipe: Recipe,
                    on input: PipelineImage,
                    mode: RenderContext.Mode,
                    phase: Phase = .all,
                    retouchMask: CIImage? = nil,
                    colorHints: [ColorHint] = []) async throws -> PipelineImage {
        let context = RenderContext(mode: mode, colorHints: colorHints)
        var working = input
        if phase != .suffix, let retouchMask {
            working.image = try await InpaintStage.apply(to: working.image,
                                                         mask: retouchMask,
                                                         ciContext: context.ciContext)
            working.statsSource = working.image
        }
        for stage in Self.stages(for: recipe, phase: phase) {
            try Task.checkCancellation()
            working = try await stage.apply(to: working, context: context)
        }
        return working
    }

    /// True when the prefix would change any pixels for this recipe/mask —
    /// callers can skip the materialized prefix entirely otherwise.
    public static func prefixIsActive(recipe: Recipe, hasRetouch: Bool) -> Bool {
        recipe.negative.enabled || recipe.clean.enabled || hasRetouch
    }

    /// Runs the prefix ONCE at full resolution and materializes the result,
    /// plus a preview proxy of it. Preview (suffix on the proxy) and export
    /// (suffix on the full-res rendition) then read the SAME healed pixels.
    /// That is the only way dust/scratch removal can be WYSIWYG: detection
    /// and inpainting are not scale-stable — even a slight proxy resample
    /// softens single-pixel grain enough that detection fires on completely
    /// different regions than at full res (issue #1: crisp preview, smeared
    /// export on a grainy archive scan).
    public func materializedPrefix(recipe: Recipe,
                                   on input: PipelineImage,
                                   retouchMask: CIImage?,
                                   proxyMaxDimension: CGFloat) async throws
        -> (fullRes: PipelineImage, proxy: PipelineImage) {
        var full = try await run(recipe: recipe, on: input, mode: .export,
                                 phase: .prefix, retouchMask: retouchMask)
        let ciContext = RenderService.context
        if let solid = ImageRasterizer.materialize(full.image, ciContext: ciContext) {
            full.image = solid
            full.statsSource = solid
        }
        try Task.checkCancellation()
        let scaled = ImageRasterizer.proxy(of: full.image, maxDimension: proxyMaxDimension)
        let proxyImage = scaled === full.image
            ? full.image
            : ImageRasterizer.materialize(scaled, ciContext: ciContext) ?? scaled
        let proxy = PipelineImage(
            image: proxyImage,
            originalExtent: proxyImage.extent,
            scale: proxyImage.extent.width / max(full.image.extent.width, 1),
            statsSource: full.image)
        return (full, proxy)
    }

    /// Fixed order: negative → clean → restore/colorize → sharpen.
    /// Not user-reorderable.
    static func stages(for recipe: Recipe, phase: Phase = .all) -> [any ProcessingStage] {
        var stages: [any ProcessingStage] = []
        if phase != .suffix {
            if recipe.negative.enabled {
                stages.append(NegativeStage(settings: recipe.negative))
            }
            if recipe.clean.enabled {
                stages.append(CleanStage(settings: recipe.clean))
            }
        }
        if phase != .prefix {
            if recipe.restore.enabled {
                stages.append(RestoreStage(settings: recipe.restore))
            }
            if recipe.faces.enabled {
                stages.append(FaceRestoreStage(settings: recipe.faces))
            }
            if recipe.colorize.enabled {
                stages.append(ColorizeStage(settings: recipe.colorize))
            }
            if recipe.denoise.enabled {
                stages.append(DenoiseStage(settings: recipe.denoise))
            }
            if recipe.sharpen.enabled {
                stages.append(SharpenStage(settings: recipe.sharpen))
            }
        }
        return stages
    }
}
