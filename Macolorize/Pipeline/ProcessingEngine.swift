import CoreImage

/// Runs a Recipe's enabled stages, in fixed pipeline order, off the main actor.
public actor ProcessingEngine {
    public static let shared = ProcessingEngine()

    public init() {}

    /// - Parameters:
    ///   - retouchMask: optional manual inpainting mask (white = heal);
    ///     applied before all recipe stages, like the Clean stage.
    ///   - colorHints: user-painted colorization hints, honored by the
    ///     Colorize stage.
    public func run(recipe: Recipe,
                    on input: PipelineImage,
                    mode: RenderContext.Mode,
                    retouchMask: CIImage? = nil,
                    colorHints: [ColorHint] = []) async throws -> PipelineImage {
        let context = RenderContext(mode: mode, colorHints: colorHints)
        var working = input
        if let retouchMask {
            working.image = try await InpaintStage.apply(to: working.image,
                                                         mask: retouchMask,
                                                         ciContext: context.ciContext)
        }
        for stage in Self.stages(for: recipe) {
            try Task.checkCancellation()
            working = try await stage.apply(to: working, context: context)
        }
        return working
    }

    /// Fixed order: clean → restore/colorize → sharpen. Not user-reorderable.
    private static func stages(for recipe: Recipe) -> [any ProcessingStage] {
        var stages: [any ProcessingStage] = []
        if recipe.negative.enabled {
            stages.append(NegativeStage(settings: recipe.negative))
        }
        if recipe.clean.enabled {
            stages.append(CleanStage(settings: recipe.clean))
        }
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
        return stages
    }
}
