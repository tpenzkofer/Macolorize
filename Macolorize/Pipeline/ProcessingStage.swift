import CoreImage

public enum StageID: String, Sendable, CaseIterable {
    case negative
    case clean
    case restore
    case faces
    case colorize
    case denoise
    case sharpen
}

/// The image flowing through the pipeline. CIImage is immutable, so this is
/// safe to pass across concurrency domains.
public struct PipelineImage: @unchecked Sendable {
    public var image: CIImage
    public let originalExtent: CGRect
    /// 1.0 = full resolution; < 1 when processing a preview proxy.
    public let scale: CGFloat
    /// The image statistics-driven stages (negative, restore) must analyze:
    /// always the FULL-RESOLUTION rendition, even when `image` is a preview
    /// proxy. Proxies are resampled in gamma space with noise averaging, so
    /// their linear-light percentiles genuinely differ from full res —
    /// analyzing the proxy made preview tone curves diverge from export
    /// (issue #1). Stages that change pixel values apply the same transform
    /// here so later analyses stay consistent.
    public var statsSource: CIImage
    /// Union of everything inpainting repainted (white = healed): manual
    /// strokes plus automatic dust/scratch detections, at `image` resolution.
    /// Drives the Show Mask overlay — pixels changing with no red shown was
    /// the biggest trust breaker of issue #1 (the overlay used to show only
    /// the classical dust mask, hiding scratch-ML hits and manual heals).
    public var healMask: CIImage?

    public init(image: CIImage, originalExtent: CGRect? = nil, scale: CGFloat = 1,
                statsSource: CIImage? = nil, healMask: CIImage? = nil) {
        self.image = image
        self.originalExtent = originalExtent ?? image.extent
        self.scale = scale
        self.statsSource = statsSource ?? image
        self.healMask = healMask
    }

    public mutating func accumulateHealMask(_ mask: CIImage) {
        healMask = (healMask.map {
            mask.applyingFilter("CIMaximumCompositing", parameters: [
                kCIInputBackgroundImageKey: $0,
            ])
        } ?? mask).cropped(to: image.extent)
    }
}

public struct RenderContext: Sendable {
    public enum Mode: Sendable, Equatable {
        case preview
        case export
    }

    public let mode: Mode
    /// User-painted colorization hints (applied by ColorizeStage).
    public let colorHints: [ColorHint]
    public var ciContext: CIContext { RenderService.context }

    public init(mode: Mode, colorHints: [ColorHint] = []) {
        self.mode = mode
        self.colorHints = colorHints
    }
}

public protocol ProcessingStage: Sendable {
    var id: StageID { get }
    func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage
}
