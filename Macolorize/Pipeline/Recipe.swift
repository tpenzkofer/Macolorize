import Foundation

/// All processing parameters for one pass over an image. Drives the
/// interactive inspector, batch runs, and the CLI alike. Stages for
/// colorize/clean/sharpen join in later phases.
public struct Recipe: Codable, Sendable, Equatable {
    public var negative = NegativeSettings()
    public var clean = CleanSettings()
    public var restore = RestoreSettings()
    public var faces = FaceSettings()
    public var colorize = ColorizeSettings()
    public var denoise = DenoiseSettings()
    public var sharpen = SharpenSettings()

    public init() {}

    // Tolerant decoding: recipes/presets saved by older versions may lack
    // newer stages; missing keys fall back to defaults.
    private enum CodingKeys: String, CodingKey {
        case negative, clean, restore, faces, colorize, denoise, sharpen
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        negative = try container.decodeIfPresent(NegativeSettings.self, forKey: .negative) ?? NegativeSettings()
        clean = try container.decodeIfPresent(CleanSettings.self, forKey: .clean) ?? CleanSettings()
        restore = try container.decodeIfPresent(RestoreSettings.self, forKey: .restore) ?? RestoreSettings()
        faces = try container.decodeIfPresent(FaceSettings.self, forKey: .faces) ?? FaceSettings()
        colorize = try container.decodeIfPresent(ColorizeSettings.self, forKey: .colorize) ?? ColorizeSettings()
        denoise = try container.decodeIfPresent(DenoiseSettings.self, forKey: .denoise) ?? DenoiseSettings()
        sharpen = try container.decodeIfPresent(SharpenSettings.self, forKey: .sharpen) ?? SharpenSettings()
    }
}

public struct DenoiseSettings: Codable, Sendable, Equatable {
    public var enabled = false
    /// 0…1 → DRUNet sigma 0…25; ~0.3 suits typical film grain.
    public var strength: Double = 0.3

    public init() {}
}

public struct NegativeSettings: Codable, Sendable, Equatable {
    public var enabled = false

    public init() {}
}

public struct FaceSettings: Codable, Sendable, Equatable {
    public var enabled = false
    /// Blend between the original face (0) and the fully restored one (1).
    public var strength: Double = 0.8

    public init() {}
}

public struct CleanSettings: Codable, Sendable, Equatable {
    public var enabled = false
    /// 0 = only extreme outliers, 1 = aggressive detection.
    public var sensitivity: Double = 0.5
    /// Also run the ML scratch detector (long structured scratches).
    public var detectScratches = true

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case enabled, sensitivity, detectScratches
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
        sensitivity = try container.decodeIfPresent(Double.self, forKey: .sensitivity) ?? 0.5
        detectScratches = try container.decodeIfPresent(Bool.self, forKey: .detectScratches) ?? true
    }
}

public enum ColorizeVariant: String, Codable, Sendable, CaseIterable, Identifiable {
    /// modelscope checkpoint — natural, faithful colors.
    case balanced
    /// artistic checkpoint — bolder, more saturated interpretation.
    case artistic

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .balanced: "Natural"
        case .artistic: "Vivid"
        }
    }
}

public struct ColorizeSettings: Codable, Sendable, Equatable {
    public var enabled = false
    public var variant: ColorizeVariant = .balanced
    /// Multiplier on predicted chroma (a/b); 1 = model output.
    public var saturation: Double = 1.0
    /// Blend between original (0) and fully colorized (1).
    public var strength: Double = 1.0

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case enabled, variant, saturation, strength
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
        variant = try container.decodeIfPresent(ColorizeVariant.self, forKey: .variant) ?? .balanced
        saturation = try container.decodeIfPresent(Double.self, forKey: .saturation) ?? 1.0
        strength = try container.decodeIfPresent(Double.self, forKey: .strength) ?? 1.0
    }
}

public enum WhiteBalanceMode: String, Codable, Sendable, CaseIterable, Identifiable {
    /// Contrast stretch only, applied equally to all channels — no color shift.
    case off
    /// Per-channel stretch plus midtone gains that equalize channel means.
    case grayWorld
    /// Per-channel stretch: brightest/darkest percentiles define white/black.
    case whitePatch

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .off: "Contrast Only"
        case .grayWorld: "Gray World"
        case .whitePatch: "White Patch"
        }
    }
}

public enum SharpenModel: String, Codable, Sendable, CaseIterable, Identifiable {
    /// RRDBNet ×2 — best quality for already-high-res scans.
    case x2plus
    /// SRVGGNetCompact ×4 — faster, stronger upscale for low-res sources.
    case generalX4

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .x2plus: "Quality 2×"
        case .generalX4: "Fast 4×"
        }
    }
}

public struct SharpenSettings: Codable, Sendable, Equatable {
    public var enabled = false
    public var model: SharpenModel = .x2plus

    public init() {}
}

public struct RestoreSettings: Codable, Sendable, Equatable {
    public var enabled = false
    /// Percentile treated as black, in percent (0…1 typical).
    public var blackPercentile: Double = 0.1
    /// Percentile treated as white, in percent (99…100 typical).
    public var whitePercentile: Double = 99.9
    public var whiteBalance: WhiteBalanceMode = .grayWorld
    /// Blend between original (0) and fully restored (1).
    public var strength: Double = 1.0

    public init() {}
}
