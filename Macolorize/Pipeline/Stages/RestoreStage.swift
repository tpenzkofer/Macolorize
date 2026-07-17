import CoreImage
import CoreImage.CIFilterBuiltins

/// Classical fade restoration for aged color slides: per-channel percentile
/// stretch (the canonical fix for dye fade) with optional gray-world midtone
/// balancing. Resolution-independent — parameters derived from a small
/// analysis render apply identically to preview and export.
public struct RestoreStage: ProcessingStage {
    public let id = StageID.restore
    let settings: RestoreSettings

    public init(settings: RestoreSettings) {
        self.settings = settings
    }

    public func apply(to input: PipelineImage, context: RenderContext) async throws -> PipelineImage {
        let stats = try Self.analyze(input.image, ciContext: context.ciContext)
        let transform = Self.channelTransform(from: stats, settings: settings)

        var restored = input.image.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: transform.scale.r, y: 0, z: 0, w: 0),
            "inputGVector": CIVector(x: 0, y: transform.scale.g, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: transform.scale.b, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            "inputBiasVector": CIVector(x: transform.bias.r, y: transform.bias.g, z: transform.bias.b, w: 0),
        ])
        restored = restored.applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1),
        ])

        if settings.strength < 1 {
            restored = restored.applyingFilter("CIDissolveTransition", parameters: [
                kCIInputImageKey: input.image,
                kCIInputTargetImageKey: restored,
                kCIInputTimeKey: settings.strength,
            ]).cropped(to: input.image.extent)
        }

        var output = input
        output.image = restored
        return output
    }

    // MARK: - Analysis

    struct ChannelStats {
        var samples: (r: [Float], g: [Float], b: [Float])

        func percentile(_ p: Double, of channel: [Float]) -> Float {
            guard !channel.isEmpty else { return 0 }
            let index = Int(Double(channel.count - 1) * (p / 100.0).clamped(to: 0...1))
            return channel[index]
        }

        func mean(of channel: [Float]) -> Float {
            guard !channel.isEmpty else { return 0 }
            return channel.reduce(0, +) / Float(channel.count)
        }
    }

    struct ChannelTransform {
        var scale: (r: CGFloat, g: CGFloat, b: CGFloat)
        var bias: (r: CGFloat, g: CGFloat, b: CGFloat)
    }

    static let analysisMaxDimension: CGFloat = 256

    /// Renders a small linear-light bitmap and returns sorted per-channel samples.
    static func analyze(_ image: CIImage, ciContext: CIContext) throws -> ChannelStats {
        let extent = image.extent
        let downScale = min(1, analysisMaxDimension / max(extent.width, extent.height))
        let small = image
            .transformed(by: .init(scaleX: downScale, y: downScale))
        let width = max(Int(small.extent.width.rounded(.down)), 1)
        let height = max(Int(small.extent.height.rounded(.down)), 1)

        var buffer = [Float](repeating: 0, count: width * height * 4)
        let rowBytes = width * 4 * MemoryLayout<Float>.size
        buffer.withUnsafeMutableBytes { raw in
            ciContext.render(small,
                             toBitmap: raw.baseAddress!,
                             rowBytes: rowBytes,
                             bounds: CGRect(x: small.extent.minX, y: small.extent.minY,
                                            width: CGFloat(width), height: CGFloat(height)),
                             format: .RGBAf,
                             colorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB))
        }

        let pixelCount = width * height
        var r = [Float](); r.reserveCapacity(pixelCount)
        var g = [Float](); g.reserveCapacity(pixelCount)
        var b = [Float](); b.reserveCapacity(pixelCount)
        for i in 0..<pixelCount {
            r.append(buffer[i * 4])
            g.append(buffer[i * 4 + 1])
            b.append(buffer[i * 4 + 2])
        }
        r.sort(); g.sort(); b.sort()
        return ChannelStats(samples: (r, g, b))
    }

    /// Combined per-channel linear map: out = gain * (x - lo) / (hi - lo).
    static func channelTransform(from stats: ChannelStats,
                                 settings: RestoreSettings) -> ChannelTransform {
        let (r, g, b) = stats.samples
        let blackP = settings.blackPercentile
        let whiteP = settings.whitePercentile

        var lo: (r: Float, g: Float, b: Float)
        var hi: (r: Float, g: Float, b: Float)
        if settings.whiteBalance == .off {
            // Luminance-based: one stretch for all channels, no color shift.
            let luminance = zip(zip(r, g), b).map { rg, b in
                0.2126 * rg.0 + 0.7152 * rg.1 + 0.0722 * b
            }.sorted()
            let lstats = ChannelStats(samples: (luminance, luminance, luminance))
            let l0 = lstats.percentile(blackP, of: luminance)
            let l1 = lstats.percentile(whiteP, of: luminance)
            lo = (l0, l0, l0)
            hi = (l1, l1, l1)
        } else {
            lo = (stats.percentile(blackP, of: r),
                  stats.percentile(blackP, of: g),
                  stats.percentile(blackP, of: b))
            hi = (stats.percentile(whiteP, of: r),
                  stats.percentile(whiteP, of: g),
                  stats.percentile(whiteP, of: b))
        }

        func safeRange(_ lo: Float, _ hi: Float) -> Float {
            max(hi - lo, 1e-4)
        }

        var gain: (r: Float, g: Float, b: Float) = (1, 1, 1)
        if settings.whiteBalance == .grayWorld {
            // Means of the *stretched* channels, then gains equalizing them.
            let meanR = (stats.mean(of: r) - lo.r) / safeRange(lo.r, hi.r)
            let meanG = (stats.mean(of: g) - lo.g) / safeRange(lo.g, hi.g)
            let meanB = (stats.mean(of: b) - lo.b) / safeRange(lo.b, hi.b)
            let target = (meanR + meanG + meanB) / 3
            func clampGain(_ g: Float) -> Float { min(max(g, 0.5), 2.0) }
            if target > 1e-4 {
                gain = (clampGain(target / max(meanR, 1e-4)),
                        clampGain(target / max(meanG, 1e-4)),
                        clampGain(target / max(meanB, 1e-4)))
            }
        }

        return ChannelTransform(
            scale: (CGFloat(gain.r / safeRange(lo.r, hi.r)),
                    CGFloat(gain.g / safeRange(lo.g, hi.g)),
                    CGFloat(gain.b / safeRange(lo.b, hi.b))),
            bias: (CGFloat(-gain.r * lo.r / safeRange(lo.r, hi.r)),
                   CGFloat(-gain.g * lo.g / safeRange(lo.g, hi.g)),
                   CGFloat(-gain.b * lo.b / safeRange(lo.b, hi.b))))
    }
}

extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
