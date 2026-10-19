import Testing
import CoreImage
@testable import MacolorizeCore

/// Runs RestoreStage on a synthetic faded image and checks the correction
/// statistically: contrast expanded, channel casts equalized.
struct PipelineSmokeTests {
    /// A "faded slide": strong red cast, lifted blacks, compressed range.
    private func makeFadedImage() -> CIImage {
        let size = 128
        var pixels = [UInt8](repeating: 0, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let i = (y * size + x) * 4
                let gradient = Double(x) / Double(size - 1)
                pixels[i] = UInt8(120 + gradient * 100)      // red: 120–220
                pixels[i + 1] = UInt8(60 + gradient * 80)    // green: 60–140
                pixels[i + 2] = UInt8(40 + gradient * 50)    // blue: 40–90
                pixels[i + 3] = 255
            }
        }
        let data = Data(pixels)
        return CIImage(bitmapData: data,
                       bytesPerRow: size * 4,
                       size: CGSize(width: size, height: size),
                       format: .RGBA8,
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    @Test func restoreExpandsContrastAndBalancesCast() async throws {
        let faded = makeFadedImage()

        var settings = RestoreSettings()
        settings.enabled = true
        settings.whiteBalance = .grayWorld
        let stage = RestoreStage(settings: settings)

        let output = try await stage.apply(
            to: PipelineImage(image: faded),
            context: RenderContext(mode: .export))

        let before = try RestoreStage.analyze(faded, ciContext: RenderService.context)
        let after = try RestoreStage.analyze(output.image, ciContext: RenderService.context)

        func range(_ samples: [Float]) -> Float {
            (samples.last ?? 0) - (samples.first ?? 0)
        }
        // Contrast expanded on every channel.
        #expect(range(after.samples.r) > range(before.samples.r))
        #expect(range(after.samples.g) > range(before.samples.g))
        #expect(range(after.samples.b) > range(before.samples.b))

        // The red cast is reduced: channel means move closer together.
        func spread(_ stats: RestoreStage.ChannelStats) -> Float {
            let means = [stats.mean(of: stats.samples.r),
                         stats.mean(of: stats.samples.g),
                         stats.mean(of: stats.samples.b)]
            return means.max()! - means.min()!
        }
        #expect(spread(after) < spread(before) / 2)
    }

    /// Issue #1: restore's tone analysis must not depend on the resolution
    /// it sees — the preview analyzes a proxy while export analyzes full
    /// res, and with an aliasing downsample the two computed different
    /// curves. Statistics from a noisy image and its half-size rendition
    /// must agree closely.
    @Test func restoreAnalysisIsScaleInvariant() throws {
        // Noisy gradient, deterministic: high-frequency content that an
        // aliasing downsample mis-samples.
        let width = 1600, height = 1200
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func rand() -> Float {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Float(seed % 10_000) / 10_000
        }
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let base = Float(x) / Float(width) * 0.6 + 0.2
                let noise = (rand() - 0.5) * 0.3
                let value = UInt8(max(0, min(255, (base + noise) * 255)))
                pixels[i] = value; pixels[i + 1] = value; pixels[i + 2] = value
                pixels[i + 3] = 255
            }
        }
        let fullRes = CIImage(bitmapData: Data(pixels), bytesPerRow: width * 4,
                              size: CGSize(width: width, height: height),
                              format: .RGBA8,
                              colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        // The "proxy": a properly prefiltered half-size rendition.
        let halfRes = fullRes
            .clampedToExtent()
            .applyingFilter("CILanczosScaleTransform", parameters: [
                kCIInputScaleKey: 0.5, kCIInputAspectRatioKey: 1.0,
            ])
            .cropped(to: CGRect(x: 0, y: 0, width: width / 2, height: height / 2))

        let statsFull = try RestoreStage.analyze(fullRes, ciContext: RenderService.context)
        let statsHalf = try RestoreStage.analyze(halfRes, ciContext: RenderService.context)

        for percentile in [0.1, 5.0, 50.0, 95.0, 99.9] {
            let pf = statsFull.percentile(percentile, of: statsFull.samples.g)
            let ph = statsHalf.percentile(percentile, of: statsHalf.samples.g)
            #expect(abs(pf - ph) < 0.015,
                    "p\(percentile): full \(pf) vs half \(ph)")
        }
    }

    /// Issue #1 regression: a preview pipeline (proxy working image +
    /// full-res statsSource) must apply exactly the same tone curve as the
    /// export pipeline. The proxy is built like ImageIO builds it — averaged
    /// in GAMMA space — which changes its linear statistics; analyzing the
    /// proxy (old behavior) produced visibly different curves.
    @Test func previewToneCurveMatchesExport() async throws {
        let width = 1024, height = 768
        var seed: UInt64 = 0x1234ABCD9876EF01
        func rand() -> Float {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Float(seed % 10_000) / 10_000
        }
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                // Flat probe patch in the center; noisy gradient elsewhere.
                let inPatch = abs(x - width / 2) < 40 && abs(y - height / 2) < 40
                let value: Float = inPatch
                    ? 0.42
                    : Float(x) / Float(width) * 0.55 + 0.15 + (rand() - 0.5) * 0.35
                let v = UInt8(max(0, min(255, value * 255)))
                pixels[i] = v; pixels[i + 1] = v; pixels[i + 2] = v
                pixels[i + 3] = 255
            }
        }
        // Gamma-space 2x box downsample (what ImageIO thumbnails do).
        let pw = width / 2, ph = height / 2
        var proxyPixels = [UInt8](repeating: 0, count: pw * ph * 4)
        for y in 0..<ph {
            for x in 0..<pw {
                var sum = 0
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    sum += Int(pixels[((y * 2 + dy) * width + (x * 2 + dx)) * 4])
                }
                let v = UInt8(sum / 4)
                let i = (y * pw + x) * 4
                proxyPixels[i] = v; proxyPixels[i + 1] = v; proxyPixels[i + 2] = v
                proxyPixels[i + 3] = 255
            }
        }
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)
        let full = CIImage(bitmapData: Data(pixels), bytesPerRow: width * 4,
                           size: CGSize(width: width, height: height),
                           format: .RGBA8, colorSpace: srgb)
        let proxy = CIImage(bitmapData: Data(proxyPixels), bytesPerRow: pw * 4,
                            size: CGSize(width: pw, height: ph),
                            format: .RGBA8, colorSpace: srgb)

        var recipe = Recipe()
        recipe.restore.enabled = true

        let exportOut = try await ProcessingEngine.shared.run(
            recipe: recipe, on: PipelineImage(image: full), mode: .export)
        let previewOut = try await ProcessingEngine.shared.run(
            recipe: recipe,
            on: PipelineImage(image: proxy, statsSource: full),
            mode: .preview)

        func patchMean(_ image: CIImage, half: Bool) -> Float {
            let size = 16
            let cx = (half ? pw : width) / 2 - size / 2
            let cy = (half ? ph : height) / 2 - size / 2
            var buffer = [Float](repeating: 0, count: size * size * 4)
            buffer.withUnsafeMutableBytes { raw in
                RenderService.context.render(image,
                                             toBitmap: raw.baseAddress!,
                                             rowBytes: size * 16,
                                             bounds: CGRect(x: cx, y: cy,
                                                            width: size, height: size),
                                             format: .RGBAf,
                                             colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
            }
            var sum: Float = 0
            for i in 0..<(size * size) { sum += buffer[i * 4] }
            return sum / Float(size * size)
        }
        let exportValue = patchMean(exportOut.image, half: false)
        let previewValue = patchMean(previewOut.image, half: true)
        // Same input value (0.42 flat patch) through both pipelines must
        // land on the same output value: identical curves.
        #expect(abs(exportValue - previewValue) < 0.004,
                "flat 0.42 patch → export \(exportValue), preview \(previewValue)")
    }

    /// Issue #1 follow-up: the phase-split pipeline (materialized full-res
    /// prefix, then suffix) must produce the same result as the single-pass
    /// export pipeline — the materialization round-trip may only cost
    /// half-float quantization.
    @Test func prefixPhaseSplitMatchesSinglePass() async throws {
        let faded = makeFadedImage()
        var recipe = Recipe()
        recipe.negative.enabled = true
        recipe.restore.enabled = true

        let singlePass = try await ProcessingEngine.shared.run(
            recipe: recipe, on: PipelineImage(image: faded), mode: .export)

        let (fullRes, proxy) = try await ProcessingEngine.shared.materializedPrefix(
            recipe: recipe, on: PipelineImage(image: faded),
            retouchMask: nil, proxyMaxDimension: 4096)
        let split = try await ProcessingEngine.shared.run(
            recipe: recipe, on: fullRes, mode: .export, phase: .suffix)

        // 128px input under the 4096 cap: the proxy must be the same rendition.
        #expect(proxy.image.extent == fullRes.image.extent)

        let a = try RestoreStage.analyze(singlePass.image, ciContext: RenderService.context)
        let b = try RestoreStage.analyze(split.image, ciContext: RenderService.context)
        for (lhs, rhs) in [(a.samples.r, b.samples.r),
                           (a.samples.g, b.samples.g),
                           (a.samples.b, b.samples.b)] {
            #expect(abs(a.mean(of: lhs) - b.mean(of: rhs)) < 0.002)
        }
    }

    @Test func zeroStrengthIsIdentity() async throws {
        let faded = makeFadedImage()
        var settings = RestoreSettings()
        settings.enabled = true
        settings.strength = 0
        let stage = RestoreStage(settings: settings)

        let output = try await stage.apply(
            to: PipelineImage(image: faded),
            context: RenderContext(mode: .export))

        let before = try RestoreStage.analyze(faded, ciContext: RenderService.context)
        let after = try RestoreStage.analyze(output.image, ciContext: RenderService.context)
        #expect(abs(before.mean(of: before.samples.r) - after.mean(of: after.samples.r)) < 0.01)
        #expect(abs(before.mean(of: before.samples.b) - after.mean(of: after.samples.b)) < 0.01)
    }

    @Test func disabledRestoreLeavesRecipeInert() async throws {
        let faded = makeFadedImage()
        let recipe = Recipe()  // restore disabled by default
        let result = try await ProcessingEngine.shared.run(
            recipe: recipe,
            on: PipelineImage(image: faded),
            mode: .preview)
        #expect(result.image === faded)
    }
}
