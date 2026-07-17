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
