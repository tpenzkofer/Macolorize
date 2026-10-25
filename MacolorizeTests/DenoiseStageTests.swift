import Testing
import CoreImage
@testable import MacolorizeCore

/// The grain profile drives DRUNet's per-pixel noise map: it must see grain
/// that grows with brightness, or the shadows get over-smoothed again
/// (issue #1 follow-up).
struct DenoiseStageTests {
    /// Horizontal gradient (encoded 0.1…0.7) with seeded Gaussian grain
    /// whose std grows with brightness: 4 + 40·v on the 0–255 scale.
    private func makeGrainyGradient() -> CIImage {
        let size = 768
        var generator = SplitMix64(seed: 42)
        var pixels = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let v = 0.1 + 0.6 * Double(x) / Double(size - 1)
                let sigma = (4 + 40 * v) / 255
                let value = min(max(v + sigma * generator.nextGaussian(), 0), 1)
                let byte = UInt8((value * 255).rounded())
                let i = (y * size + x) * 4
                pixels[i] = byte; pixels[i + 1] = byte; pixels[i + 2] = byte
            }
        }
        return CIImage(bitmapData: Data(pixels), bytesPerRow: size * 4,
                       size: CGSize(width: size, height: size), format: .RGBA8,
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    @Test func grainProfileFollowsBrightness() {
        let profile = GrainProfile.measure(makeGrainyGradient(),
                                           ciContext: RenderService.context)
        func knot(at brightness: Double) -> Double {
            profile.knots[Int((brightness * Double(GrainProfile.knotCount - 1)).rounded())]
        }
        let dark = knot(at: 0.2), bright = knot(at: 0.6)
        // True grain: 12 at 0.2, 28 at 0.6 (0–255 scale).
        #expect(dark > 12 * 0.75 && dark < 12 * 1.25, "dark sigma \(dark)")
        #expect(bright > 28 * 0.75 && bright < 28 * 1.25, "bright sigma \(bright)")
        #expect(bright / dark > 1.8, "grain must grow with brightness")
    }

    @Test func noiseMapScalesWithStrength() throws {
        let image = makeGrainyGradient()
        let profile = GrainProfile.measure(image, ciContext: RenderService.context)
        let map = profile.noiseMap(for: image, factor: 2, maxSigma: 150)
        var pixel = [Float](repeating: 0, count: 4)
        RenderService.context.render(map, toBitmap: &pixel, rowBytes: 16,
                                     bounds: CGRect(x: 700, y: 384, width: 1, height: 1),
                                     format: .RGBAf, colorSpace: nil)
        let sigma = Double(pixel[0]) * 255
        // Bright end (~0.65 encoded): ~30 measured × 2.
        #expect(sigma > 45 && sigma < 80, "map sigma \(sigma)")
    }
}

/// Deterministic RNG so the grain fixture is identical on every run.
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func nextUniform() -> Double {
        Double(next() >> 11) / Double(1 << 53)
    }

    mutating func nextGaussian() -> Double {
        let u1 = max(nextUniform(), .leastNonzeroMagnitude)
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * nextUniform())
    }
}
