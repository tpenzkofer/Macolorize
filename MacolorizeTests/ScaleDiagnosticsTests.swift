import Testing
import Foundation
import CoreImage
@testable import MacolorizeCore

/// Diagnostics on a real problem scan — inert unless MACOLORIZE_SAMPLE
/// names an image path. Documents the issue #1 mechanism: the dust
/// detector is not scale-stable, so even a slightly resampled proxy
/// (which softens single-pixel film grain) detects far less than full
/// resolution does. That is why heals must be computed once, at full
/// res, and shared by preview and export.
struct ScaleDiagnosticsTests {
    @Test func dustMaskCoverageByScale() async throws {
        guard let path = ProcessInfo.processInfo.environment["MACOLORIZE_SAMPLE"] else {
            return
        }
        let loaded = try ImageLoader.load(URL(fileURLWithPath: path))

        func coverage(_ image: CIImage, sensitivity: Double) -> Double {
            let mask = CleanStage.dustMask(for: image, sensitivity: sensitivity)
            let average = mask.applyingFilter("CIAreaAverage", parameters: [
                kCIInputExtentKey: CIVector(cgRect: mask.extent),
            ])
            var pixel = [Float](repeating: 0, count: 4)
            pixel.withUnsafeMutableBytes { raw in
                RenderService.context.render(average, toBitmap: raw.baseAddress!,
                                             rowBytes: 16,
                                             bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                                             format: .RGBAf,
                                             colorSpace: nil)
            }
            return Double(pixel[0])
        }

        let size = loaded.pixelSize
        let proxyExtent = loaded.proxy.extent
        print("sample \(Int(size.width))×\(Int(size.height)), " +
              "proxy \(Int(proxyExtent.width))×\(Int(proxyExtent.height))")

        let relative = CleanStage.relativeContrast(of: loaded.fullResolution)
        let grain = CleanStage.meanAndSigma(of: relative,
                                            ciContext: RenderService.context)
        let stats = try RestoreStage.analyze(relative, ciContext: RenderService.context)
        let percentiles = [50.0, 90, 99, 99.9].map {
            String(format: "P%.4g=%.3f", $0, stats.percentile($0, of: stats.samples.r))
        }
        print(String(format: "relative contrast: μ=%.3f σ=%.3f floor(μ+4σ)=%.3f  %@",
                     grain.mean, grain.sigma, grain.mean + 4 * grain.sigma,
                     percentiles.joined(separator: " ")))
        for s in [0.25, 0.5, 0.75, 1.0] {
            let full = coverage(loaded.fullResolution, sensitivity: s)
            let proxy = coverage(loaded.proxy, sensitivity: s)
            print(String(format: "sensitivity %.2f: mask covers %.2f%% at full res, %.2f%% on the proxy",
                         s, full * 100, proxy * 100))
        }

        // Red overlay of what sensitivity 0.5 heals, for visual inspection,
        // written next to the sample.
        let mask = CleanStage.dustMask(for: loaded.fullResolution, sensitivity: 0.5)
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0,
                                         colorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!)!)
            .cropped(to: loaded.fullResolution.extent)
        let overlay = red.applyingFilter("CIBlendWithMask", parameters: [
            kCIInputBackgroundImageKey: loaded.fullResolution,
            kCIInputMaskImageKey: mask,
        ])
        let preview = ImageRasterizer.proxy(of: overlay, maxDimension: 1600)
        let out = URL(fileURLWithPath: path).deletingLastPathComponent()
            .appendingPathComponent("mask_overlay.png")
        try RenderService.context.writePNGRepresentation(
            of: preview, to: out, format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        print("mask overlay written to \(out.path)")
    }
}
