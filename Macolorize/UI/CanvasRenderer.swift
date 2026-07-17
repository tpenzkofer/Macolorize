import AppKit
import MetalKit
import CoreImage
import MacolorizeCore

/// Draws the current preview CIImage into the MTKView drawable via
/// CIRenderDestination. Draw calls only happen on the main thread because the
/// view is configured with isPaused + enableSetNeedsDisplay.
@MainActor
final class CanvasRenderer: NSObject, MTKViewDelegate {
    private let commandQueue: MTLCommandQueue = RenderService.device.makeCommandQueue()!

    var image: CIImage?
    var comparisonImage: CIImage?
    var maskOverlay: CIImage?
    var wipe: CGFloat = 0.5
    var transform = CanvasTransform()

    private static let backgroundColor = CIColor(red: 0.11, green: 0.11, blue: 0.12)

    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
            render(in: view)
        }
    }

    private static let debugLogging =
        ProcessInfo.processInfo.environment["MACOLORIZE_DEBUG_RENDER"] != nil

    private func render(in view: MTKView) {
        // Self-heal drawable/layout mismatches: some layout paths (inspector
        // animation, split-view resize) change bounds without a redraw or
        // before autoResizeDrawable catches up.
        let backingScale = view.window?.backingScaleFactor ?? 2
        let expectedSize = CGSize(width: view.bounds.width * backingScale,
                                  height: view.bounds.height * backingScale)
        if expectedSize.width > 0, expectedSize.height > 0,
           abs(view.drawableSize.width - expectedSize.width) > 1
            || abs(view.drawableSize.height - expectedSize.height) > 1 {
            view.drawableSize = expectedSize
        }

        let drawableSize = view.drawableSize
        guard drawableSize.width > 0, drawableSize.height > 0,
              let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer()
        else { return }

        if Self.debugLogging {
            let line = "render drawableSize=\(drawableSize) bounds=\(view.bounds.size) " +
                "texture=\(drawable.texture.width)x\(drawable.texture.height)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }

        let bounds = CGRect(origin: .zero, size: drawableSize)
        var output = CIImage(color: Self.backgroundColor).cropped(to: bounds)

        if let image {
            let screenScale = drawableSize.width / max(view.bounds.width, 1)
            let placement = fitTransform(for: image.extent,
                                         in: drawableSize,
                                         screenScale: screenScale)
            var placed = image.transformed(by: placement)

            // Detection-mask review: tint masked pixels red.
            if let maskOverlay {
                let maskScale = CGAffineTransform(
                    scaleX: image.extent.width / max(maskOverlay.extent.width, 1),
                    y: image.extent.height / max(maskOverlay.extent.height, 1))
                let placedMask = maskOverlay
                    .transformed(by: maskScale.concatenating(placement))
                let red = CIImage(color: CIColor(red: 1, green: 0.1, blue: 0.1))
                    .cropped(to: placed.extent)
                placed = red.applyingFilter("CIBlendWithMask", parameters: [
                    kCIInputBackgroundImageKey: placed,
                    kCIInputMaskImageKey: placedMask,
                ]).cropped(to: placed.extent)
            }

            // Before/after wipe: comparison (original) left of the divider.
            if let comparisonImage {
                let placedOriginal = comparisonImage.transformed(by: placement)
                let splitX = drawableSize.width * wipe
                let rightMask = CIImage(color: .white)
                    .cropped(to: CGRect(x: splitX, y: 0,
                                        width: max(drawableSize.width - splitX, 0),
                                        height: drawableSize.height))
                placed = placed.applyingFilter("CIBlendWithMask", parameters: [
                    kCIInputBackgroundImageKey: placedOriginal,
                    kCIInputMaskImageKey: rightMask,
                ])
            }
            output = placed.composited(over: output)
        }

        let destination = CIRenderDestination(width: Int(drawableSize.width),
                                              height: Int(drawableSize.height),
                                              pixelFormat: view.colorPixelFormat,
                                              commandBuffer: commandBuffer) {
            drawable.texture
        }
        destination.colorSpace = view.colorspace ?? CGColorSpace(name: CGColorSpace.sRGB)!

        do {
            try RenderService.context.startTask(toRender: output, to: destination)
            if view.presentsWithTransaction {
                // Synchronous present inside the current CA transaction so the
                // frame always matches the layer's current bounds.
                commandBuffer.commit()
                commandBuffer.waitUntilScheduled()
                drawable.present()
            } else {
                commandBuffer.present(drawable)
                commandBuffer.commit()
            }
        } catch {
            // A failed frame is dropped; the next needsDisplay retries.
        }
    }

    /// Fit-to-view centered, then user zoom around the center, then pan.
    /// pan is in view points; CI's y axis points up, hence the sign flip.
    private func fitTransform(for extent: CGRect,
                              in drawableSize: CGSize,
                              screenScale: CGFloat) -> CGAffineTransform {
        guard extent.width > 0, extent.height > 0 else { return .identity }
        let fit = min(drawableSize.width / extent.width, drawableSize.height / extent.height)
        let scale = fit * transform.zoom
        let tx = (drawableSize.width - extent.width * scale) / 2
            - extent.minX * scale
            + transform.pan.width * screenScale
        let ty = (drawableSize.height - extent.height * scale) / 2
            - extent.minY * scale
            - transform.pan.height * screenScale
        return CGAffineTransform(a: scale, b: 0, c: 0, d: scale, tx: tx, ty: ty)
    }
}
