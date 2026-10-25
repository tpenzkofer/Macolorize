import SwiftUI
import MetalKit
import MacolorizeCore

/// Zoom/pan state for the canvas. zoom == 1 means fit-to-view.
struct CanvasTransform: Equatable {
    var zoom: CGFloat = 1
    var pan: CGSize = .zero
}

/// MTKView that redraws on geometry changes. SwiftUI only calls
/// updateNSView on state changes, so without this a resize (window drag,
/// inspector animating in) leaves stale texture content on screen.
final class CanvasMTKView: MTKView {
    /// Two-finger trackpad pan (scroll events never reach SwiftUI gestures).
    var onScroll: ((CGSize) -> Void)?

    override func scrollWheel(with event: NSEvent) {
        guard let onScroll else {
            super.scrollWheel(with: event)
            return
        }
        onScroll(CGSize(width: event.scrollingDeltaX,
                        height: event.scrollingDeltaY))
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        needsDisplay = true
    }

    override func layout() {
        super.layout()
        needsDisplay = true
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        needsDisplay = true
    }
}

/// Metal-backed image canvas. 24 MP scans are never routed through SwiftUI's
/// Image — the CIImage is rendered straight into the drawable.
struct CanvasView: NSViewRepresentable {
    var image: CIImage?
    /// When set, shown left of the wipe divider (before/after compare).
    var comparisonImage: CIImage?
    /// Detection mask tinted red over the image when set.
    var maskOverlay: CIImage?
    /// Wipe position as a fraction of the view width.
    var wipe: CGFloat = 0.5
    var transform: CanvasTransform
    /// Two-finger trackpad scroll, for panning.
    var onScroll: ((CGSize) -> Void)?

    func makeCoordinator() -> CanvasRenderer {
        CanvasRenderer()
    }

    func makeNSView(context: Context) -> MTKView {
        let view = CanvasMTKView(frame: .zero, device: RenderService.device)
        view.delegate = context.coordinator
        view.framebufferOnly = false
        view.colorPixelFormat = .rgba16Float
        view.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        view.autoResizeDrawable = true
        // Keep drawable presentation synchronized with Core Animation layout;
        // without this, resizes show stale frames scaled into the new bounds.
        view.presentsWithTransaction = true
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.image = image
        context.coordinator.comparisonImage = comparisonImage
        context.coordinator.maskOverlay = maskOverlay
        context.coordinator.wipe = wipe
        context.coordinator.transform = transform
        (view as? CanvasMTKView)?.onScroll = onScroll
        view.needsDisplay = true
    }
}
