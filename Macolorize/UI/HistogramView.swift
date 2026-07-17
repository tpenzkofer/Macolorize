import SwiftUI
import MacolorizeCore

/// Compact RGB histogram of the current preview, additive-blended like the
/// usual photo-app style.
struct HistogramView: View {
    let data: HistogramData

    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size).insetBy(dx: 0, dy: 0)),
                         with: .color(.black.opacity(0.85)))
            context.blendMode = .plusLighter
            draw(channel: data.red, color: .red, in: &context, size: size)
            draw(channel: data.green, color: .green, in: &context, size: size)
            draw(channel: data.blue, color: Color(red: 0.35, green: 0.45, blue: 1),
                 in: &context, size: size)
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func draw(channel: [Float], color: Color,
                      in context: inout GraphicsContext, size: CGSize) {
        guard channel.count > 1 else { return }
        var path = Path()
        let step = size.width / CGFloat(channel.count - 1)
        path.move(to: CGPoint(x: 0, y: size.height))
        for (index, value) in channel.enumerated() {
            path.addLine(to: CGPoint(x: CGFloat(index) * step,
                                     y: size.height * (1 - CGFloat(value) * 0.95)))
        }
        path.addLine(to: CGPoint(x: size.width, y: size.height))
        path.closeSubpath()
        context.fill(path, with: .color(color.opacity(0.55)))
    }
}
