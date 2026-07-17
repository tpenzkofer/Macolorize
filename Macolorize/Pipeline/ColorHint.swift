import CoreGraphics
import CoreImage
import Foundation

/// A painted color hint: stroke geometry (normalized, like BrushStroke) plus
/// the sRGB color the user wants that region to be. During colorization the
/// hint's a/b chroma replaces the model's prediction inside the stroke,
/// feathered so it spreads naturally.
public struct ColorHint: Codable, Sendable, Equatable {
    public var stroke: BrushStroke
    public var red: Double
    public var green: Double
    public var blue: Double

    public init(stroke: BrushStroke, red: Double, green: Double, blue: Double) {
        self.stroke = stroke
        self.red = red
        self.green = green
        self.blue = blue
    }

    /// The hint's chroma in Lab units (skimage convention, sRGB D65).
    public var ab: (a: Double, b: Double) {
        Self.srgbToLab(red: red, green: green, blue: blue).ab
    }

    static func srgbToLab(red: Double, green: Double, blue: Double)
        -> (l: Double, ab: (a: Double, b: Double)) {
        func linear(_ c: Double) -> Double {
            c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        let r = linear(red), g = linear(green), b = linear(blue)
        // sRGB (D65) → XYZ
        let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        // Reference white D65
        let xn = x / 0.95047, yn = y / 1.0, zn = z / 1.08883
        func f(_ t: Double) -> Double {
            t > 216.0 / 24389.0 ? cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0
        }
        let fx = f(xn), fy = f(yn), fz = f(zn)
        return (l: 116 * fy - 16, ab: (a: 500 * (fx - fy), b: 200 * (fy - fz)))
    }
}

public enum ColorHintRenderer {
    /// Blends each hint's constant a/b into the predicted ab image inside a
    /// feathered stroke mask. `ab` and the result carry a in R, b in G.
    public static func apply(hints: [ColorHint],
                             to predictedAB: CIImage,
                             imageExtent: CGRect) -> CIImage {
        guard !hints.isEmpty else { return predictedAB }
        let featherRadius = max(imageExtent.width, imageExtent.height) * 0.01

        var result = predictedAB
        for hint in hints {
            guard let mask = RetouchMask.rasterize(
                strokes: [hint.stroke],
                size: CGSize(width: predictedAB.extent.width,
                             height: predictedAB.extent.height)) else { continue }

            let chroma = hint.ab
            // Encode ab in [0,1] then expand with an unclamped matrix. The
            // color is declared in linear space so color management leaves
            // the encoded values untouched (default CIColor is sRGB and
            // would be gamma-converted on the way into the working space).
            let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
            let encoded = CIImage(color: CIColor(red: chroma.a / 256.0 + 0.5,
                                                 green: chroma.b / 256.0 + 0.5,
                                                 blue: 0,
                                                 colorSpace: linear) ?? .black)
                .cropped(to: predictedAB.extent)
            let constantAB = encoded.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 256, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 256, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 1, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: -128, y: -128, z: 0, w: 0),
            ])

            let feathered = mask
                .transformed(by: .init(
                    scaleX: predictedAB.extent.width / mask.extent.width,
                    y: predictedAB.extent.height / mask.extent.height))
                .applyingFilter("CIGaussianBlur", parameters: [
                    kCIInputRadiusKey: featherRadius * predictedAB.extent.width / imageExtent.width,
                ])
                .cropped(to: predictedAB.extent)

            result = constantAB.applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: result,
                kCIInputMaskImageKey: feathered,
            ]).cropped(to: predictedAB.extent)
        }
        return result
    }
}
