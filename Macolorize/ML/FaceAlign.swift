import CoreImage
import Vision

/// A detected face and the similarity transform mapping image coordinates
/// into the model's 512×512 aligned-crop space.
public struct AlignedFace: Sendable {
    public let toCrop: CGAffineTransform
}

public enum FaceAlign {
    public static let cropSize: CGFloat = 512

    /// FFHQ 512 five-point template (left eye, right eye, nose, left mouth,
    /// right mouth) converted to CI's bottom-left origin (y flipped).
    static let template: [CGPoint] = [
        CGPoint(x: 192.98138, y: 512 - 239.94708),
        CGPoint(x: 318.90277, y: 512 - 240.19360),
        CGPoint(x: 256.63416, y: 512 - 314.01935),
        CGPoint(x: 201.26117, y: 512 - 371.41043),
        CGPoint(x: 313.08905, y: 512 - 371.15118),
    ]

    /// Detects faces and returns alignment transforms. Small/low-confidence
    /// faces are skipped (restoring a 30 px face invents detail from nothing).
    public static func detectFaces(in image: CIImage, minFaceSize: CGFloat = 96) throws -> [AlignedFace] {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        try handler.perform([request])
        guard let observations = request.results else { return [] }

        let imageSize = image.extent.size
        var faces: [AlignedFace] = []
        for observation in observations {
            guard let landmarks = observation.landmarks,
                  let leftPupil = landmarks.leftPupil?.pointsInImage(imageSize: imageSize).first,
                  let rightPupil = landmarks.rightPupil?.pointsInImage(imageSize: imageSize).first,
                  let noseRegion = landmarks.nose?.pointsInImage(imageSize: imageSize),
                  let lips = landmarks.outerLips?.pointsInImage(imageSize: imageSize),
                  !noseRegion.isEmpty, lips.count >= 2
            else { continue }

            let faceWidth = observation.boundingBox.width * imageSize.width
            guard faceWidth >= minFaceSize else { continue }

            // Order eyes and mouth corners by image x so Vision's left/right
            // naming convention can't flip the alignment.
            let eyes = [leftPupil, rightPupil].sorted { $0.x < $1.x }
            let nose = CGPoint(x: noseRegion.map(\.x).reduce(0, +) / CGFloat(noseRegion.count),
                               y: noseRegion.map(\.y).reduce(0, +) / CGFloat(noseRegion.count))
            let mouthLeft = lips.min { $0.x < $1.x }!
            let mouthRight = lips.max { $0.x < $1.x }!

            let source = [eyes[0], eyes[1], nose, mouthLeft, mouthRight]
            // The image is y-up; the template is y-up (pre-flipped) — but the
            // template's "left mouth" is below the eyes in y-down space, i.e.
            // template mouth y < eye y after flip. Source mouth is also below
            // eyes in CI coords (smaller y). Conventions match.
            guard let transform = similarityTransform(from: source, to: template) else { continue }
            faces.append(AlignedFace(toCrop: transform))
        }
        return faces
    }

    /// Least-squares 2D similarity (rotation + uniform scale + translation,
    /// no reflection) mapping `source` points onto `target`.
    static func similarityTransform(from source: [CGPoint], to target: [CGPoint]) -> CGAffineTransform? {
        guard source.count == target.count, source.count >= 2 else { return nil }
        let n = CGFloat(source.count)
        let meanS = CGPoint(x: source.map(\.x).reduce(0, +) / n,
                            y: source.map(\.y).reduce(0, +) / n)
        let meanT = CGPoint(x: target.map(\.x).reduce(0, +) / n,
                            y: target.map(\.y).reduce(0, +) / n)

        var a: CGFloat = 0, b: CGFloat = 0, norm: CGFloat = 0
        for (s, t) in zip(source, target) {
            let sx = s.x - meanS.x, sy = s.y - meanS.y
            let tx = t.x - meanT.x, ty = t.y - meanT.y
            a += sx * tx + sy * ty
            b += sx * ty - sy * tx
            norm += sx * sx + sy * sy
        }
        guard norm > 1e-6 else { return nil }
        let c = a / norm   // scale·cosθ
        let s = b / norm   // scale·sinθ
        let tx = meanT.x - (c * meanS.x - s * meanS.y)
        let ty = meanT.y - (s * meanS.x + c * meanS.y)
        return CGAffineTransform(a: c, b: s, c: -s, d: c, tx: tx, ty: ty)
    }
}
