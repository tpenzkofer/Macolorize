import CoreImage
import CoreML
import CoreVideo

public enum TiledModelRunnerError: Error, LocalizedError {
    case pixelBufferAllocation
    case unexpectedOutput(String)

    public var errorDescription: String? {
        switch self {
        case .pixelBufferAllocation: "Could not allocate tile buffers."
        case .unexpectedOutput(let name): "Model returned no image for \(name)."
        }
    }
}

/// Runs a fixed-input-size image-to-image Core ML model over a large image by
/// tiling. Tile outputs are cropped to their non-overlapping core region and
/// stitched; overlap keeps every kept pixel ≥ overlap/2 inside its tile, so
/// seams sit within the model's receptive field and vanish.
public struct TiledModelRunner: Sendable {
    public let tileSize: Int
    public let overlap: Int
    public let outputScale: Int
    public let inputName: String
    public let outputName: String
    /// Tiles per MLBatchProvider prediction (bounds peak memory).
    public let batchSize: Int

    public init(tileSize: Int = 512,
                overlap: Int = 128,
                outputScale: Int,
                inputName: String = "image",
                outputName: String = "upscaled",
                batchSize: Int = 8) {
        self.tileSize = tileSize
        self.overlap = overlap
        self.outputScale = outputScale
        self.inputName = inputName
        self.outputName = outputName
        self.batchSize = batchSize
    }

    public func run(model: SendableModel,
                    over image: CIImage,
                    ciContext: CIContext,
                    progress: (@Sendable (Double) -> Void)? = nil) async throws -> CIImage {
        let grid = TileGrid(extent: image.extent, tileSize: tileSize, overlap: overlap)
        let scale = CGFloat(outputScale)

        // sRGB-encoded pixels: SR models are trained on gamma-encoded images.
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        // Sample past the image edge by clamping, for border tiles.
        let clamped = image.clampedToExtent()

        var composited: [CIImage] = []
        var processed = 0

        for chunk in grid.tiles.chunked(into: batchSize) {
            try Task.checkCancellation()

            var inputs: [MLDictionaryFeatureProvider] = []
            for tile in chunk {
                let buffer = try Self.makePixelBuffer(size: tileSize)
                let tileImage = clamped
                    .cropped(to: tile.source)
                    .transformed(by: .init(translationX: -tile.source.minX,
                                           y: -tile.source.minY))
                ciContext.render(tileImage, to: buffer, bounds: CGRect(x: 0, y: 0,
                                 width: CGFloat(tileSize), height: CGFloat(tileSize)),
                                 colorSpace: srgb)
                inputs.append(try MLDictionaryFeatureProvider(
                    dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)]))
            }

            let batch = MLArrayBatchProvider(array: inputs)
            let results = try model.mlModel.predictions(from: batch,
                                                        options: MLPredictionOptions())

            for index in 0..<results.count {
                let tile = chunk[index]
                guard let outBuffer = results.features(at: index)
                    .featureValue(for: outputName)?.imageBufferValue else {
                    throw TiledModelRunnerError.unexpectedOutput(outputName)
                }
                // Tile output lives at tile.source * scale; keep only the core.
                let outImage = CIImage(cvPixelBuffer: outBuffer,
                                       options: [.colorSpace: srgb])
                    .transformed(by: .init(translationX: tile.source.minX * scale,
                                           y: tile.source.minY * scale))
                let scaledCore = CGRect(x: tile.core.minX * scale,
                                        y: tile.core.minY * scale,
                                        width: tile.core.width * scale,
                                        height: tile.core.height * scale)
                composited.append(outImage.cropped(to: scaledCore))
            }

            processed += chunk.count
            progress?(Double(processed) / Double(grid.tiles.count))
        }

        let outputExtent = CGRect(x: image.extent.minX * scale,
                                  y: image.extent.minY * scale,
                                  width: image.extent.width * scale,
                                  height: image.extent.height * scale)
        var stitched = CIImage(color: .black).cropped(to: outputExtent)
        for piece in composited {
            stitched = piece.composited(over: stitched)
        }
        return stitched
    }

    static func makePixelBuffer(size: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                         kCVPixelFormatType_32BGRA,
                                         attributes as CFDictionary, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw TiledModelRunnerError.pixelBufferAllocation
        }
        return buffer
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
