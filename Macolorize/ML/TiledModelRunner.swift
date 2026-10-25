import CoreImage
import CoreML
import CoreVideo

public enum TiledModelRunnerError: Error, LocalizedError {
    case pixelBufferAllocation
    case unexpectedOutput(String)
    case kernelUnavailable

    public var errorDescription: String? {
        switch self {
        case .pixelBufferAllocation: "Could not allocate tile buffers."
        case .unexpectedOutput(let name): "Model returned no image for \(name)."
        case .kernelUnavailable: "Tile blending kernel failed to compile."
        }
    }
}

/// Feathered tile accumulation: out = acc + tile.rgb · weight. Neighbor tile
/// weights are complementary linear ramps, so the accumulated coverage is
/// exactly 1 everywhere and seams vanish even in flat regions.
enum TileBlendKernel {
    private static let source = """
    #include <CoreImage/CoreImage.h>
    using namespace metal;

    [[stitchable]] float4 weightMultiply(coreimage::sample_t tile,
                                         coreimage::sample_t weight) {
        return float4(tile.rgb * weight.r, 0.0);
    }
    """

    static let kernel: CIColorKernel? = {
        (try? CIKernel.kernels(withMetalString: source))?
            .compactMap { $0 as? CIColorKernel }.first
    }()
}

/// Runs a fixed-input-size image-to-image Core ML model over a large image by
/// tiling, stitching tile outputs with feathered cross-fades.
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

    /// - Parameter featureBuilder: builds the model inputs for one tile
    ///   buffer and its source rect in image coordinates; defaults to
    ///   `[inputName: buffer]`. Use to add extra model inputs (e.g. the
    ///   denoiser's noise map).
    public func run(model: SendableModel,
                    over image: CIImage,
                    ciContext: CIContext,
                    featureBuilder: (@Sendable (CVPixelBuffer, CGRect) throws -> MLFeatureProvider)? = nil,
                    progress: (@Sendable (Double) -> Void)? = nil) async throws -> CIImage {
        guard let kernel = TileBlendKernel.kernel else {
            throw TiledModelRunnerError.kernelUnavailable
        }
        let grid = TileGrid(extent: image.extent, tileSize: tileSize, overlap: overlap)
        let scale = CGFloat(outputScale)

        // sRGB-encoded pixels: the models are trained on gamma-encoded images.
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        // Sample past the image edge by clamping, for border tiles.
        let clamped = image.clampedToExtent()

        _ = kernel
        var pieces: [(tile: Tile, output: CIImage)] = []
        var processed = 0

        for chunk in grid.tiles.chunked(into: batchSize) {
            try Task.checkCancellation()

            var inputs: [MLFeatureProvider] = []
            for tile in chunk {
                let buffer = try Self.makePixelBuffer(size: tileSize)
                let tileImage = clamped
                    .cropped(to: tile.source)
                    .transformed(by: .init(translationX: -tile.source.minX,
                                           y: -tile.source.minY))
                ciContext.render(tileImage, to: buffer, bounds: CGRect(x: 0, y: 0,
                                 width: CGFloat(tileSize), height: CGFloat(tileSize)),
                                 colorSpace: srgb)
                if let featureBuilder {
                    inputs.append(try featureBuilder(buffer, tile.source))
                } else {
                    inputs.append(try MLDictionaryFeatureProvider(
                        dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)]))
                }
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
                pieces.append((tile, CIImage(cvPixelBuffer: outBuffer,
                                             options: [.colorSpace: srgb])))
            }

            processed += chunk.count
            progress?(Double(processed) / Double(grid.tiles.count))
        }
        return try Self.stitch(pieces: pieces,
                               inputExtent: image.extent,
                               tileSize: tileSize,
                               outputScale: outputScale)
    }

    /// Feathered stitching of per-tile outputs (each in its own 0-origin
    /// coordinates) into the scaled output extent.
    static func stitch(pieces: [(tile: Tile, output: CIImage)],
                       inputExtent: CGRect,
                       tileSize: Int,
                       outputScale: Int) throws -> CIImage {
        guard let kernel = TileBlendKernel.kernel else {
            throw TiledModelRunnerError.kernelUnavailable
        }
        let scale = CGFloat(outputScale)
        let outputExtent = CGRect(x: inputExtent.minX * scale,
                                  y: inputExtent.minY * scale,
                                  width: inputExtent.width * scale,
                                  height: inputExtent.height * scale)
        var accumulated = CIImage(color: .black).cropped(to: outputExtent)

        for (tile, output) in pieces {
            let tileOutRect = CGRect(x: tile.source.minX * scale,
                                     y: tile.source.minY * scale,
                                     width: tile.source.width * scale,
                                     height: tile.source.height * scale)
            let placed = output.transformed(by: .init(translationX: tileOutRect.minX,
                                                      y: tileOutRect.minY))
            // clampedToExtent: bilinear sampling at the outermost pixels
            // would otherwise mix in transparent-outside and dim the image
            // border. Safe because the kernel only evaluates inside the rect.
            let weightImage = Self.weightImage(for: tile, tileSize: tileSize)
                .clampedToExtent()
                .transformed(by: CGAffineTransform(scaleX: scale, y: scale)
                    .concatenating(.init(translationX: tileOutRect.minX,
                                         y: tileOutRect.minY)))

            // The multiply kernel is evaluated ONLY over the tile's own rect,
            // so it can never sample its inputs out of extent (chained blend
            // kernels get fused by Core Image and then read past crop rects —
            // that bug showed up as edge-smeared, mostly black output).
            // Weighted pieces then combine by plain additive compositing.
            let weightedRegion = tileOutRect.intersection(outputExtent)
            guard let weighted = kernel.apply(
                extent: weightedRegion,
                arguments: [placed, weightImage]) else {
                throw TiledModelRunnerError.kernelUnavailable
            }
            accumulated = weighted.applyingFilter("CIAdditionCompositing", parameters: [
                kCIInputBackgroundImageKey: accumulated,
            ]).cropped(to: outputExtent)
        }
        return accumulated
    }

    /// Single-channel float CIImage of the tile's feathered weights.
    static func weightImage(for tile: Tile, tileSize: Int) -> CIImage {
        let weights = TileGrid.weightMap(for: tile, tileSize: tileSize)
        let data = weights.withUnsafeBufferPointer { Data(buffer: $0) }
        return CIImage(bitmapData: data,
                       bytesPerRow: tileSize * MemoryLayout<Float>.size,
                       size: CGSize(width: tileSize, height: tileSize),
                       format: .Rf,
                       colorSpace: nil)
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
