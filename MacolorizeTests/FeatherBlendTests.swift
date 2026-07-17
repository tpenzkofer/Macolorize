import Testing
import CoreGraphics
import CoreImage
import Foundation
@testable import MacolorizeCore

/// The feathered tile weights must sum to exactly 1 at every pixel —
/// otherwise stitched outputs show faint brightness lines at tile seams.
struct FeatherBlendTests {
    private func coverage(extent: CGRect, tileSize: Int, overlap: Int) -> [Float] {
        let grid = TileGrid(extent: extent, tileSize: tileSize, overlap: overlap)
        let width = Int(extent.width), height = Int(extent.height)
        var sum = [Float](repeating: 0, count: width * height)
        for tile in grid.tiles {
            let weights = TileGrid.weightMap(for: tile, tileSize: tileSize)
            let ox = Int(tile.source.minX - extent.minX)
            let oy = Int(tile.source.minY - extent.minY)
            for ty in 0..<tileSize {
                // Weight row 0 is the tile's TOP; CI y is up. Image row index
                // (top-down) for tile row ty:
                let imageYTopDown = (height - (oy + tileSize)) + ty
                guard imageYTopDown >= 0, imageYTopDown < height else { continue }
                for tx in 0..<tileSize {
                    let x = ox + tx
                    guard x >= 0, x < width else { continue }
                    sum[imageYTopDown * width + x] += weights[ty * tileSize + tx]
                }
            }
        }
        return sum
    }

    @Test func weightsSumToOneEverywhere() {
        let sums = coverage(extent: CGRect(x: 0, y: 0, width: 1500, height: 1000),
                            tileSize: 512, overlap: 128)
        let minSum = sums.min() ?? 0
        let maxSum = sums.max() ?? 0
        #expect(abs(minSum - 1) < 0.001, "min coverage \(minSum)")
        #expect(abs(maxSum - 1) < 0.001, "max coverage \(maxSum)")
    }

    @Test func weightsSumToOneWithShiftedEdgeTiles() {
        // Dimensions chosen so the last tile shifts back (irregular overlap).
        let sums = coverage(extent: CGRect(x: 0, y: 0, width: 1000, height: 700),
                            tileSize: 512, overlap: 96)
        #expect(abs((sums.min() ?? 0) - 1) < 0.001)
        #expect(abs((sums.max() ?? 0) - 1) < 0.001)
    }

    @Test func weightImageRoundTripsThroughCoreImage() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 1500, height: 1000),
                            tileSize: 512, overlap: 128)
        // Interior tile with fades on all sides.
        let tile = grid.tiles.first {
            $0.fade.left > 0 && $0.fade.right > 0 && $0.fade.bottom > 0 && $0.fade.top > 0
        }!
        let weights = TileGrid.weightMap(for: tile, tileSize: 512)
        let image = TiledModelRunner.weightImage(for: tile, tileSize: 512)

        func sample(_ x: Int, _ yCI: Int) -> Float {
            var pixel = [Float](repeating: 0, count: 4)
            pixel.withUnsafeMutableBytes { raw in
                RenderService.context.render(image,
                                             toBitmap: raw.baseAddress!,
                                             rowBytes: 16,
                                             bounds: CGRect(x: CGFloat(x), y: CGFloat(yCI),
                                                            width: 1, height: 1),
                                             format: .RGBAf,
                                             colorSpace: nil)
            }
            return pixel[0]
        }
        // Bitmap row r ↔ CI y = 511 - r.
        for (x, row) in [(256, 256), (10, 256), (500, 10), (64, 460)] {
            let expected = weights[row * 512 + x]
            let got = sample(x, 511 - row)
            #expect(abs(got - expected) < 0.02,
                    "at x=\(x) row=\(row): got \(got), expected \(expected)")
        }
    }

    /// Identity check: stitching an image's own tiles must reproduce the
    /// image exactly (this is what broke as visible seams / black output).
    @Test func stitchingIdentityTilesReproducesInput() throws {
        let width = 700, height = 600
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                pixels[i] = UInt8((x * 255) / width)
                pixels[i + 1] = UInt8((y * 255) / height)
                pixels[i + 2] = 128
                pixels[i + 3] = 255
            }
        }
        let input = CIImage(bitmapData: Data(pixels), bytesPerRow: width * 4,
                            size: CGSize(width: width, height: height),
                            format: .RGBA8,
                            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))

        let tileSize = 512
        let grid = TileGrid(extent: input.extent, tileSize: tileSize, overlap: 128)
        // "Identity model": each tile's output is its own input crop.
        let pieces = grid.tiles.map { tile in
            (tile: tile,
             output: input.clampedToExtent()
                .cropped(to: tile.source)
                .transformed(by: .init(translationX: -tile.source.minX,
                                       y: -tile.source.minY)))
        }
        let stitched = try TiledModelRunner.stitch(pieces: pieces,
                                                   inputExtent: input.extent,
                                                   tileSize: tileSize,
                                                   outputScale: 1)

        var out = [UInt8](repeating: 0, count: width * height * 4)
        out.withUnsafeMutableBytes { raw in
            RenderService.context.render(stitched,
                                         toBitmap: raw.baseAddress!,
                                         rowBytes: width * 4,
                                         bounds: input.extent,
                                         format: .RGBA8,
                                         colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        }
        var maxDiff = 0
        var worst = (x: 0, y: 0)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    let i = (y * width + x) * 4 + c
                    let d = abs(Int(out[i]) - Int(pixels[i]))
                    if d > maxDiff { maxDiff = d; worst = (x, y) }
                }
            }
        }
        // Diagnostic on failure: error by region (fade bands are x 188..512, y 88..512).
        if maxDiff > 2 {
            var probes: [String] = []
            for (px, py) in [(0, 0), (50, 40), (650, 560), (350, 300), (100, 300)] {
                let i = (py * width + px) * 4
                probes.append("(\(px),\(py)): got [\(out[i]),\(out[i+1]),\(out[i+2])] want [\(pixels[i]),\(pixels[i+1]),\(pixels[i+2])]")
            }
            try? probes.joined(separator: "\n").write(
                toFile: "/tmp/identity_probes.txt", atomically: true, encoding: .utf8)
            func regionMax(_ xr: Range<Int>, _ yr: Range<Int>) -> Int {
                var m = 0
                for y in yr { for x in xr { for c in 0..<3 {
                    m = max(m, abs(Int(out[(y*width+x)*4+c]) - Int(pixels[(y*width+x)*4+c])))
                } } }
                return m
            }
            print("worst at \(worst); interior(60..180,600..699? no) regions:")
            print("  no-fade region (0..180, 0..80): \(regionMax(0..<180, 0..<80))")
            print("  x-fade only (200..500, 0..80): \(regionMax(200..<500, 0..<80))")
            print("  y-fade only (0..180, 100..500): \(regionMax(0..<180, 100..<500))")
            print("  xy-fade (200..500, 100..500): \(regionMax(200..<500, 100..<500))")
            print("  far corner (600..699, 520..599): \(regionMax(600..<700, 520..<600))")
        }
        #expect(maxDiff <= 2, "max channel difference \(maxDiff) at \(worst)")
    }


    @Test func weightImageOrientationAsymmetricTile() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 1500, height: 1000),
                            tileSize: 512, overlap: 128)
        // Edge tile shifted back: strongly asymmetric x fades.
        let tile = grid.tiles.first { $0.fade.left > 200 && $0.fade.right == 0 }!
        let weights = TileGrid.weightMap(for: tile, tileSize: 512)
        let image = TiledModelRunner.weightImage(for: tile, tileSize: 512)
        func sample(_ x: Int, _ yCI: Int) -> Float {
            var pixel = [Float](repeating: 0, count: 4)
            pixel.withUnsafeMutableBytes { raw in
                RenderService.context.render(image, toBitmap: raw.baseAddress!,
                    rowBytes: 16, bounds: CGRect(x: CGFloat(x), y: CGFloat(yCI), width: 1, height: 1),
                    format: .RGBAf, colorSpace: nil)
            }
            return pixel[0]
        }
        print("fade:", tile.fade)
        for x in [5, 60, 150, 290, 400, 505] {
            print("x=\(x): bitmap=\(weights[256 * 512 + x]) ci=\(sample(x, 255))")
        }
        #expect(abs(sample(5, 255) - weights[256 * 512 + 5]) < 0.02)
        #expect(abs(sample(505, 255) - weights[256 * 512 + 505]) < 0.02)
    }


    @Test func stitchedCoverageProfile() throws {
        let width = 700, height = 600
        let extent = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
        let grid = TileGrid(extent: extent, tileSize: 512, overlap: 128)
        let white = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 512, height: 512))
        let pieces = grid.tiles.map { (tile: $0, output: white) }
        let stitched = try TiledModelRunner.stitch(pieces: pieces, inputExtent: extent,
                                                   tileSize: 512, outputScale: 1)
        var out = [Float](repeating: 0, count: width * height * 4)
        out.withUnsafeMutableBytes { raw in
            RenderService.context.render(stitched, toBitmap: raw.baseAddress!,
                rowBytes: width * 16, bounds: extent, format: .RGBAf, colorSpace: nil)
        }
        // Profile along x at bitmap row 40 (CI y=560, single-y zone) and row 300.
        var lines: [String] = []
        for row in [40, 300] {
            let profile = stride(from: 0, to: width, by: 25).map {
                String(format: "%.2f", out[(row * width + $0) * 4])
            }.joined(separator: " ")
            lines.append("row \(row): \(profile)")
        }
        try lines.joined(separator: "\n").write(
            toFile: "/tmp/coverage_profile.txt", atomically: true, encoding: .utf8)
        let minV = out.indices.filter { $0 % 4 == 0 }.map { out[$0] }.min() ?? 0
        #expect(minV > 0.99, "min effective coverage \(minV) — profile in /tmp/coverage_profile.txt")
    }

    @Test func singleTileHasUniformWeightOne() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 300, height: 200),
                            tileSize: 512, overlap: 128)
        let weights = TileGrid.weightMap(for: grid.tiles[0], tileSize: 512)
        #expect(weights.allSatisfy { $0 == 1 })
    }
}
