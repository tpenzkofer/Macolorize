import CoreGraphics

/// One tile of a fixed-size grid over an image.
public struct Tile: Equatable, Sendable {
    /// Region sampled from the input (always tileSize², may extend past the
    /// image edge — pad by clamping when rendering).
    public let source: CGRect
    /// Region of `source` whose output is kept. Cores partition the image
    /// exactly; each core edge is ≥ overlap/2 away from its tile edge (except
    /// at image borders), which puts seams well inside the receptive field.
    public let core: CGRect
}

/// Pure tile geometry: positions, edge clamping, core partition.
public struct TileGrid: Sendable {
    public let tiles: [Tile]
    public let tileSize: Int
    public let extent: CGRect

    public init(extent: CGRect, tileSize: Int, overlap: Int) {
        precondition(overlap >= 0 && overlap < tileSize, "overlap must be < tileSize")
        self.tileSize = tileSize
        self.extent = extent

        let positionsX = Self.positions(length: Int(extent.width), tile: tileSize, overlap: overlap)
        let positionsY = Self.positions(length: Int(extent.height), tile: tileSize, overlap: overlap)

        var tiles: [Tile] = []
        for (yIndex, y) in positionsY.enumerated() {
            for (xIndex, x) in positionsX.enumerated() {
                let source = CGRect(x: extent.minX + CGFloat(x),
                                    y: extent.minY + CGFloat(y),
                                    width: CGFloat(tileSize),
                                    height: CGFloat(tileSize))
                let coreX = Self.coreRange(positions: positionsX, index: xIndex,
                                           tile: tileSize, length: Int(extent.width))
                let coreY = Self.coreRange(positions: positionsY, index: yIndex,
                                           tile: tileSize, length: Int(extent.height))
                let core = CGRect(x: extent.minX + CGFloat(coreX.lowerBound),
                                  y: extent.minY + CGFloat(coreY.lowerBound),
                                  width: CGFloat(coreX.count),
                                  height: CGFloat(coreY.count))
                tiles.append(Tile(source: source, core: core))
            }
        }
        self.tiles = tiles
    }

    /// Tile origins along one axis. Stride = tile − overlap; the last tile is
    /// shifted back so it ends exactly at the edge (when length ≥ tile).
    static func positions(length: Int, tile: Int, overlap: Int) -> [Int] {
        guard length > tile else { return [0] }
        let stride = tile - overlap
        var result: [Int] = []
        var position = 0
        while true {
            if position + tile >= length {
                result.append(length - tile)
                break
            }
            result.append(position)
            position += stride
        }
        return result
    }

    /// Non-overlapping core range for tile `index`: midpoints between
    /// neighboring tile starts/ends.
    static func coreRange(positions: [Int], index: Int, tile: Int, length: Int) -> Range<Int> {
        let start: Int
        let end: Int
        if index == 0 {
            start = 0
        } else {
            let previousEnd = positions[index - 1] + tile
            start = (positions[index] + previousEnd) / 2
        }
        if index == positions.count - 1 {
            end = min(length, positions[index] + tile)
        } else {
            let thisEnd = positions[index] + tile
            end = (positions[index + 1] + thisEnd) / 2
        }
        return start..<end
    }
}
