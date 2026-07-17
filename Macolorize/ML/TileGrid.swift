import CoreGraphics

/// Feathering geometry for one tile along both axes. Fades are capped to the
/// zone where exactly two tiles overlap; `leftZero`/`bottomZero` blank the
/// tile where a shifted edge tile lands under THREE tiles (the earlier pair's
/// ramps already sum to 1 there).
public struct TileFade: Equatable, Sendable {
    public var leftZero: CGFloat = 0
    public var left: CGFloat = 0
    public var right: CGFloat = 0
    public var bottomZero: CGFloat = 0
    public var bottom: CGFloat = 0
    public var top: CGFloat = 0
}

/// One tile of a fixed-size grid over an image.
public struct Tile: Equatable, Sendable {
    /// Region sampled from the input (always tileSize², may extend past the
    /// image edge — pad by clamping when rendering).
    public let source: CGRect
    /// Region of `source` whose output is kept when stitching by hard
    /// partition (mask merging). Cores partition the image exactly.
    public let core: CGRect
    /// Feathered-stitching geometry; neighbor ramps are complementary so
    /// total coverage is exactly 1 everywhere.
    public let fade: TileFade
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
                let fx = Self.axisFade(positions: positionsX, index: xIndex, tile: tileSize)
                let fy = Self.axisFade(positions: positionsY, index: yIndex, tile: tileSize)
                let fade = TileFade(leftZero: fx.zero, left: fx.low, right: fx.high,
                                    bottomZero: fy.zero, bottom: fy.low, top: fy.high)
                tiles.append(Tile(source: source, core: core, fade: fade))
            }
        }
        self.tiles = tiles
    }

    /// Feather geometry along one axis for tile `index`:
    /// - `zero`: width at the low edge where this tile must contribute
    ///   nothing (covered by the two previous tiles' complementary ramps)
    /// - `low`: 0→1 ramp width after the zero zone (pairs with the previous
    ///   tile's `high` ramp over the exact same band)
    /// - `high`: 1→0 ramp width at the high edge (pairs with the next tile)
    static func axisFade(positions: [Int], index: Int, tile: Int)
        -> (zero: CGFloat, low: CGFloat, high: CGFloat) {
        let p = positions
        var zero = 0, low = 0, high = 0
        if index >= 1 {
            let overlap = max(p[index - 1] + tile - p[index], 0)
            let tripleEnd = index >= 2 ? max(0, p[index - 2] + tile - p[index]) : 0
            zero = min(tripleEnd, overlap)
            low = overlap - zero
        }
        if index + 1 < p.count {
            let bandStart = index >= 1 ? max(p[index + 1], p[index - 1] + tile) : p[index + 1]
            high = max(p[index] + tile - bandStart, 0)
        }
        return (CGFloat(zero), CGFloat(low), CGFloat(high))
    }

    /// Float weight bitmap (tileSize², row 0 = top) for feathered stitching:
    /// zeros in triple-overlap zones, then linear ramps across the pair
    /// bands, complementary between neighbors — coverage sums to exactly 1.
    public static func weightMap(for tile: Tile, tileSize: Int) -> [Float] {
        func axis(_ count: Int, zeroLow: CGFloat, fadeLow: CGFloat, fadeHigh: CGFloat) -> [Float] {
            var values = [Float](repeating: 1, count: count)
            let zero = Int(zeroLow.rounded())
            let lo = Int(fadeLow.rounded())
            let hi = Int(fadeHigh.rounded())
            for i in 0..<min(zero, count) { values[i] = 0 }
            if lo > 0 {
                for i in 0..<min(lo, max(count - zero, 0)) {
                    values[zero + i] = (Float(i) + 0.5) / Float(lo)
                }
            }
            if hi > 0 {
                for i in 0..<min(hi, count) {
                    values[count - 1 - i] = min(values[count - 1 - i], (Float(i) + 0.5) / Float(hi))
                }
            }
            return values
        }
        let rx = axis(tileSize, zeroLow: tile.fade.leftZero,
                      fadeLow: tile.fade.left, fadeHigh: tile.fade.right)
        // Row 0 of the bitmap is the TOP of the tile (max y in CI coords);
        // the zero+low ramp belongs to the bottom (previous-neighbor) side.
        let ryBottomUp = axis(tileSize, zeroLow: tile.fade.bottomZero,
                              fadeLow: tile.fade.bottom, fadeHigh: tile.fade.top)
        var weights = [Float](repeating: 0, count: tileSize * tileSize)
        for row in 0..<tileSize {
            let wy = ryBottomUp[tileSize - 1 - row]
            for x in 0..<tileSize {
                weights[row * tileSize + x] = wy * rx[x]
            }
        }
        return weights
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
