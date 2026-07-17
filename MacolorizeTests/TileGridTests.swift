import Testing
import CoreGraphics
@testable import MacolorizeCore

struct TileGridTests {
    @Test func coresPartitionTheImageExactly() {
        let extent = CGRect(x: 0, y: 0, width: 3000, height: 2000)
        let grid = TileGrid(extent: extent, tileSize: 512, overlap: 128)

        // Total core area equals image area (exact partition, no gaps/overlap).
        let totalCoreArea = grid.tiles.reduce(0.0) { $0 + $1.core.width * $1.core.height }
        #expect(totalCoreArea == extent.width * extent.height)

        // Every core is contained in its source tile.
        for tile in grid.tiles {
            #expect(tile.source.contains(tile.core) || tile.source == tile.core)
        }

        // No two cores intersect.
        for i in 0..<grid.tiles.count {
            for j in (i + 1)..<grid.tiles.count {
                let intersection = grid.tiles[i].core.intersection(grid.tiles[j].core)
                #expect(intersection.isNull || intersection.width == 0 || intersection.height == 0)
            }
        }
    }

    @Test func coreEdgesStayInsideReceptiveField() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 3000, height: 2000),
                            tileSize: 512, overlap: 128)
        for tile in grid.tiles {
            // Interior core edges must sit ≥ overlap/2 from the tile border.
            if tile.core.minX > 0 {
                #expect(tile.core.minX - tile.source.minX >= 64)
            }
            if tile.core.maxX < 3000 {
                #expect(tile.source.maxX - tile.core.maxX >= 64)
            }
        }
    }

    @Test func allSourceTilesAreFixedSize() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 3000, height: 2000),
                            tileSize: 512, overlap: 128)
        for tile in grid.tiles {
            #expect(tile.source.width == 512)
            #expect(tile.source.height == 512)
        }
    }

    @Test func smallImageGetsSingleTile() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 300, height: 200),
                            tileSize: 512, overlap: 128)
        #expect(grid.tiles.count == 1)
        // Source extends past the edge (padded by clamping at render time);
        // core covers exactly the image.
        #expect(grid.tiles[0].core == CGRect(x: 0, y: 0, width: 300, height: 200))
    }

    @Test func exactTileSizeImageGetsSingleTile() {
        let grid = TileGrid(extent: CGRect(x: 0, y: 0, width: 512, height: 512),
                            tileSize: 512, overlap: 128)
        #expect(grid.tiles.count == 1)
        #expect(grid.tiles[0].source == CGRect(x: 0, y: 0, width: 512, height: 512))
        #expect(grid.tiles[0].core == grid.tiles[0].source)
    }

    @Test func nonZeroOriginExtentIsHandled() {
        let extent = CGRect(x: 100, y: 50, width: 1000, height: 800)
        let grid = TileGrid(extent: extent, tileSize: 512, overlap: 128)
        let union = grid.tiles.reduce(CGRect.null) { $0.union($1.core) }
        #expect(union == extent)
    }
}
