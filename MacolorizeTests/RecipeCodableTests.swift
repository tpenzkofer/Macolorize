import Testing
import Foundation
@testable import MacolorizeCore

struct RecipeCodableTests {
    @Test func roundTripPreservesAllSettings() throws {
        var recipe = Recipe()
        recipe.restore.enabled = true
        recipe.restore.blackPercentile = 0.42
        recipe.restore.whitePercentile = 99.5
        recipe.restore.whiteBalance = .whitePatch
        recipe.restore.strength = 0.7

        let data = try JSONEncoder().encode(recipe)
        let decoded = try JSONDecoder().decode(Recipe.self, from: data)
        #expect(decoded == recipe)
    }

    @Test func defaultRecipeIsInert() {
        let recipe = Recipe()
        #expect(!recipe.restore.enabled)
    }
}
