import Foundation
import CoreImage
import MacolorizeCore

// Headless pipeline runner.
//
// usage: macolorize-cli [options] <input-file-or-folder> [output-file]
//   --recipe <file.json>   full recipe (see --emit-recipe)
//   --restore              enable fade restoration with defaults
//   --format <tiff16|png|jpeg>
//   --emit-recipe [path]   write a template recipe JSON and exit
//
// A folder input processes every image inside it (results in <folder>/Macolorized).

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(code)
}

var arguments = Array(CommandLine.arguments.dropFirst())
var recipe = Recipe()
var format = ExportFormat.tiff16

@MainActor
func takeOptionValue(_ name: String) -> String? {
    guard let index = arguments.firstIndex(of: name) else { return nil }
    guard index + 1 < arguments.count else { fail("\(name) needs a value", code: 64) }
    let value = arguments[index + 1]
    arguments.removeSubrange(index...index + 1)
    return value
}

if let emitIndex = arguments.firstIndex(of: "--emit-recipe") {
    arguments.remove(at: emitIndex)
    var template = Recipe()
    template.restore.enabled = true
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try! encoder.encode(template)
    if let path = arguments.first {
        try data.write(to: URL(fileURLWithPath: path))
        print(path)
    } else {
        print(String(data: data, encoding: .utf8)!)
    }
    exit(0)
}

if let recipePath = takeOptionValue("--recipe") {
    do {
        let data = try Data(contentsOf: URL(fileURLWithPath: recipePath))
        recipe = try JSONDecoder().decode(Recipe.self, from: data)
    } catch {
        fail("could not read recipe \(recipePath): \(error.localizedDescription)", code: 64)
    }
}

if let flagIndex = arguments.firstIndex(of: "--restore") {
    recipe.restore.enabled = true
    arguments.remove(at: flagIndex)
}

var retouchMask: CIImage?
if let maskPath = takeOptionValue("--mask") {
    guard let mask = CIImage(contentsOf: URL(fileURLWithPath: maskPath)) else {
        fail("could not read mask \(maskPath)", code: 64)
    }
    retouchMask = mask
}

if let formatName = takeOptionValue("--format") {
    guard let parsed = ExportFormat(rawValue: formatName) else {
        fail("unknown format \(formatName) (tiff16|png|jpeg)", code: 64)
    }
    format = parsed
}

guard let inputPath = arguments.first else {
    FileHandle.standardError.write(Data("""
    usage: macolorize-cli [--recipe file.json] [--restore] [--format tiff16|png|jpeg] \
    <input-file-or-folder> [output-file]
           macolorize-cli --emit-recipe [path]

    """.utf8))
    exit(64)
}
let inputURL = URL(fileURLWithPath: inputPath)

var isDirectory: ObjCBool = false
guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory) else {
    fail("no such file: \(inputURL.path)", code: 66)
}

if isDirectory.boolValue {
    let urls = ImageLoader.imageFiles(in: inputURL)
    guard !urls.isEmpty else { fail("no images in \(inputURL.path)", code: 66) }
    var failures = 0
    for await event in BatchRunner.run(urls: urls, recipe: recipe, format: format) {
        switch event {
        case .started(let url):
            print("processing \(url.lastPathComponent)…")
        case .finished(_, let output):
            print("  → \(output.path)")
        case .failed(let source, let message):
            failures += 1
            FileHandle.standardError.write(
                Data("  ✗ \(source.lastPathComponent): \(message)\n".utf8))
        }
    }
    exit(failures == 0 ? 0 : 1)
} else {
    do {
        let loaded = try ImageLoader.load(inputURL)
        let outputURL = arguments.count >= 2
            ? URL(fileURLWithPath: arguments[1])
            : ExportNaming.destination(for: inputURL, format: format)
        let result = try await ProcessingEngine.shared.run(
            recipe: recipe,
            on: PipelineImage(image: loaded.fullResolution),
            mode: .export,
            retouchMask: retouchMask)
        try ImageExporter.export(result.image, from: inputURL, to: outputURL, format: format)
        print(outputURL.path)
    } catch {
        fail(error.localizedDescription)
    }
}
