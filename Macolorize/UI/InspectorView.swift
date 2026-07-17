import SwiftUI
import MacolorizeCore

struct InspectorView: View {
    @Bindable var session: EditSession

    @State private var presetStore = PresetStore()
    @State private var showSavePreset = false
    @State private var presetName = ""

    private let sharpenAvailable = ModelStore.isAvailable(.realesrganX2)
    private let colorizeAvailable = ModelStore.isAvailable(.ddcolor)
    private let retouchModelAvailable = ModelStore.isAvailable(.lama)
    private let facesAvailable = ModelStore.isAvailable(.gfpgan)
    private let denoiseAvailable = ModelStore.isAvailable(.drunet)

    var body: some View {
        Form {
            if let histogram = session.histogram {
                Section {
                    HistogramView(data: histogram)
                        .frame(height: 72)
                }
            }

            Section("Presets") {
                Menu {
                    ForEach(presetStore.presets) { preset in
                        Button(preset.name) {
                            session.recipe = preset.recipe
                        }
                    }
                    if !presetStore.presets.isEmpty {
                        Divider()
                        Menu("Delete") {
                            ForEach(presetStore.presets) { preset in
                                Button(preset.name, role: .destructive) {
                                    presetStore.delete(preset)
                                }
                            }
                        }
                        Divider()
                    }
                    Button("Save Current Settings…") {
                        presetName = ""
                        showSavePreset = true
                    }
                } label: {
                    Label("Apply Preset", systemImage: "slider.horizontal.2.square.on.square")
                }
            }

            Section {
                Toggle("Color Negative", isOn: $session.recipe.negative.enabled)
                    .font(.headline)
                if session.recipe.negative.enabled {
                    Text("Inverts the scan and removes the orange film mask. Combine with Restore Faded Color for tone and balance.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Adjustments")
            }

            Section {
                Toggle("Remove Dust & Scratches", isOn: $session.recipe.clean.enabled)
                    .font(.headline)
                    .disabled(!retouchModelAvailable)

                VStack(alignment: .leading) {
                    Text("Sensitivity \(Int(session.recipe.clean.sensitivity * 100))%")
                        .font(.caption)
                    Slider(value: $session.recipe.clean.sensitivity, in: 0...1)
                }
                .disabled(!session.recipe.clean.enabled || !retouchModelAvailable)

                if ModelStore.isAvailable(.scratchnet) {
                    Toggle("Detect Long Scratches (ML)", isOn: $session.recipe.clean.detectScratches)
                        .disabled(!session.recipe.clean.enabled || !retouchModelAvailable)
                }

                if !retouchModelAvailable {
                    Text("Model not installed — run `make models` in the project folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if session.recipe.clean.enabled {
                    Text("Small specks are healed automatically; use the Retouch brush for anything it misses.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Adjustments")
            }

            Section {
                Toggle("Restore Faded Color", isOn: $session.recipe.restore.enabled)
                    .font(.headline)

                Group {
                    Picker("White Balance", selection: $session.recipe.restore.whiteBalance) {
                        ForEach(WhiteBalanceMode.allCases) { mode in
                            Text(mode.label).tag(mode)
                        }
                    }

                    VStack(alignment: .leading) {
                        Text("Black Point \(session.recipe.restore.blackPercentile, format: .number.precision(.fractionLength(2)))%")
                            .font(.caption)
                        Slider(value: $session.recipe.restore.blackPercentile, in: 0...2)
                    }

                    VStack(alignment: .leading) {
                        Text("White Point \(session.recipe.restore.whitePercentile, format: .number.precision(.fractionLength(2)))%")
                            .font(.caption)
                        Slider(value: $session.recipe.restore.whitePercentile, in: 98...100)
                    }

                    VStack(alignment: .leading) {
                        Text("Strength \(Int(session.recipe.restore.strength * 100))%")
                            .font(.caption)
                        Slider(value: $session.recipe.restore.strength, in: 0...1)
                    }
                }
                .disabled(!session.recipe.restore.enabled)
            }

            Section {
                Toggle("Restore Faces", isOn: $session.recipe.faces.enabled)
                    .font(.headline)
                    .disabled(!facesAvailable)

                VStack(alignment: .leading) {
                    Text("Strength \(Int(session.recipe.faces.strength * 100))%")
                        .font(.caption)
                    Slider(value: $session.recipe.faces.strength, in: 0...1)
                }
                .disabled(!session.recipe.faces.enabled || !facesAvailable)

                if !facesAvailable {
                    Text("Model not installed — run `make models` in the project folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if session.recipe.faces.enabled {
                    Text("Sharpens detected faces (GFPGAN); the rest of the image is untouched.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section {
                Toggle("Colorize B&W", isOn: $session.recipe.colorize.enabled)
                    .font(.headline)
                    .disabled(!colorizeAvailable)

                Group {
                    if ModelStore.isAvailable(.ddcolorArtistic) {
                        Picker("Style", selection: $session.recipe.colorize.variant) {
                            ForEach(ColorizeVariant.allCases) { variant in
                                Text(variant.label).tag(variant)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                    VStack(alignment: .leading) {
                        Text("Saturation \(Int(session.recipe.colorize.saturation * 100))%")
                            .font(.caption)
                        Slider(value: $session.recipe.colorize.saturation, in: 0...2)
                    }
                    VStack(alignment: .leading) {
                        Text("Strength \(Int(session.recipe.colorize.strength * 100))%")
                            .font(.caption)
                        Slider(value: $session.recipe.colorize.strength, in: 0...1)
                    }
                }
                .disabled(!session.recipe.colorize.enabled || !colorizeAvailable)

                if !colorizeAvailable {
                    Text("Model not installed — run `make models` in the project folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section {
                Toggle("Reduce Film Grain", isOn: $session.recipe.denoise.enabled)
                    .font(.headline)
                    .disabled(!denoiseAvailable)

                VStack(alignment: .leading) {
                    Text("Strength \(Int(session.recipe.denoise.strength * 100))%")
                        .font(.caption)
                    Slider(value: $session.recipe.denoise.strength, in: 0...1)
                }
                .disabled(!session.recipe.denoise.enabled || !denoiseAvailable)

                if !denoiseAvailable {
                    Text("Model not installed — run `make models` in the project folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section {
                Toggle("Sharpen & Upscale", isOn: $session.recipe.sharpen.enabled)
                    .font(.headline)
                    .disabled(!sharpenAvailable)

                Picker("Model", selection: $session.recipe.sharpen.model) {
                    ForEach(SharpenModel.allCases) { model in
                        Text(model.label).tag(model)
                    }
                }
                .disabled(!session.recipe.sharpen.enabled || !sharpenAvailable)

                if !sharpenAvailable {
                    Text("Model not installed — run `make models` in the project folder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if session.recipe.sharpen.enabled {
                    Text("Preview shows the effect at screen size; export upscales fully.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .alert("Save Preset", isPresented: $showSavePreset) {
            TextField("Preset Name", text: $presetName)
            Button("Save") {
                try? presetStore.save(name: presetName, recipe: session.recipe)
            }
            .disabled(presetName.isEmpty)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Saves the current adjustments for reuse.")
        }
    }
}
