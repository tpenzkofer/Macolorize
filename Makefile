XCODEPROJ := Macolorize.xcodeproj
DERIVED   := build
APP       := $(DERIVED)/Build/Products/Debug/Macolorize.app
CLI       := $(DERIVED)/Build/Products/Debug/MacolorizeCLI

.PHONY: generate build test run cli models clean

generate:
	xcodegen generate

build: generate
	xcodebuild -project $(XCODEPROJ) -scheme Macolorize -configuration Debug \
		-destination 'platform=macOS' -derivedDataPath $(DERIVED) build

test: generate
	xcodebuild -project $(XCODEPROJ) -scheme Macolorize -configuration Debug \
		-destination 'platform=macOS' -derivedDataPath $(DERIVED) test

run: build
	open $(APP)

models:
	cd tools && ./fetch_and_convert_all.sh

# Copy converted models into the sandboxed app's container so a dev build
# finds them without importing through the UI. Launch the app once first.
CONTAINER_MODELS := $(HOME)/Library/Containers/com.penzkofer.Macolorize/Data/Library/Application\ Support/Macolorize/Models
install-models:
	mkdir -p $(CONTAINER_MODELS)
	cp -R Models/*.mlpackage $(CONTAINER_MODELS)/

# Release artifacts: slim app + models archive (upload both to the GitHub release).
VERSION ?= dev
release: generate
	xcodebuild -project $(XCODEPROJ) -scheme Macolorize -configuration Release \
		-destination 'platform=macOS' -derivedDataPath $(DERIVED) build
	mkdir -p dist
	ditto -c -k --keepParent $(DERIVED)/Build/Products/Release/Macolorize.app dist/Macolorize-$(VERSION).zip
	ditto -c -k Models dist/Macolorize-Models.zip
	@ls -lh dist/

clean:
	rm -rf $(DERIVED) $(XCODEPROJ) dist
