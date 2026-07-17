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

clean:
	rm -rf $(DERIVED) $(XCODEPROJ)
