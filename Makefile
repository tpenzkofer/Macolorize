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

# Release artifacts: signed DMG + models archives. The base archive stays
# pinned at v1.1.0; upload only the DMG and Macolorize-Models-Update-1.3.zip
# (ModelDownloader.archives) to the release.
VERSION ?= dev
NOTARY_PROFILE ?= macolorize
RELEASE_APP := $(DERIVED)/Build/Products/Release/Macolorize.app
SIGN_ID := Developer ID Application

release: generate
	xcodebuild -project $(XCODEPROJ) -scheme Macolorize -configuration Release \
		-destination 'platform=macOS' -derivedDataPath $(DERIVED) build
	mkdir -p dist
	ditto -c -k Models dist/Macolorize-Models.zip
	ditto -c -k --keepParent Models/DRUNetMap.mlpackage dist/Macolorize-Models-Update-1.3.zip
	$(MAKE) dmg VERSION=$(VERSION)
	@ls -lh dist/

dmg:
	rm -rf dist/dmg-staging dist/Macolorize-$(VERSION).dmg
	mkdir -p dist/dmg-staging
	cp -R $(RELEASE_APP) dist/dmg-staging/
	ln -s /Applications dist/dmg-staging/Applications
	hdiutil create -volname Macolorize -srcfolder dist/dmg-staging -ov -format UDZO \
		dist/Macolorize-$(VERSION).dmg -quiet
	codesign --force --timestamp --sign "$(SIGN_ID)" dist/Macolorize-$(VERSION).dmg
	rm -rf dist/dmg-staging

# One-time setup: xcrun notarytool store-credentials macolorize \
#   --apple-id <apple-id> --team-id ZM6YGES9JH   (asks for an app-specific password)
notarize:
	xcrun notarytool submit dist/Macolorize-$(VERSION).dmg \
		--keychain-profile $(NOTARY_PROFILE) --wait
	xcrun stapler staple dist/Macolorize-$(VERSION).dmg
	xcrun stapler staple $(RELEASE_APP)
	spctl -a -t open --context context:primary-signature -vv dist/Macolorize-$(VERSION).dmg

clean:
	rm -rf $(DERIVED) $(XCODEPROJ) dist
