# Output directory from your preset
BUILD_DIR := build

IS_WINDOWS := $(findstring Windows_NT,$(OS))

# Platform-safe delete command
ifeq ($(IS_WINDOWS),Windows_NT)
  RM = rm -rf # Powershell
  BUILD_DIR_WIN := $(subst /,\\,$(BUILD_DIR))  # convert slashes for Windows
else
  RM = rm -rf
  BUILD_DIR_WIN := $(BUILD_DIR)
endif

.PHONY: all configure build rebuild clean help check-preset

all: help

check-preset:
ifndef preset
	$(error 'preset' is not set. Please specify a preset with preset=<preset-name>. Run 'make help' for more information)
endif

help:
	@echo "Usage: make [target] PRESET=<preset-name> [target=<target-name>]"
	@echo ""
	@echo "Available targets:"
	@echo "  configure  - Configure the project (CMake configure step)"
	@echo "  build      - Build the project (CMake build step). Use target=install to install"
	@echo "  rebuild    - Clean, configure, and build"
	@echo "  clean      - Remove build directory"
	@echo ""
	@echo "Required variables:"
	@echo "  preset     - CMake preset to use (REQUIRED)"
	@echo "               Example: make build preset=ninja-release"
	@echo ""
	@echo "Optional variables:"
	@echo "  target     - Specific target to build (optional)"
	@echo "               Example: make build preset=ninja-release target=image_visualizer"

configure: check-preset
	cmake --preset $(preset)

build: check-preset
ifdef target
	cmake --build --preset $(preset) --target $(target)
else
	cmake --build --preset $(preset)
endif

rebuild: check-preset
	cmake --fresh --preset $(preset)
ifdef target
	cmake --build --preset $(preset) --target $(target)
else
	cmake --build --preset $(preset)
endif

clean:
	$(RM) $(BUILD_DIR_WIN)