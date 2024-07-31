NAME := "llama"
# CPPFLAGS += "-Wno-c++11-extensions"
STDVER := "-std=c++11"
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)
EXTENSION_SUFFIX := $(shell python3-config --extension-suffix)
PY_INCLUDES := $(shell python3-config --includes)
EXTENSION := $(NAME)$(EXTENSION_SUFFIX)
BIND_DIR := scripts/bind

MACOS_SDK=$(shell xcrun --show-sdk-path)
SYS_CPP_INCLUDE=$(MACOS_SDK)/usr/include/c++/v1
SYS_C_INCLUDE=$(MACOS_SDK)/usr/include
PWD=$(shell pwd)

ALL_INCLUDES=$(BIND_DIR)/all_includes.hpp
CONFIG_FILE=$(BIND_DIR)/config.txt
NAMESPACE_TO_BIND=""
NAMESPACE_TO_SKIP=""

export CPLUS_INCLUDE_PATH := $(SYS_CPP_INCLUDE)
export C_INCLUDE_PATH := $(SYS_C_INCLUDE)
export PATH := $(PWD)/bin:$(PATH)


.PHONY: all build bind test clean regen

all: build

bind:
	@mkdir -p bind
	@binder \
		--root-module $(NAME) \
		--prefix $(PWD)/bind \
		--config=$(CONFIG_FILE) \
		--bind $(NAMESPACE_TO_BIND) \
		--suppress-errors \
		$(ALL_INCLUDES) \
		-- $(STDVER) \
		-I$(PWD)/include \
		-DNDEBUG


build:
	@mkdir -p build && cd build && cmake .. && cmake --build . --config Release

test:
	@pytest

clean:
	@rm -rf .pytest_cache
	@rm -rf __pycache__
	@rm -rf tests/__pycache__
	@rm -rf outputs
	@rm -rf bind build


regen: clean bind

