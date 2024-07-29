# set path so `faust` be queried for the path to stdlib
export PATH := $(PWD)/bin:$(PATH)

WITH_DYLIB=0

MIN_OSX_VER := -mmacosx-version-min=13.6

LLAMA_STATICLIB := ./lib/libfaust.a
INTERP_TESTS := tests/test_faust_interp

LIBLAMMA := lib/libllama.a

.PHONY: cmake clean setup setup_inplace wheel

all: cmake

$(LIBLAMMA):
	@scripts/setup.sh

cmake: $(LIBLAMMA)
	@mkdir -p build && cd build && cmake .. -DLLAMA_SHAREDLIB=$(WITH_DYLIB) && make

setup:
	@python3 setup.py build

setup_inplace:
	@python3 setup.py build_ext --inplace
	@rm -rf build

wheel:
	@echo "WITH_DYLIB=$(WITH_DYLIB)"
	@python3 setup.py bdist_wheel
ifeq ($(WITH_DYLIB),1)
	delocate-wheel -v dist/*.whl 
endif


test: test_cyllama test_pbllama test_nbllama prep_tests
	@echo "DONE"

test_cyllama: cmake prep_tests
	@python3 tests/test_cyllama.py

test_pbllama: cmake prep_tests
	@python3 tests/test_pbllama.py

test_nbllama: cmake prep_tests
	@python3 tests/test_nbllama.py

clean:
	@rm -rf build dist *.egg-info

