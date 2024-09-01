# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)
MODEL := models/gemma-2-9b-it-IQ4_XS.gguf

WITH_DYLIB=0

MIN_OSX_VER := -mmacosx-version-min=13.6

LIBLAMMA := ./lib/libllama.a

.PHONY: cmake clean reset setup setup_inplace wheel bind

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

bind:
	@rm -rf bind
	@make -f scripts/bind/bind.mk bind


.PHONY: test test_simple test_cy test_pb test_nb prep_tests bench_cy bench_nb bench_pb bump

test:
	@pytest

test_simple:
	@g++ -std=c++14 -o build/simple \
		-I./include -L./lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		lib/libllama.a lib/libggml.a lib/libcommon.a \
		tests/simple.cpp
	@./build/simple -m $(MODEL) -p "Who invented algebra" -n 512

test_cy:
	@cd tests && python3 cy_simple.py

test_pb:
	@cd tests && python3 pb_simple.py

test_nb:
	@cd tests && python3 nb_simple.py

bench_pb:
	@cd tests && hyperfine 'python3 pb_simple.py'

bench_nb:
	@cd tests && hyperfine 'python3 nb_simple.py'

bench_cy:
	@cd tests && hyperfine 'python3 cy_simple.py'

bump:
	@scripts/bump.sh

clean:
	@rm -rf build dist *.egg-info

reset:
	@rm -rf build bin lib include

