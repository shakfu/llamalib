# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)

# MODEL := models/gemma-2-9b-it-IQ4_XS.gguf
# MODEL := models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
MODEL := models/Llama-3.2-1B-Instruct-Q6_K.gguf

WITH_DYLIB=0

MIN_OSX_VER := -mmacosx-version-min=13.6

LIBLAMMA := ./lib/libllama.a


.PHONY: cmake clean reset setup setup_inplace wheel bind header

all: cmake

$(LIBLAMMA):
	@scripts/setup.sh

cmake: $(LIBLAMMA)
	@touch projects/cyllama/cyllama.pyx
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

build/include:
	@scripts/header_utils.py --force-overwrite --output_dir build/include include

bind: build/include
	@rm -rf build/bind
	@make -f scripts/bind/bind.mk bind


.PHONY: test test_simple test_main test_retrieve test_model \
		test_cy test_pb test_pb_hl test_nb bump clean reset

test:
	@pytest

test_simple:
	@g++ -std=c++14 -o build/simple \
		-I./include -L./lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		lib/libllama.a lib/libggml.a lib/libcommon.a \
		tests/simple.cpp
	@./build/simple -m $(MODEL) \
		-p "When did the French Revolution start?" -n 512

test_main:
	@g++ -std=c++14 -o build/main \
		-I./include -L./lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		lib/libllama.a lib/libggml.a lib/libcommon.a \
		tests/main.cpp
	@./build/main -m $(MODEL) --log-disable \
		-p "When did the French Revolution start?" -n 512

test_retrieve:
	@./bin/llama-retrieval --model models/all-MiniLM-L6-v2-Q5_K_S.gguf \
		--top-k 3 --context-file README.md \
		--context-file LICENSE \
		--chunk-size 100 \
		--chunk-separator .

$(MODEL):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-IQ4_XS.gguf

test_model: $(MODEL)
	@./bin/llama-simple -c 512 -n 512 -m $(MODEL) \
	-p "Number of planets in our solar system"

test_cy: 
	@cd tests && python3 cy_simple.py

test_pb:
	@cd tests && python3 pb_simple.py

test_pb_hl:
	@cd tests && python3 pb_simple_highlevel.py

test_nb:
	@cd tests && python3 nb_simple.py

test_llava:
	@./bin/llama-llava-cli -m models/llava-llama-3-8b-v1_1-int4.gguf \
		--mmproj models/llava-llama-3-8b-v1_1-mmproj-f16.gguf \
		--image tests/media/flower.jpg -c 4096 -e \
		-p "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe this image<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

bump:
	@scripts/bump.sh

clean:
	@rm -rf build dist *.egg-info

reset:
	@rm -rf build bin lib include


