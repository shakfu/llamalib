# llamalib - compiled python llama.cpp wrappers

The project provides three different python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp) which is likely the most active open-source compiled LLM inference engine. The python wrapping frameworks used are [cython](https://github.com/cython/cython), [pybind11](https://github.com/pybind/pybind11), and [nanobind](https://github.com/wjakob/nanobind) and share the common feature of being compiled, and in this project statically linked, against `llama.cpp`.

Development goals are to:

- Stay up-to-date with bleeding-edge `llama.cpp`.

- Produce a minimal, performant, compiled, thin python wrapper around the core `llama-cli` feature-set of `llama.cpp`.

- Integrate and wrap `llava-cli` features.

- Integrate and wrap features from related projects such as [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

- Learn about the internals of this popular C++/C LLM inference engine along the way.

Given that there is a fairly mature, well-maintained and performant ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project and that llm inference is gpu-driven rather than cpu-driven, this all may see quite redundant. Nonetheless, we anticipate some benefits to using compiled wrappers:

- Packaging benefits with respect to self-contained statically compiled extension modules.

- There may be some performance improvements in the use of compiled wrappers over the use of ctypes.

- It may be possible to incorporate external optimizations more readily into compiled wrappers, and

- It provides a basis for integration with other code written in a wrapper variant.

- It may be useful in case one wants to de-couple the python frontend and wrapper backends to existing frameworks: for example, it may be useful to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.

- This is the most efficient way, for me at least, to learn about the underlying technologies.


## Status

Development only on macOS to keep things simple. The following table provide an overview of the current wrapping/dev status:


| status                       | pbllama       | nbllama       | cyllama       |
| :--------------------------- | :-----------: | :-----------: | :-----------: |
| wrapper-type                 | pybind11 	   | nanobind 	   | cython 	   |
| wrap llama.h         		   | 1 			   | 1 			   | 1 			   |
| wrap high-level simple-cli   | 1 			   | 1 			   | 1 			   |
| wrap low-level simple-cli    | 1 			   | 1 			   | 1 			   |
| wrap low-level llama-cli     | 0 			   | 0 			   | 0 			   |
  

The initial milestone for each wrapper type was to create a high-level wrapper of the `simple.cpp` llama.cpp example, following by a low-level one. The high-level wrapper c++ code is placed in `llamalib.h` single-header library, and wrapping is complete for all three frameworks. The final object is to fully wrap the functionality of `llama-cli` for all three wrapper-types.

Nonetheless, not all wrapping efforts proceed at an equal pace: in general, the cython wrapper will likely be the most advanced of the 3.

It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!


## Setup

To build `llamalib`:

1. A recent version of `python3` (testing on python 3.12)

2. `cmake`, which can be installed on MacOS using [homebrew]() with `brew install cmake`

3. The following python wrapping libraries, if you don't already have them. All python dependencies can be installed via `pip install -r requirements.txt` (feel free to use `virtualenv` if you like):

	- [cython](https://cython.org)
	- [pybind11](https://github.com/pybind/pybind11)
	- [nanobind](https://github.com/wjakob/nanobind)

With the above dependencies installed, download and build the `llamalib` system, just type the following:

```sh
git clone https://github.com/shakfu/llamalib.git
cd llamalib
make
```

This will:

1. Download and build `llama.cpp`
2. Install it into `bin`, `include`, and `lib` in the cloned `llamalib` folder
3. Build `cyllama` (`cython` wrapper)
4. Build `pbllama` (`pybind11` wrapper)
5. Build `nbllama` (`nanobind` wrapper)


## Testing

As a first step, you should download a smallish llm in the `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). This [document](https://github.com/shakfu/llamalib/blob/main/docs/model-performance.md) provides some examples of models which have been known to work on a 16GB M1 Macbook air.

A good model to start with is [Llama-3.2-1B-Instruct-Q6_K.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q6_K.gguf). After downloading it, place the model in the `llamalib/models` folder and run:

```sh
bin/llama-simple -c 512 -n 512 -m models/Llama-3.2-1B-Instruct-Q6_K.gguf \
	-p "Is mathematics discovered or invented?"
```

Now, you will need `pytest` installed to run tests:

```sh
pytest
```

If all tests pass, feel free to `cd` into the `tests` directory and run some examples directly, for example:


```sh
cd tests && python3 cy_simple.py`
```

## TODO

- [x] wrap llama-simple

- [ ] wrap llama-cli

- [ ] wrap llama-llava-cli

- [ ] wrap [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

- [ ] wrap [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)

