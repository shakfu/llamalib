# llamalib

Early stage experiments with different python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp). The purpose is to learn about the internals of a popular c++/c LLM inference engine and understand how it works and interacts with other related software such as [ggml](https://github.com/ggerganov/ggml).

Not yet sure how this project will evolve beyond that, but the aim for each of the variants to support the core featureset of `llama-cli` with respect to supported models.

Given that there is a fairly mature ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project, this all seems quite redundant.

Nonetheless, there may be some performance benefits from the use of compiled wrappers and more importantly, valuable incidental benefits such as learning more about the underlying system which is being wrapped. A future development idea may be to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.


## Status

Development only on macOS to keep things simple.

Still early days. The initial milestone is to enabling each wrapper variant a version of the `simple.cpp` test to be run.

- pybind11: first wrapper interation done. A test to replicate `simple.cpp` with the pybind11 wrapper crashes during the while loop, so low-level wrapper is a work-in-progress. High-level simple prompt wrapper working.

- nanobind: follows pybind11 implementation as they are quite similar. First wrapper implemented with some gaps. Tests will follow pybind11 testing

- cython: `llama.pxd` is done. Some work done along the lines of `llama-cpp-python` but mostly paused pending completion of the low-levl pybind11 wrapper.

It goes without saying that any help / collaboration / contributions would be welcome!


## Setup

`llamalib` requires:

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

As a first step type:

```sh
make test_model
```

This downloads a [test model](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF) from huggingface and places in `llamalib/models`, in this case `gemma-2-9b-it-IQ4_XS.gguf` and runs the `bin/llama-simple` cli with the model and a basic prompt to ensure it works.

If this works ok and you see a reasonable answer, then test high-level pybind11 wrapper:


```sh
make test_pb_highlevel
```

which is equivalent to: `cd tests && python3 pb_highlevel.py`


To run the low-level pybind11 wrapper:

```sh
make test_pb
```


