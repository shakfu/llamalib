# llamalib

Early stage experiments with different python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp). The purpose is to learn about the internals of a popular c++/c LLM inference engine and understand how it works and interacts with other related software such as [ggml](https://github.com/ggerganov/ggml).

Not yet sure how this project will evolve beyond that, but the aim for each of the variants to support the core featureset of `llama-cli` with respect to supported models.

Given that there is a fairly mature ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project, this all seems quite redundant.

Nonetheless, there may be some performance benefits from the use of compiled wrappers and some incidental benefits such as learning more about the underlying system. A future development idea may be to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.

## Status

Development on macOS to keep things simpler.

Still early days. The initial milestone is to enabling each wrapper variant a version of the `simple.cpp` test to be run.

- pybind11: first wrapper interation done. A test to replicate `simple.cpp` using the wrapper is revealing some remaining parts to be wrapped. Crashing bugs still in test. high-level simple prompt wrapper working (WIP)

- nanobind: follows pybind11 implementation as they are quite similar. First wrapper implemented with some gaps. Tests will follow pybind11 testing

- cython: `llama.pxd` is done. Work paused pending tests the pybind11 wrapper.


## Usage

```sh

git clone https://github.com/shakfu/llamalib.git
cd llamalib
make
```

This will:

1. Download and build `llama.cpp`
2. Install it into `bin`, `include`, and `lib` in the project folder
3. Build `cyllama` (`cython` wrapper)
4. Build `pbllama` (`pybind11` wrapper)
5. Build `nbllama` (`nanobind` wrapper)


## Testing

First step will get a [test model](rom https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/blob/main/gemma-2-9b-it-IQ4_XS.gguf ) from huggingface, in this case `gemma-2-9b-it-IQ4_XS.gguf` and run the `bin/llama-simple` api with the model and a basic prompt to ensure it works.

```sh
make test_model
```

If this works ok and you see an answer, then 


```sh
make test_pb_highlevel
```

which is equivalent to: `cd tests && python3 pb_highlevel.py`





