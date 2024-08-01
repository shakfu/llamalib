# llamalib

Early stage experiments with different python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp). The purpose is to learn about the internals of a popular c++/c LLM inference engine and understand how it works and interacts with other related software such as [ggml](https://github.com/ggerganov/ggml).

Not yet sure how this project will evolve beyond that, but the aim for each of the variants to support the core featureset of `llama-cli` with respect to supported models.

Given that there is a fairly mature ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project, this all seems quite redundant.

Nonetheless, there may be some performance benefits from the use of compiled wrappers. A future development idea may be to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.


## Status

Development on macOS to keep things simpler.

Still early days. The initial milestone will be enabling for  each wrapper variant a version of the `simple.cpp` test to be run.

- cython: `llama.pxd` is done.

- pybind11: first wrapper interation almost there.

- nanobind: follows pybind11 implementation as they are quite similar. First wrapper still not complete.




## Usage

```sh
make
```

This will:

1. This will download and build `llama.cpp`
2. Install it into `bin`, `include`, and `lib`
3. Build cyllama (cython wrapper)
4. Build pbllama (pybind11 wrapper)
5. Build nbllama (nanobind wrapper)

