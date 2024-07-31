# llamalib

Early stage experiments with different python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp). The purpose is to learn about the internals a c++/c LLM inference engine and understand how they work and interact with other software.

Given that there is a fairly mature ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project. One promising idea is to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute back to that project if successful.

Starting with cython wrapper first.. other variants are added intermittently. Early development will be on macOS to keep things simple.


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

