# llamalib

Early stage experiments with different python wrappers of [llama.cpp](https://github.com/ggerganov/llama.cpp)

Starting with cython wrapper first.. other variants may not be built (but infrastructre is in place in any case.)

Early development will be on macOS for convenience.

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

