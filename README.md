# llamalib

Early stage experiments with different python wrappers of [llama.cpp](https://github.com/ggerganov/llama.cpp)

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

