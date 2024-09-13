# llamalib

The project includes experiments with different python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp). The purpose is to learn about the internals of a popular c++/c LLM inference engine. It tries to keep up with the latest changes in `llama.cpp`.

The aim for each of the variants to programmatically support the core feature-set of `llama-cli` with respect to supported models.

Given that there is a fairly mature ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project, this all seems quite redundant.

Nonetheless, this is the most efficient way to learn about the underlying technologies and there may be some performance benefits the use of compiled wrappers as a bonus. A future development idea may be to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.


## Status

Development only on macOS to keep things simple. The following table provide an overview of the current development status:


| milestone               | pbllama       | nbllama       | cyllama       |
| :---------------------- | :-----------: | :-----------: | :-----------: |
| wrapper-type            | pybind11 	  | nanobind 	  | cython 	      |
| llama.h                 | 1 			  | 1 			  | 1 			  |
| high-level simple-cli   | 1 			  | 0 			  | 0 			  |
| low-level simple-cli    | 1 			  | 0 			  | 0 			  |
| low-level llama-cli     | 0 			  | 0 			  | 0 			  |
  

The initial milestone for each wrapper type is to create a high-level wrapper of the `simple.cpp` llama.cpp example, following by a low-level one. The final aim is to wrap the functionality of `llama-cli`.

It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!


## Setup

To builc `llamalib`:

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
make test_pb_hl
```

which is equivalent to: `cd tests && python3 pb_simple_highlevel.py`


To run the low-level pybind11 wrapper:

```sh
make test_pb
```


