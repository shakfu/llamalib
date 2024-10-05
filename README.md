# llamalib - compiled python llama.cpp wrappers

The project includes three different experimental python wrappers of @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp) which is likely "at the frontier of open-source compiled LLM inference". The purpose is to learn about the internals of this popular c++/c LLM inference engine while wrapping it for use by python code. It tries to keep up with the latest changes in the `llama.cpp` main branch.

A project goal is that each of the python wrapper variants should programmatically support the core feature-set of `llama.cpp` with respect to supported `.gguf`-format models.

Given that there is a fairly mature and well-maintained ctypes based wrapper provided by @abetlen's [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) project and that llm inference is gpu-driven rather than cpu-driven, this all may see quite redundant. Irrespective, there are some benefits to developing alternative python wrappers to `llama.cpp`:

- There may be some incidental performance benefits to the use of compiled wrappers over the use of ctypes.

- It may be possible to incorporate external optimizations more readily into compiled wrappers, and

- It may be useful in case one wants to de-couple the python frontend and wrapper backends to existing frameworks: that is a future development idea may be to just replace the ctypes wrapper in `llama-cpp-python` with one of compiled python wrappers and contribute it back as a PR.

- This is the most efficient way, for me at least, to learn about the underlying technologies.


## Status

Development only on macOS to keep things simple. The following table provide an overview of the current wrapping/dev status, note that `hl` relates to `high-level` and `ll` relates to `low-level`:


| status                       | pbllama       | nbllama       | cyllama       |
| :--------------------------- | :-----------: | :-----------: | :-----------: |
| wrapper-type                 | pybind11 	   | nanobind 	   | cython 	   |
| wrap llama.h         		   | 1 			   | 1 			   | 1 			   |
| wrap hl simple-cli  		   | 1 			   | 1 			   | 1 			   |
| wrap ll simple-cli    	   | 1 			   | 1 			   | 1 			   |
| wrap ll llama-cli     	   | 0 			   | 0 			   | 0 			   |
  

The initial milestone for each wrapper type was to create a high-level wrapper of the `simple.cpp` llama.cpp example, following by a low-level one. The high-level wrapper c++ code is placed in `llamalib.h` single-header library, and wrapping is complete for all three frameworks. The final object is to fully wrap the functionality of `llama-cli` for all three wrapper-types.

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

As a first step type, you should download a `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). The following models have been known to work on a 16GB M1 Macbook air. A good start is [Llama-3.2-1B-Instruct-Q6_K.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q6_K.gguf).

After downloading it, place the model in the `llamalib/models` folder and run:

```sh
bin/llama-simple -c 512 -n 512 -m models/Llama-3.2-1B-Instruct-Q6_K.gguf \
	-p "Is mathematics discovered or invented?"
```

Now, you will need `pytest` installed to run tests:

```sh
pytest
```

If all tests pass, feel free to cd in the `tests` directory and run some examples directly

```sh
cd tests && python3 pb_simple_highlevel.py`
```

