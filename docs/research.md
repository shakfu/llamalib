# Research


## Llama

- [Meta's Llama page](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)

## Tested GGUF models

- [gemma-2-9b-it-IQ4_XS.gguf](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF)
- [mistral-7b-instruct-v0.1.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- [qwen2-1_5b-instruct-q5_k_m.gguf](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF)
- [codellama-7b.Q4_K_M.gguf](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF)

pending (due to memory errors)

- LongWriter-llama3.1-8b-Q3_K_L.gguf
- LongWriter-llama3.1-8b-Q5_K_S.gguf
- codellama-7b.Q4_K_M.gguf



## llama.cpp

- [llama.cpp wikipedia](https://en.wikipedia.org/wiki/Llama.cpp)

- [llama-cpp-python docs](https://llama-cpp-python.readthedocs.io/en/latest/)

- [Understanding how LLM inference works with llama.cpp](https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/)

- [datacamp llama.cpp tutorial](https://www.datacamp.com/tutorial/llama-cpp-tutorial)

- [Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)

- [langchain-llama-cpp](https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/)

- [lama.cpp: The Ultimate Guide to Efficient LLM Inference and Applications](https://pyimagesearch.com/2024/08/26/llama-cpp-the-ultimate-guide-to-efficient-llm-inference-and-applications/)

- [Llama.cpp Tutorial: A Complete Guide to Efficient LLM Inference and Implementation](https://www.datacamp.com/tutorial/llama-cpp-tutorial)

- [Calling Llama.cpp from langchain](https://stackoverflow.com/questions/77753658/langchain-local-llama-compatible-model)

## Related Projects

- [llama.py](https://github.com/daskol/llama.py)

- [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp)

- [llamaindex](https://docs.llamaindex.ai) and [llama-cpp on llamaindex](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp/)

## Allocating Memory for 2d Matrices in c/c++

- [Three ways to allocate memory for a 2-dimensional matrix in C++](https://secure.eld.leidenuniv.nl/~moene/Home/tips/matrix2d/)

- [Contiguous allocation of 2-D arrays in c++](https://dev.to/drakargx/c-contiguous-allocation-of-2-d-arrays-446m)

- [How to dynamically allocate a contiguous block of memory for a 2D array](https://stackoverflow.com/questions/13534966/how-to-dynamically-allocate-a-contiguous-block-of-memory-for-a-2d-array)

- [Freaky way of allocating two-dimensional array?](https://stackoverflow.com/questions/36794202/freaky-way-of-allocating-two-dimensional-array)

- [How to dynamically allocate a 2D array in C?](https://www.geeksforgeeks.org/dynamically-allocate-2d-array-c/)

- [2D dynamic array in continuous memory locations (C)](https://gsamaras.wordpress.com/code/2d-dynamic-array-in-continuous-memory-locations-c/)

also [2d-dynamic-array-c](https://gsamaras.wordpress.com/code/2d-dynamic-array-c/)

```c++
int** allocate2D(int** A, const int N, const int M) {
    int i;
    int *t0;
 
    A = malloc(M * sizeof (int*)); /* Allocating pointers */
    t0 = malloc(N * M * sizeof (int)); /* Allocating data */
    for (i = 0; i < M; i++)
        A[i] = t0 + i * (N);
 
    return A;
}
 
void free2Darray(int** p) {
    free(p[0]);
    free(p);
}
```

- [cprogramming.com: Dynamically allocate memory to create 2D array](https://cboard.cprogramming.com/c-programming/143055-dynamically-allocate-memory-create-2d-array.html#post1067593)

- [The proper way to dynamically create a two-dimensional array in C](https://www.dimlucas.com/index.php/2017/02/18/the-proper-way-to-dynamically-create-a-two-dimensional-array-in-c/)

- [Arrays in C: dynamically allocated 2D arrays](https://www.cs.swarthmore.edu/~newhall/unixhelp/C_arrays.html)

- [How to understand numpy strides for layman?](https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman)


## pybind11

- [returning numpy arrays via pybind11](https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11/44682603#44682603)

- [Pybind11 and std::vector -- How to free data using capsules?](https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules)

- [How to send a PyCapsule from C++ to python using pybind11](https://stackoverflow.com/questions/61560207/how-to-send-a-pycapsule-from-c-to-python-using-pybind11)

- [How to bind c structure with an array of another structure as a member in pybind11?](https://stackoverflow.com/questions/60950999/how-to-bind-c-structure-with-an-array-of-another-structure-as-a-member-using-py)

- [Returning and passing around raw POD pointers (arrays) with Python, C++, and pybind11](https://stackoverflow.com/questions/48982143/returning-and-passing-around-raw-pod-pointers-arrays-with-python-c-and-pyb)

- [passing pointer to C++ from python using pybind11](https://stackoverflow.com/questions/57990269/passing-pointer-to-c-from-python-using-pybind11)

- [Pybind11: Wrap a struct with a pointer member?](https://stackoverflow.com/questions/68292760/pybind11-wrap-a-struct-with-a-pointer-member)

- [pybind11-numpy-example](https://github.com/ssciwr/pybind11-numpy-example)

- [Dealing with Opaque Pointers in Pybind11](https://stackoverflow.com/questions/50641461/dealing-with-opaque-pointers-in-pybind11)

```cpp
PYBIND_MODULE(mymodule, m) {
  py::class_<mystruct>(m, "mystruct");
  m.def("f1", &myfunction1);
  m.def("f2", &myfunction2);
}
```

> If you wish to avoid conflict with other pybind11 modules that might declare types on this third-party type, consider using `py::module_local()` refered to in the docs [here](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings) 


see also:

- [PyBind11 : share opaque pointers between independently-built C++ modules through Python](https://stackoverflow.com/questions/76272413/pybind11-share-opaque-pointers-between-independently-built-c-modules-through)

- [Mixing type conversions and opaque types with pybind11](https://stackoverflow.com/questions/58169847/mixing-type-conversions-and-opaque-types-with-pybind11) also [issue-1940](https://github.com/pybind/pybind11/issues/1940)

- [Does there exists alternative for opaque pointer compared with boost python?](https://github.com/pybind/pybind11/issues/1778)


## GGUF Compatible Projects

(taken from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF))

- [llama.cpp](https://github.com/ggerganov/llama.cpp), the source project for GGUF. Offers a CLI and a server option.

- [text-generation-webui](https://github.com/oobabooga/text-generation-webui, the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.

- [KoboldCpp](https://github.com/LostRuins/koboldcpp), a fork of llama-cpp, fully featured web UI, with GPU accel across all platforms and GPU architectures. Especially good for story telling.

- [LM Studio](https://lmstudio.ai), an easy-to-use and powerful local GUI for Windows and macOS (Silicon), with GPU acceleration.

- [oLLMS Web UI](https://github.com/ParisNeo/lollms-webui), a great web UI with many interesting and unique features, including a full model library for easy model selection.

- [backyard.ai](https://backyard.ai/hub), an attractive and easy to use character-based chat GUI for Windows and macOS (both Silicon and Intel), with GPU acceleration.

- [ctransformers](https://github.com/marella/ctransformers), a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.

- [candle](https://github.com/huggingface/candle), a Rust ML framework with a focus on performance, including GPU support, and ease of use.

- [llama-cpp-agent](https://github.com/Maximilian-Winter/llama-cpp-agent) - The llama-cpp-agent framework is a tool designed for easy interaction with Large Language Models (LLMs). Allowing users to chat with LLM models, execute structured function calls and get structured output. Works also with models not fine-tuned to JSON output and function calls.

- [open-interpreter](https://github.com/OpenInterpreter/open-interpreter) - A natural language interface for computers

## Inference Engines

- [vllm](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient inference and serving engine for LLMs (python + c++)

- [text-generation-inference](https://github.com/huggingface/text-generation-inference) - Large Language Model Text Generation Inference by Huggingface (python + rust) for a comparison between vllm and tgi see [this article](https://medium.com/@rohit.k/tgi-vs-vllm-making-informed-choices-for-llm-deployment-37c56d7ff705)

- [localAI](https://github.com/mudler/LocalAI) - The free, Open Source alternative to OpenAI, Claude and others. Self-hosted and local-first. Drop-in replacement for OpenAI, running on consumer-grade hardware. No GPU required. Runs gguf, transformers, diffusers and many more models architectures. Features: Generate Text, Audio, Video, Images, Voice Cloning, Distributed inference

## RAGs

- [graphrag](https://github.com/microsoft/graphrag) - Microsoft modular graph-based Retrieval-Augmented Generation (RAG) system

- [ragatouille](https://github.com/AnswerDotAI/RAGatouille) - Easily use and train state of the art late-interaction retrieval methods (ColBERT) in any RAG pipeline. Designed for modularity and ease-of-use, backed by research.

- [llama_index](https://github.com/run-llama/llama_index) - LlamaIndex is a data framework for your LLM applications

- [LARS](https://github.com/abgulati/LARS) - An application for running LLMs locally on your device, with your documents, facilitating detailed citations in generated responses (llama.cpp-based)

- [kotaemon](https://github.com/Cinnamon/kotaemon) - An open-source RAG-based tool for chatting with your documents. (llamacpp-based)

- [experimenting-with-RAGs](https://github.com/AtaUllahB/experimenting-with-RAGs)

- [just-rag](https://github.com/samuelint/just-rag)

- [pymupdf/RAG](https://github.com/pymupdf/RAG) - RAG (Retrieval-Augmented Generation) Chatbot Examples Using PyMuPDF


