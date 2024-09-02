# Research

## Links

- [Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)

- [datacamp llama.cpp tutorial](https://www.datacamp.com/tutorial/llama-cpp-tutorial)

- [llama-cpp-python docs](https://llama-cpp-python.readthedocs.io/en/latest/)

- [langchain-llama-cpp](https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/)

## Articles

- [Understanding how LLM inference works with llama.cpp](https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/)



## Releated Projects

- [llama.py](https://github.com/daskol/llama.py)

- [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp)

- [llamaindex](https://docs.llamaindex.ai) and [llama-cpp on llamaindex](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp/)


## raw pointers and pybind11

- [How to bind c structure with an array of another structure as a member in pybind11?](https://stackoverflow.com/questions/60950999/how-to-bind-c-structure-with-an-array-of-another-structure-as-a-member-using-py)


## numpy and pybind11

- [pybind11-numpy-example](https://github.com/ssciwr/pybind11-numpy-example)


## Opaque Pointers in Pybind11


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

-  