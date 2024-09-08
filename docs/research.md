# Research

## Llama

- [Meta's Llama page](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)

## llama.cpp

- [llama-cpp-python docs](https://llama-cpp-python.readthedocs.io/en/latest/)

- [Understanding how LLM inference works with llama.cpp](https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/)

- [datacamp llama.cpp tutorial](https://www.datacamp.com/tutorial/llama-cpp-tutorial)

- [Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)

- [langchain-llama-cpp](https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/)

- [lama.cpp: The Ultimate Guide to Efficient LLM Inference and Applications](https://pyimagesearch.com/2024/08/26/llama-cpp-the-ultimate-guide-to-efficient-llm-inference-and-applications/)

- [Llama.cpp Tutorial: A Complete Guide to Efficient LLM Inference and Implementation](https://www.datacamp.com/tutorial/llama-cpp-tutorial)

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

-  