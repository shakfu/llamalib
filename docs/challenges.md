# challenging conversions


## logits

[definition](https://developers.google.com/machine-learning/glossary/#logits):

The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving a multi-class classification problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.


```c++


```



## Convert to std::vector<>
```c++
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <common.h>
#include <llama.h>

namespace py = pybind11;



PYBIND11_MODULE(pbllama, m) {

    m.def("llama_get_logits", (float* (*)(const struct llama_context *)) &llama_get_logits, "", py::arg("ctx"));
    m.def("llama_get_logits_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_logits_ith, "", py::arg("ctx"), py::arg("i"));

    m.def("llama_get_embeddings", (float* (*)(const struct llama_context *)) &llama_get_embeddings, "", py::arg("ctx"));
    m.def("llama_get_embeddings_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_embeddings_ith, "", py::arg("ctx"), py::arg("i"));
    m.def("llama_get_embeddings_seq", (float* (*)(const struct llama_context *, llama_seq_id)) &llama_get_embeddings_seq, "", py::arg("ctx"), py::arg("seq_id"));

}
```

