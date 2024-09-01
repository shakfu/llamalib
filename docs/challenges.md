# challenging conversions


## logits

```c++

struct llama_context:

    // ...

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_t buf_output = nullptr;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;


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

