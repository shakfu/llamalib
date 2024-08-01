#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <llama.h>

#include <memory>

namespace nb = nanobind;



NB_MODULE(nbllama, m) {
    m.doc() = "nanobind pyllama wrapper"; // optional module docstring

    struct llama_model {};
    struct llama_context {};

    nb::class_<llama_token_data>(m, "llama_token_data")
        .def(nb::init<>())
        .def_rw("id", &llama_token_data::id)         // token id
        .def_rw("logit", &llama_token_data::logit)   // log-odds of the token
        .def_rw("p", &llama_token_data::p);          // probability of the token

    m.def("llama_context_default_params", (struct llama_context_params (*)()) &llama_context_default_params, "C++: llama_context_default_params() --> struct llama_context_params");

    m.def("llama_model_quantize_default_params", (struct llama_model_quantize_params (*)()) &llama_model_quantize_default_params, "C++: llama_model_quantize_default_params() --> struct llama_model_quantize_params");

    m.def("llama_backend_init", &llama_backend_init, "Initialize the llama + ggml backend");

    m.def("llama_numa_init", &llama_numa_init, "C++: llama_numa_init(enum ggml_numa_strategy) --> void", nb::arg("numa"));

    m.def("llama_backend_free", &llama_backend_free, "Call once at the end of the program - currently only used for MPI");

    m.def("llama_load_model_from_file", (struct llama_model (*)(const char *, struct llama_model_params *)) &llama_load_model_from_file, "Load a model from file", nb::arg("path_model"), nb::arg("params"));

    m.def("llama_time_us", &llama_time_us, "C++: llama_time_us() --> int");

    m.def("llama_max_devices", &llama_max_devices, "C++: llama_max_devices() --> int");

    m.def("llama_supports_mmap", &llama_supports_mmap, "C++: llama_supports_mmap() --> bool");

    m.def("llama_supports_mlock", &llama_supports_mlock, "C++: llama_supports_mlock() --> bool");

    m.def("llama_supports_gpu_offload", &llama_supports_gpu_offload, "C++: llama_supports_gpu_offload() --> bool");

    m.def("llama_model_quantize", (int (*)(const char *, const char *, const struct llama_model_quantize_params *)) &llama_model_quantize, "C++: llama_model_quantize(const char *, const char *, const struct llama_model_quantize_params *) --> int", nb::arg("fname_inp"), nb::arg("fname_out"), nb::arg("params"));

}
