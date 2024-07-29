#include <pybind11/pybind11.h>

#include <llama.h>

namespace py = pybind11;


// typedef struct llama_model* t_model;


PYBIND11_MODULE(pyllama, m) {
    m.doc() = "pybind11 pyllama wrapper"; // optional module docstring

    // py::class_<struct llama_model>(m, "Model");

    py::class_<struct llama_model*>(m, "Model");
    py::class_<struct llama_context*>(m, "Context");

    m.def("backend_init", &llama_backend_init, "Initialize the llama + ggml backend.\n\nCall once at the start of a program.");
    m.def("backend_free", &llama_backend_free, "Call once at the end of the program - currently only used for MPI.");

    // m.def("free_model", [](t_model model) {
    //     llama_free_model(model);
    // }, "Free llama model.");

}