#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>



#include <common.h>
#include <llama.h>

namespace py = pybind11;


struct llama_model {};
struct llama_context {};
struct llama_lora_adapter {};


template <class T> class ptr_wrapper
{
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T* ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
        T& operator* () const { return *ptr; }
        T* operator->() const { return  ptr; }
        T* get() const { return ptr; }
        void destroy() { delete ptr; }
        T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};

py::array wrap_array_ptr(float *v) {
  auto capsule = py::capsule(
      &v, [](void *v) { delete reinterpret_cast<std::vector<float> *>(v); });
  return py::array(static_cast<pybind11::ssize_t>(sizeof(v)), v, capsule);
}


PYBIND11_MODULE(pbllama, m) {
    m.doc() = "pbllama: pybind11 llama.cpp wrapper"; // optional module docstring
    m.attr("__version__") = "0.0.1";


    // -----------------------------------------------------------------------
    // attributes
    m.attr("LLAMA_DEFAULT_SEED") = 0xFFFFFFFF;

    // -----------------------------------------------------------------------
    // ggml.h

    py::enum_<ggml_numa_strategy>(m, "ggml_numa_strategy", py::arithmetic(), "")
        .value("GGML_NUMA_STRATEGY_DISABLED", GGML_NUMA_STRATEGY_DISABLED)
        .value("GGML_NUMA_STRATEGY_DISTRIBUTE", GGML_NUMA_STRATEGY_DISTRIBUTE)
        .value("GGML_NUMA_STRATEGY_ISOLATE", GGML_NUMA_STRATEGY_ISOLATE)
        .value("GGML_NUMA_STRATEGY_NUMACTL", GGML_NUMA_STRATEGY_NUMACTL)
        .value("GGML_NUMA_STRATEGY_MIRROR", GGML_NUMA_STRATEGY_MIRROR)
        .value("GGML_NUMA_STRATEGY_COUNT", GGML_NUMA_STRATEGY_COUNT)
        .export_values();

    m.def("ggml_time_us", (int64_t (*)(void)) &ggml_time_us);

    // -----------------------------------------------------------------------
    // llama.h

    py::class_<llama_model> (m, "llama_model", "")
        .def(py::init<>());

    py::class_<llama_context> (m, "llama_context", "")
        .def(py::init<>());

    py::class_<llama_lora_adapter> (m, "llama_lora_adapter", "")
        .def(py::init<>());

    py::enum_<enum llama_vocab_type>(m, "llama_vocab_type", py::arithmetic(), "")
        .value("LLAMA_VOCAB_TYPE_NONE", LLAMA_VOCAB_TYPE_NONE)
        .value("LLAMA_VOCAB_TYPE_SPM", LLAMA_VOCAB_TYPE_SPM)
        .value("LLAMA_VOCAB_TYPE_BPE", LLAMA_VOCAB_TYPE_BPE)
        .value("LLAMA_VOCAB_TYPE_WPM", LLAMA_VOCAB_TYPE_WPM)
        .value("LLAMA_VOCAB_TYPE_UGM", LLAMA_VOCAB_TYPE_UGM)
        .export_values();

    py::enum_<enum llama_vocab_pre_type>(m, "llama_vocab_pre_type", py::arithmetic(), "")
        .value("LLAMA_VOCAB_PRE_TYPE_DEFAULT", LLAMA_VOCAB_PRE_TYPE_DEFAULT)
        .value("LLAMA_VOCAB_PRE_TYPE_LLAMA3", LLAMA_VOCAB_PRE_TYPE_LLAMA3)
        .value("LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM", LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM)
        .value("LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER", LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER)
        .value("LLAMA_VOCAB_PRE_TYPE_FALCON", LLAMA_VOCAB_PRE_TYPE_FALCON)
        .value("LLAMA_VOCAB_PRE_TYPE_MPT", LLAMA_VOCAB_PRE_TYPE_MPT)
        .value("LLAMA_VOCAB_PRE_TYPE_STARCODER", LLAMA_VOCAB_PRE_TYPE_STARCODER)
        .value("LLAMA_VOCAB_PRE_TYPE_GPT2", LLAMA_VOCAB_PRE_TYPE_GPT2)
        .value("LLAMA_VOCAB_PRE_TYPE_REFACT", LLAMA_VOCAB_PRE_TYPE_REFACT)
        .value("LLAMA_VOCAB_PRE_TYPE_COMMAND_R", LLAMA_VOCAB_PRE_TYPE_COMMAND_R)
        .value("LLAMA_VOCAB_PRE_TYPE_STABLELM2", LLAMA_VOCAB_PRE_TYPE_STABLELM2)
        .value("LLAMA_VOCAB_PRE_TYPE_QWEN2", LLAMA_VOCAB_PRE_TYPE_QWEN2)
        .value("LLAMA_VOCAB_PRE_TYPE_OLMO", LLAMA_VOCAB_PRE_TYPE_OLMO)
        .value("LLAMA_VOCAB_PRE_TYPE_DBRX", LLAMA_VOCAB_PRE_TYPE_DBRX)
        .value("LLAMA_VOCAB_PRE_TYPE_SMAUG", LLAMA_VOCAB_PRE_TYPE_SMAUG)
        .value("LLAMA_VOCAB_PRE_TYPE_PORO", LLAMA_VOCAB_PRE_TYPE_PORO)
        .value("LLAMA_VOCAB_PRE_TYPE_CHATGLM3", LLAMA_VOCAB_PRE_TYPE_CHATGLM3)
        .value("LLAMA_VOCAB_PRE_TYPE_CHATGLM4", LLAMA_VOCAB_PRE_TYPE_CHATGLM4)
        .value("LLAMA_VOCAB_PRE_TYPE_VIKING", LLAMA_VOCAB_PRE_TYPE_VIKING)
        .value("LLAMA_VOCAB_PRE_TYPE_JAIS", LLAMA_VOCAB_PRE_TYPE_JAIS)
        .value("LLAMA_VOCAB_PRE_TYPE_TEKKEN", LLAMA_VOCAB_PRE_TYPE_TEKKEN)
        .value("LLAMA_VOCAB_PRE_TYPE_SMOLLM", LLAMA_VOCAB_PRE_TYPE_SMOLLM)
        .value("LLAMA_VOCAB_PRE_TYPE_CODESHELL", LLAMA_VOCAB_PRE_TYPE_CODESHELL)
        .value("LLAMA_VOCAB_PRE_TYPE_BLOOM", LLAMA_VOCAB_PRE_TYPE_BLOOM)
        .value("LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH", LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH)
        .value("LLAMA_VOCAB_PRE_TYPE_EXAONE", LLAMA_VOCAB_PRE_TYPE_EXAONE)
        .export_values();

    py::enum_<enum llama_rope_type>(m, "llama_rope_type", py::arithmetic(), "")
        .value("LLAMA_ROPE_TYPE_NONE", LLAMA_ROPE_TYPE_NONE)
        .value("LLAMA_ROPE_TYPE_NORM", LLAMA_ROPE_TYPE_NORM)
        .value("LLAMA_ROPE_TYPE_NEOX", LLAMA_ROPE_TYPE_NEOX)
        .export_values();

    py::enum_<enum llama_token_type>(m, "llama_token_type", py::arithmetic(), "")
        .value("LLAMA_TOKEN_TYPE_UNDEFINED", LLAMA_TOKEN_TYPE_UNDEFINED)
        .value("LLAMA_TOKEN_TYPE_NORMAL", LLAMA_TOKEN_TYPE_NORMAL)
        .value("LLAMA_TOKEN_TYPE_UNKNOWN", LLAMA_TOKEN_TYPE_UNKNOWN)
        .value("LLAMA_TOKEN_TYPE_CONTROL", LLAMA_TOKEN_TYPE_CONTROL)
        .value("LLAMA_TOKEN_TYPE_USER_DEFINED", LLAMA_TOKEN_TYPE_USER_DEFINED)
        .value("LLAMA_TOKEN_TYPE_UNUSED", LLAMA_TOKEN_TYPE_UNUSED)
        .value("LLAMA_TOKEN_TYPE_BYTE", LLAMA_TOKEN_TYPE_BYTE)
        .export_values();

    py::enum_<enum llama_token_attr>(m, "llama_token_attr", py::arithmetic(), "")
        .value("LLAMA_TOKEN_ATTR_UNDEFINED", LLAMA_TOKEN_ATTR_UNDEFINED)
        .value("LLAMA_TOKEN_ATTR_UNKNOWN", LLAMA_TOKEN_ATTR_UNKNOWN)
        .value("LLAMA_TOKEN_ATTR_UNUSED", LLAMA_TOKEN_ATTR_UNUSED)
        .value("LLAMA_TOKEN_ATTR_NORMAL", LLAMA_TOKEN_ATTR_NORMAL)
        .value("LLAMA_TOKEN_ATTR_CONTROL", LLAMA_TOKEN_ATTR_CONTROL)
        .value("LLAMA_TOKEN_ATTR_USER_DEFINED", LLAMA_TOKEN_ATTR_USER_DEFINED)
        .value("LLAMA_TOKEN_ATTR_BYTE", LLAMA_TOKEN_ATTR_BYTE)
        .value("LLAMA_TOKEN_ATTR_NORMALIZED", LLAMA_TOKEN_ATTR_NORMALIZED)
        .value("LLAMA_TOKEN_ATTR_LSTRIP", LLAMA_TOKEN_ATTR_LSTRIP)
        .value("LLAMA_TOKEN_ATTR_RSTRIP", LLAMA_TOKEN_ATTR_RSTRIP)
        .value("LLAMA_TOKEN_ATTR_SINGLE_WORD", LLAMA_TOKEN_ATTR_SINGLE_WORD)
        .export_values();

    py::enum_<enum llama_ftype>(m, "llama_ftype", py::arithmetic(), "")
        .value("LLAMA_FTYPE_ALL_F32", LLAMA_FTYPE_ALL_F32)
        .value("LLAMA_FTYPE_MOSTLY_F16", LLAMA_FTYPE_MOSTLY_F16)
        .value("LLAMA_FTYPE_MOSTLY_Q4_0", LLAMA_FTYPE_MOSTLY_Q4_0)
        .value("LLAMA_FTYPE_MOSTLY_Q4_1", LLAMA_FTYPE_MOSTLY_Q4_1)
        .value("LLAMA_FTYPE_MOSTLY_Q8_0", LLAMA_FTYPE_MOSTLY_Q8_0)
        .value("LLAMA_FTYPE_MOSTLY_Q5_0", LLAMA_FTYPE_MOSTLY_Q5_0)
        .value("LLAMA_FTYPE_MOSTLY_Q5_1", LLAMA_FTYPE_MOSTLY_Q5_1)
        .value("LLAMA_FTYPE_MOSTLY_Q2_K", LLAMA_FTYPE_MOSTLY_Q2_K)
        .value("LLAMA_FTYPE_MOSTLY_Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S)
        .value("LLAMA_FTYPE_MOSTLY_Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M)
        .value("LLAMA_FTYPE_MOSTLY_Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L)
        .value("LLAMA_FTYPE_MOSTLY_Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S)
        .value("LLAMA_FTYPE_MOSTLY_Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M)
        .value("LLAMA_FTYPE_MOSTLY_Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S)
        .value("LLAMA_FTYPE_MOSTLY_Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M)
        .value("LLAMA_FTYPE_MOSTLY_Q6_K", LLAMA_FTYPE_MOSTLY_Q6_K)
        .value("LLAMA_FTYPE_MOSTLY_IQ2_XXS", LLAMA_FTYPE_MOSTLY_IQ2_XXS)
        .value("LLAMA_FTYPE_MOSTLY_IQ2_XS", LLAMA_FTYPE_MOSTLY_IQ2_XS)
        .value("LLAMA_FTYPE_MOSTLY_Q2_K_S", LLAMA_FTYPE_MOSTLY_Q2_K_S)
        .value("LLAMA_FTYPE_MOSTLY_IQ3_XS", LLAMA_FTYPE_MOSTLY_IQ3_XS)
        .value("LLAMA_FTYPE_MOSTLY_IQ3_XXS", LLAMA_FTYPE_MOSTLY_IQ3_XXS)
        .value("LLAMA_FTYPE_MOSTLY_IQ1_S", LLAMA_FTYPE_MOSTLY_IQ1_S)
        .value("LLAMA_FTYPE_MOSTLY_IQ4_NL", LLAMA_FTYPE_MOSTLY_IQ4_NL)
        .value("LLAMA_FTYPE_MOSTLY_IQ3_S", LLAMA_FTYPE_MOSTLY_IQ3_S)
        .value("LLAMA_FTYPE_MOSTLY_IQ3_M", LLAMA_FTYPE_MOSTLY_IQ3_M)
        .value("LLAMA_FTYPE_MOSTLY_IQ2_S", LLAMA_FTYPE_MOSTLY_IQ2_S)
        .value("LLAMA_FTYPE_MOSTLY_IQ2_M", LLAMA_FTYPE_MOSTLY_IQ2_M)
        .value("LLAMA_FTYPE_MOSTLY_IQ4_XS", LLAMA_FTYPE_MOSTLY_IQ4_XS)
        .value("LLAMA_FTYPE_MOSTLY_IQ1_M", LLAMA_FTYPE_MOSTLY_IQ1_M)
        .value("LLAMA_FTYPE_MOSTLY_BF16", LLAMA_FTYPE_MOSTLY_BF16)
        .value("LLAMA_FTYPE_MOSTLY_Q4_0_4_4", LLAMA_FTYPE_MOSTLY_Q4_0_4_4)
        .value("LLAMA_FTYPE_MOSTLY_Q4_0_4_8", LLAMA_FTYPE_MOSTLY_Q4_0_4_8)
        .value("LLAMA_FTYPE_MOSTLY_Q4_0_8_8", LLAMA_FTYPE_MOSTLY_Q4_0_8_8)
        .value("LLAMA_FTYPE_MOSTLY_TQ1_0", LLAMA_FTYPE_MOSTLY_TQ1_0)
        .value("LLAMA_FTYPE_MOSTLY_TQ2_0", LLAMA_FTYPE_MOSTLY_TQ2_0)
        .value("LLAMA_FTYPE_GUESSED", LLAMA_FTYPE_GUESSED)
        .export_values();

    py::enum_<enum llama_rope_scaling_type>(m, "llama_rope_scaling_type", py::arithmetic(), "")
        .value("LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED", LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED)
        .value("LLAMA_ROPE_SCALING_TYPE_NONE", LLAMA_ROPE_SCALING_TYPE_NONE)
        .value("LLAMA_ROPE_SCALING_TYPE_LINEAR", LLAMA_ROPE_SCALING_TYPE_LINEAR)
        .value("LLAMA_ROPE_SCALING_TYPE_YARN", LLAMA_ROPE_SCALING_TYPE_YARN)
        .value("LLAMA_ROPE_SCALING_TYPE_MAX_VALUE", LLAMA_ROPE_SCALING_TYPE_MAX_VALUE)
        .export_values();

    py::enum_<enum llama_pooling_type>(m, "llama_pooling_type", py::arithmetic(), "")
        .value("LLAMA_POOLING_TYPE_UNSPECIFIED", LLAMA_POOLING_TYPE_UNSPECIFIED)
        .value("LLAMA_POOLING_TYPE_NONE", LLAMA_POOLING_TYPE_NONE)
        .value("LLAMA_POOLING_TYPE_MEAN", LLAMA_POOLING_TYPE_MEAN)
        .value("LLAMA_POOLING_TYPE_CLS", LLAMA_POOLING_TYPE_CLS)
        .value("LLAMA_POOLING_TYPE_LAST", LLAMA_POOLING_TYPE_LAST)
        .export_values();

    py::enum_<enum llama_attention_type>(m, "llama_attention_type", py::arithmetic(), "")
        .value("LLAMA_ATTENTION_TYPE_UNSPECIFIED", LLAMA_ATTENTION_TYPE_UNSPECIFIED)
        .value("LLAMA_ATTENTION_TYPE_CAUSAL", LLAMA_ATTENTION_TYPE_CAUSAL)
        .value("LLAMA_ATTENTION_TYPE_NON_CAUSAL", LLAMA_ATTENTION_TYPE_NON_CAUSAL)
        .export_values();

    py::enum_<enum llama_split_mode>(m, "llama_split_mode", py::arithmetic(), "")
        .value("LLAMA_SPLIT_MODE_NONE", LLAMA_SPLIT_MODE_NONE)
        .value("LLAMA_SPLIT_MODE_LAYER", LLAMA_SPLIT_MODE_LAYER)
        .value("LLAMA_SPLIT_MODE_ROW", LLAMA_SPLIT_MODE_ROW)
        .export_values();

    py::class_<llama_token_data, std::shared_ptr<llama_token_data>>(m, "llama_token_data")
        .def( py::init( [](){ return new llama_token_data(); } ))
        .def_readwrite("id", &llama_token_data::id)         // token id
        .def_readwrite("logit", &llama_token_data::logit)   // log-odds of the token
        .def_readwrite("p", &llama_token_data::p);          // probability of the token


    py::class_<llama_token_data_array, std::shared_ptr<llama_token_data_array>> (m, "llama_token_data_array", "")
        .def( py::init( [](){ return new llama_token_data_array(); } ) )
        .def_readwrite("size", &llama_token_data_array::size)
        .def_readwrite("selected", &llama_token_data_array::selected)
        .def_readwrite("sorted", &llama_token_data_array::sorted);
        // llama_token  *  token;
        // float        *  embd;
        // llama_pos    *  pos;
        // int32_t      *  n_seq_id;
        // llama_seq_id ** seq_id;
        // int8_t       *  logits; // TODO: rename this to "output"


    py::class_<llama_batch, std::shared_ptr<llama_batch>> (m, "llama_batch", "")
        .def( py::init( [](){ return new llama_batch(); } ) )
        .def_readwrite("n_tokens", &llama_batch::n_tokens)
        // .def_readwrite("token", &llama_batch::token)
        // .def_readwrite("embd", &llama_batch::embd)
        // .def_readwrite("pos", &llama_batch::pos)
        // .def_readwrite("n_seq_id", &llama_batch::n_seq_id)
        // .def_readwrite("seq_id", &llama_batch::seq_id)
        // FIXME: this is WRONG!!
        .def_property_readonly("logits", [](llama_batch& self) -> std::vector<int8_t> {
            std::vector<int8_t> result(self.logits, self.logits + self.n_tokens);
            return result;
        })
        .def_readwrite("all_pos_0", &llama_batch::all_pos_0)
        .def_readwrite("all_pos_1", &llama_batch::all_pos_1)
        .def_readwrite("all_seq_id", &llama_batch::all_seq_id);

    py::enum_<llama_model_kv_override_type>(m, "llama_model_kv_override_type", py::arithmetic(), "")
        .value("LLAMA_KV_OVERRIDE_TYPE_INT", LLAMA_KV_OVERRIDE_TYPE_INT)
        .value("LLAMA_KV_OVERRIDE_TYPE_FLOAT", LLAMA_KV_OVERRIDE_TYPE_FLOAT)
        .value("LLAMA_KV_OVERRIDE_TYPE_BOOL", LLAMA_KV_OVERRIDE_TYPE_BOOL)
        .value("LLAMA_KV_OVERRIDE_TYPE_STR", LLAMA_KV_OVERRIDE_TYPE_STR)
        .export_values();

    py::class_<llama_model_kv_override, std::shared_ptr<llama_model_kv_override>> (m, "llama_model_kv_override", "")
        .def( py::init( [](){ return new llama_model_kv_override(); } ) )
        .def_readwrite("tag", &llama_model_kv_override::tag);
    
    py::class_<llama_model_params, std::shared_ptr<llama_model_params>> (m, "llama_model_params", "")
        .def( py::init( [](){ return new llama_model_params(); } ) )
        .def_readwrite("n_gpu_layers", &llama_model_params::n_gpu_layers)
        .def_readwrite("split_mode", &llama_model_params::split_mode)
        .def_readwrite("main_gpu", &llama_model_params::main_gpu)
        .def_readwrite("vocab_only", &llama_model_params::vocab_only)
        .def_readwrite("use_mmap", &llama_model_params::use_mmap)
        .def_readwrite("use_mlock", &llama_model_params::use_mlock)
        .def_readwrite("check_tensors", &llama_model_params::check_tensors);
    
    py::class_<llama_context_params, std::shared_ptr<llama_context_params>> (m, "llama_context_params", "")
        .def( py::init( [](){ return new llama_context_params(); } ) )
        .def_readwrite("n_ctx", &llama_context_params::n_ctx)
        .def_readwrite("n_batch", &llama_context_params::n_batch)
        .def_readwrite("n_ubatch", &llama_context_params::n_ubatch)
        .def_readwrite("n_seq_max", &llama_context_params::n_seq_max)
        .def_readwrite("n_threads", &llama_context_params::n_threads)
        .def_readwrite("n_threads_batch", &llama_context_params::n_threads_batch)
        .def_readwrite("rope_scaling_type", &llama_context_params::rope_scaling_type)
        .def_readwrite("pooling_type", &llama_context_params::pooling_type)
        .def_readwrite("attention_type", &llama_context_params::attention_type)
        .def_readwrite("rope_freq_base", &llama_context_params::rope_freq_base)
        .def_readwrite("rope_freq_scale", &llama_context_params::rope_freq_scale)
        .def_readwrite("yarn_ext_factor", &llama_context_params::yarn_ext_factor)
        .def_readwrite("yarn_attn_factor", &llama_context_params::yarn_attn_factor)
        .def_readwrite("yarn_beta_fast", &llama_context_params::yarn_beta_fast)
        .def_readwrite("yarn_beta_slow", &llama_context_params::yarn_beta_slow)
        .def_readwrite("yarn_orig_ctx", &llama_context_params::yarn_orig_ctx)
        .def_readwrite("defrag_thold", &llama_context_params::defrag_thold)
        .def_readwrite("type_k", &llama_context_params::type_k)
        .def_readwrite("type_v", &llama_context_params::type_v)
        .def_readwrite("logits_all", &llama_context_params::logits_all)
        .def_readwrite("embeddings", &llama_context_params::embeddings)
        .def_readwrite("offload_kqv", &llama_context_params::offload_kqv)
        .def_readwrite("flash_attn", &llama_context_params::flash_attn);

    py::class_<llama_model_quantize_params, std::shared_ptr<llama_model_quantize_params>> (m, "llama_model_quantize_params", "")
        .def( py::init( [](){ return new llama_model_quantize_params(); } ) )
        .def_readwrite("nthread", &llama_model_quantize_params::nthread)
        .def_readwrite("ftype", &llama_model_quantize_params::ftype)
        .def_readwrite("output_tensor_type", &llama_model_quantize_params::output_tensor_type)
        .def_readwrite("token_embedding_type", &llama_model_quantize_params::token_embedding_type)
        .def_readwrite("allow_requantize", &llama_model_quantize_params::allow_requantize)
        .def_readwrite("quantize_output_tensor", &llama_model_quantize_params::quantize_output_tensor)
        .def_readwrite("only_copy", &llama_model_quantize_params::only_copy)
        .def_readwrite("pure", &llama_model_quantize_params::pure);

    py::class_<llama_logit_bias, std::shared_ptr<llama_logit_bias>> (m, "llama_logit_bias", "")
        .def( py::init( [](){ return new llama_logit_bias(); } ) );
        // .def_readwrite("token", &llama_model_quantize_params::token)
        // .def_readwrite("bias", &llama_model_quantize_params::bias);

    py::class_<llama_sampler_chain_params, std::shared_ptr<llama_sampler_chain_params>> (m, "llama_sampler_chain_params", "")
        .def( py::init( [](){ return new llama_sampler_chain_params(); } ) )
        .def_readwrite("no_perf", &llama_sampler_chain_params::no_perf);

    py::class_<llama_chat_message, std::shared_ptr<llama_chat_message>> (m, "llama_chat_message", "")
        .def( py::init( [](){ return new llama_chat_message(); } ) )
        .def_readwrite("role", &llama_chat_message::role)
        .def_readwrite("content", &llama_chat_message::content);


    m.def("llama_model_default_params", (struct llama_model_params (*)()) &llama_model_default_params);
    m.def("llama_context_default_params", (struct llama_context_params (*)()) &llama_context_default_params);
    m.def("llama_sampler_chain_default_params", (struct llama_sampler_chain_params (*)()) &llama_sampler_chain_default_params);
    m.def("llama_model_quantize_default_params", (struct llama_model_quantize_params (*)()) &llama_model_quantize_default_params);
    m.def("llama_backend_init", (void (*)()) &llama_backend_init, "Initialize the llama + ggml backend.");
    m.def("llama_numa_init", (void (*)(enum ggml_numa_strategy)) &llama_numa_init, "C++: llama_numa_init(enum ggml_numa_strategy) --> void", py::arg("numa"));
    m.def("llama_backend_free", (void (*)()) &llama_backend_free, "Call once at the end of the program.");
    m.def("llama_load_model_from_file", (struct llama_model * (*)(const char *, struct llama_model_params *)) &llama_load_model_from_file, "Load a model from file", py::arg("path_model"), py::arg("params"), py::return_value_policy::reference);
    m.def("llama_free_model", (void (*)(struct llama_model *)) &llama_free_model, "Free a model", py::arg("model"));

    m.def("llama_new_context_with_model", (struct llama_context * (*)(struct llama_model *, struct llama_context_params)) &llama_new_context_with_model, "New context with model", py::arg("model"), py::arg("params"), py::return_value_policy::reference);
    
    m.def("llama_free", (void (*)(struct llama_context *)) &llama_free, "Free context", py::arg("ctx"));
    m.def("llama_time_us", (int64_t (*)()) &llama_time_us, "C++: llama_time_us() --> int");
    m.def("llama_max_devices", (size_t (*)()) &llama_max_devices, "C++: llama_max_devices() --> int");
    m.def("llama_supports_mmap", (bool (*)()) &llama_supports_mmap, "C++: llama_supports_mmap() --> bool");
    m.def("llama_supports_mlock", (bool (*)()) &llama_supports_mlock, "C++: llama_supports_mlock() --> bool");
    m.def("llama_supports_gpu_offload", (bool (*)()) &llama_supports_gpu_offload, "C++: llama_supports_gpu_offload() --> bool");

    m.def("llama_n_ctx", (uint32_t (*)(const struct llama_context *)) &llama_n_ctx, "get n_ctx from context", py::arg("ctx"));
    m.def("llama_n_batch", (uint32_t (*)(const struct llama_context *)) &llama_n_batch, "get n_batch from context", py::arg("ctx"));
    m.def("llama_n_ubatch", (uint32_t (*)(const struct llama_context *)) &llama_n_ubatch, "get n_ubatch from context", py::arg("ctx"));
    m.def("llama_n_seq_max", (uint32_t (*)(const struct llama_context *)) &llama_n_seq_max, "get n_seq_max from context", py::arg("ctx"));

    m.def("llama_n_vocab", (int32_t (*)(const struct llama_model *)) &llama_n_vocab, "get n_vocab from model", py::arg("model"));
    m.def("llama_n_ctx_train", (int32_t (*)(const struct llama_model *)) &llama_n_ctx_train, "get n_ctx_train from model", py::arg("model"));
    m.def("llama_n_embd", (int32_t (*)(const struct llama_model *)) &llama_n_embd, "get n_embed from model", py::arg("model"));
    m.def("llama_n_layer", (int32_t (*)(const struct llama_model *)) &llama_n_layer, "get n_layer from model", py::arg("model"));

    m.def("llama_get_model", (const struct llama_model * (*)(const struct llama_context *)) &llama_get_model, "get model from context", py::arg("ctx"));
    m.def("get_llama_pooling_type", (enum llama_pooling_type (*)(const struct llama_context *)) &llama_pooling_type, "get pooling_type from context", py::arg("ctx"));
    m.def("get_llama_vocab_type", (enum llama_vocab_type (*)(const struct llama_model *)) &llama_vocab_type, "get vocab_type from model", py::arg("model"));
    m.def("get_llama_rope_type", (enum llama_rope_type (*)(const struct llama_model *)) &llama_rope_type, "get rope_type from model", py::arg("model"));

    m.def("llama_rope_freq_scale_train", (float (*)(const struct llama_model *)) &llama_rope_freq_scale_train, "get rope_freq_scale_train from model", py::arg("model"));

    m.def("llama_model_meta_val_str", (int32_t (*)(const struct llama_model *, const char *, char *, size_t)) &llama_model_meta_val_str, "get meta_val_str from model", py::arg("model"), py::arg("key"), py::arg("buf"), py::arg("buf_size"));
    m.def("llama_model_meta_count", (int32_t (*)(const struct llama_model *)) &llama_model_meta_count, "get meta_count from model", py::arg("model"));
    m.def("llama_model_meta_key_by_index", (int32_t (*)(const struct llama_model *, int32_t, char *, size_t)) &llama_model_meta_key_by_index, "get meta_key_by_index from model", py::arg("model"), py::arg("i"), py::arg("buf"), py::arg("buf_size"));
    m.def("llama_model_meta_val_str_by_index", (int32_t (*)(const struct llama_model *, int32_t, char *, size_t)) &llama_model_meta_val_str_by_index, "get meta_val_str_by_index from model", py::arg("model"), py::arg("i"), py::arg("buf"), py::arg("buf_size"));
    m.def("llama_model_desc", (int32_t (*)(const struct llama_model *, char *, size_t)) &llama_model_desc, "get model_desc from model", py::arg("model"), py::arg("buf"), py::arg("buf_size"));

    m.def("llama_model_size", (uint64_t (*)(const struct llama_model *)) &llama_model_size, "get model_size from model", py::arg("model"));
    m.def("llama_model_n_params", (uint64_t (*)(const struct llama_model *)) &llama_model_n_params, "get model_n_params from model", py::arg("model"));

    m.def("llama_get_model_tensor", (struct ggml_tensor* (*)(struct llama_model *)) &llama_get_model_tensor, "get model tensor from model", py::arg("model"));
    m.def("llama_get_model_tensor", (struct ggml_tensor* (*)(struct llama_model *, const char *)) &llama_get_model_tensor, "get named model tensor from model", py::arg("model"), py::arg("name"));

    m.def("llama_model_has_encoder", (bool (*)(const struct llama_model *)) &llama_model_has_encoder, "model has encoder?", py::arg("model"));

    m.def("llama_model_decoder_start_token", (llama_token (*)(const struct llama_model *)) &llama_model_decoder_start_token, "get decoder_start_token from model", py::arg("model"));

    m.def("llama_model_is_recurrent", (bool (*)(const struct llama_model *)) &llama_model_is_recurrent, "check if model is recurrent", py::arg("model"));

    m.def("llama_model_quantize", (int (*)(const char *, const char *, const struct llama_model_quantize_params *)) &llama_model_quantize, "C++: llama_model_quantize(const char *, const char *, const struct llama_model_quantize_params *) --> int", py::arg("fname_inp"), py::arg("fname_out"), py::arg("params"));

    m.def("llama_lora_adapter_init", (struct llama_lora_adapter (*)(const struct llama_model *, const char *)) &llama_lora_adapter_init, "", py::arg("model"), py::arg("path_lora"));
    m.def("llama_lora_adapter_set", (int32_t (*)(struct llama_context*, struct llama_lora_adapter*, float)) &llama_lora_adapter_set, "", py::arg("ctx"), py::arg("adapter"), py::arg("scale"));
    m.def("llama_lora_adapter_remove", (int32_t (*)(struct llama_context*, struct llama_lora_adapter*)) &llama_lora_adapter_remove, "", py::arg("ctx"), py::arg("adapter"));
    m.def("llama_lora_adapter_clear", (void (*)(struct llama_context*)) &llama_lora_adapter_clear, "", py::arg("ctx"));
    m.def("llama_lora_adapter_free", (void (*)(struct llama_lora_adapter*)) &llama_lora_adapter_free, "", py::arg("adapter"));

    m.def("llama_control_vector_apply", (int32_t (*)(struct llama_context * , const float*, size_t, int32_t, int32_t, int32_t)) &llama_model_quantize, "", py::arg("lctx"), py::arg("data"), py::arg("len"), py::arg("n_embd"), py::arg("il_start"), py::arg("il_end"));

    py::class_<llama_kv_cache_view_cell, std::shared_ptr<llama_kv_cache_view_cell>>(m, "llama_kv_cache_view_cell")
        .def( py::init( [](){ return new llama_kv_cache_view_cell(); } ))
        .def_readwrite("pos", &llama_kv_cache_view_cell::pos);

    py::class_<llama_kv_cache_view, std::shared_ptr<llama_kv_cache_view>>(m, "llama_kv_cache_view", "")
        .def( py::init( [](){ return new llama_kv_cache_view(); } ) )
        .def_readwrite("n_cells", &llama_kv_cache_view::n_cells)
        .def_readwrite("n_seq_max", &llama_kv_cache_view::n_seq_max)
        .def_readwrite("token_count", &llama_kv_cache_view::token_count)
        .def_readwrite("used_cells", &llama_kv_cache_view::used_cells)
        .def_readwrite("max_contiguous", &llama_kv_cache_view::max_contiguous)
        .def_readwrite("max_contiguous_idx", &llama_kv_cache_view::max_contiguous_idx);

    m.def("llama_kv_cache_view_init", (struct llama_kv_cache_view (*)(const struct llama_context *, int32_t)) &llama_kv_cache_view_init, "", py::arg("ctx"), py::arg("n_seq_max"));
    m.def("llama_kv_cache_view_free", (void (*)(struct llama_kv_cache_view *)) &llama_kv_cache_view_free, "", py::arg("view"));
    m.def("llama_kv_cache_view_update", (void (*)(const struct llama_context *, struct llama_kv_cache_view *)) &llama_kv_cache_view_update, "", py::arg("ctx"), py::arg("view"));
    m.def("llama_get_kv_cache_token_count", (int32_t (*)(const struct llama_context *)) &llama_get_kv_cache_token_count, "", py::arg("ctx"));
    m.def("llama_get_kv_cache_used_cells", (int32_t (*)(const struct llama_context *)) &llama_get_kv_cache_used_cells, "", py::arg("ctx"));
    m.def("llama_kv_cache_clear", (void (*)(const struct llama_context *)) &llama_kv_cache_clear, "", py::arg("ctx"));
    m.def("llama_kv_cache_seq_rm", (bool (*)(const struct llama_context *, llama_seq_id, llama_pos, llama_pos)) &llama_kv_cache_seq_rm, "", py::arg("ctx"), py::arg("seq_id"), py::arg("p0"), py::arg("p1"));
    m.def("llama_kv_cache_seq_cp", (void (*)(const struct llama_context *, llama_seq_id, llama_seq_id, llama_pos, llama_pos)) &llama_kv_cache_seq_cp, "", py::arg("ctx"), py::arg("seq_id_src"), py::arg("seq_id_dst"), py::arg("p0"), py::arg("p1"));
    m.def("llama_kv_cache_seq_keep", (void (*)(const struct llama_context *, llama_seq_id)) &llama_kv_cache_seq_keep, "", py::arg("ctx"), py::arg("seq_id"));
    m.def("llama_kv_cache_seq_add", (void (*)(const struct llama_context *, llama_seq_id, llama_pos, llama_pos, llama_pos)) &llama_kv_cache_seq_add, "", py::arg("ctx"), py::arg("seq_id"), py::arg("p0"), py::arg("p1"), py::arg("delta"));
    m.def("llama_kv_cache_seq_div", (void (*)(const struct llama_context *, llama_seq_id, llama_pos, llama_pos, int)) &llama_kv_cache_seq_div, "", py::arg("ctx"), py::arg("seq_id"), py::arg("p0"), py::arg("p1"), py::arg("d"));
    m.def("llama_kv_cache_seq_pos_max", (llama_pos (*)(const struct llama_context *, llama_seq_id)) &llama_kv_cache_seq_pos_max, "", py::arg("ctx"), py::arg("seq_id"));
    m.def("llama_kv_cache_defrag", (void (*)(const struct llama_context *)) &llama_kv_cache_defrag, "", py::arg("ctx"));
    m.def("llama_kv_cache_update", (void (*)(const struct llama_context *)) &llama_kv_cache_update, "", py::arg("ctx"));

    m.def("llama_batch_get_one", (struct llama_batch (*)(int *, int, int, int)) &llama_batch_get_one, "C++: llama_batch_get_one(int *, int, int, int) --> struct llama_batch", py::arg("tokens"), py::arg("n_tokens"), py::arg("pos_0"), py::arg("seq_id"));
    m.def("llama_batch_init", (struct llama_batch (*)(int, int, int)) &llama_batch_init, "C++: llama_batch_init(int, int, int) --> struct llama_batch", py::arg("n_tokens"), py::arg("embd"), py::arg("n_seq_max"));
    m.def("llama_batch_free", (void (*)(struct llama_batch)) &llama_batch_free, "C++: llama_batch_free(struct llama_batch) --> void", py::arg("batch"));

    m.def("llama_encode", (int32_t (*)(const struct llama_context *, struct llama_batch)) &llama_encode, "", py::arg("ctx"), py::arg("batch"));
    m.def("llama_decode", (int32_t (*)(const struct llama_context *, struct llama_batch)) &llama_decode, "", py::arg("ctx"), py::arg("batch"));

    m.def("llama_set_n_threads", (void (*)(const struct llama_context *, uint32_t, uint32_t)) &llama_set_n_threads, "", py::arg("ctx"), py::arg("n_threads"), py::arg("n_threads_batch"));
    m.def("llama_n_threads", (uint32_t (*)(const struct llama_context *)) &llama_n_threads, "", py::arg("ctx"));
    m.def("llama_n_threads_batch", (uint32_t (*)(const struct llama_context *)) &llama_n_threads_batch, "", py::arg("ctx"));

    m.def("llama_set_embeddings", (void (*)(const struct llama_context *, bool)) &llama_set_embeddings, "", py::arg("ctx"), py::arg("embeddings"));

    m.def("llama_set_causal_attn", (void (*)(const struct llama_context *, bool)) &llama_set_causal_attn, "", py::arg("ctx"), py::arg("causal_attn"));

    // void llama_set_abort_callback(struct llama_context* ctx, ggml_abort_callback abort_callback, void* abort_callback_data);

    m.def("llama_synchronize", (void (*)(const struct llama_context *)) &llama_synchronize, "", py::arg("ctx"));
    
    m.def("llama_get_logits", (float* (*)(const struct llama_context *)) &llama_get_logits, R"pbdoc(
        Token logits obtained from the last call to llama_decode()

        The logits for which llama_batch.logits[i] != 0 are stored contiguously 
        in the order they have appeared in the batch.

        Rows: number of tokens for which llama_batch.logits[i] != 0
        Cols: n_vocab

        Rows: n_tokens provided with llama_batch
        Cols: n_vocab

        Returns:
            Pointer to the logits buffer of shape (n_tokens, n_vocab)
    )pbdoc", py::arg("ctx"));

    // m.def("llama_get_logits_vec", [](struct llama_context * ctx, size_t n_tokens, size_t n_vocab) -> py::array_t<float> {
    //     float * logits = llama_get_logits(ctx);
    //     constexpr size_t elem_size = sizeof(float);
    //     size_t shape[2]{n_tokens, n_vocab};
    //     size_t strides[2]{n_tokens * elem_size, elem_size};
    //     auto arr = py::array_t<float>(shape, strides);
    //     auto view = arr.mutable_unchecked<2>();
    //     for(size_t i = 0; i < arr.shape(0); i++) {
    //         for(size_t j = 0; j < arr.shape(1); j++) {
    //             view(i,j) = logits[i];
    //         }
    //     }
    //     return arr;
    // }, R"pbdoc(
    //     Token logits obtained from the last call to llama_decode()

    //     The logits for which llama_batch.logits[i] != 0 are stored contiguously 
    //     in the order they have appeared in the batch.

    //     Rows: number of tokens for which llama_batch.logits[i] != 0
    //     Cols: n_vocab
    // )pbdoc", py::arg("ctx"), py::arg("n_tokens"), py::arg("n_vocab"));

    // m.def("llama_get_logits_vec", [](struct llama_context * ctx, int32_t n_tokens) -> py::array_t<float> {
    //     float * logits = llama_get_logits(ctx);
    //     constexpr size_t elem_size = sizeof(float);
    //     size_t shape[1]{static_cast<size_t>(n_tokens),};
    //     size_t strides[1]{static_cast<size_t>(n_tokens) * elem_size,};
    //     auto arr = py::array_t<float>(shape, strides);
    //     auto view = arr.mutable_unchecked<1>();
    //     for(size_t i = 0; i < arr.shape(0); i++) {
    //         view(i) = logits[i];
    //     }
    //     return arr;
    // }, R"pbdoc(
    //     Token logits obtained from the last call to llama_decode()

    //     The logits for which llama_batch.logits[i] != 0 are stored contiguously 
    //     in the order they have appeared in the batch.

    //     Rows: number of tokens for which llama_batch.logits[i] != 0
    //     Cols: n_vocab
    // )pbdoc", py::arg("ctx"), py::arg("n_tokens"));

    // m.def("llama_get_logits", [](struct llama_context * ctx) -> py::array {
    //     auto logits = llama_get_logits(ctx);
    //     return wrap_array_ptr(logits);
    // }, R"pbdoc(
    //     Token logits obtained from the last call to llama_decode()

    //     The logits for which llama_batch.logits[i] != 0 are stored contiguously 
    //     in the order they have appeared in the batch.

    //     Rows: number of tokens for which llama_batch.logits[i] != 0
    //     Cols: n_vocab
    // )pbdoc", py::arg("ctx"));

    // m.def("llama_get_logits", [](struct llama_context * ctx) -> std::vector<float> {
    //     const struct llama_model * model = llama_get_model(ctx);
    //     int32_t n_vocab = llama_n_vocab(model);
    //     float * logits = llama_get_logits(ctx);
    //     std::vector<float> result(logits, logits + n_vocab);
    //     return result;
    // }, R"pbdoc(
    //     Token logits obtained from the last call to llama_decode()

    //     The logits for which llama_batch.logits[i] != 0 are stored contiguously 
    //     in the order they have appeared in the batch.

    //     Rows: number of tokens for which llama_batch.logits[i] != 0
    //     Cols: n_vocab
    // )pbdoc", py::arg("ctx"));

    m.def("llama_get_logits_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_logits_ith, R"pbdoc(
        Logits for the ith token. 

        For positive indices, Equivalent to:

            llama_get_logits(ctx) + ctx->output_ids[i] * n_vocab
        
        Negative indicies can be used to access logits in reverse order, -1 is the last logit.
        returns NULL for invalid ids.
    )pbdoc", py::arg("ctx"), py::arg("i"));

    // m.def("llama_get_logits_ith", [](struct llama_context * ctx, int32_t i) -> std::vector<float> {
    //      const struct llama_model * model = llama_get_model(ctx);
    //     int32_t n_vocab = llama_n_vocab(model);
    //     float * logits = llama_get_logits_ith(ctx, i);
    //     std::vector<float> result(logits, logits + n_vocab);
    //     return result;
    // }, R"pbdoc(
    //     Logits for the ith token. 

    //     For positive indices, Equivalent to:

    //         llama_get_logits(ctx) + ctx->output_ids[i] * n_vocab
        
    //     Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    //     returns NULL for invalid ids.
    // )pbdoc", py::arg("ctx"), py::arg("i"));

    m.def("llama_get_embeddings", (float* (*)(const struct llama_context *)) &llama_get_embeddings, R"pbdoc(
        Get all output token embeddings.

        when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
        the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
        in the order they have appeared in the batch.
        shape: [n_outputs*n_embd]

        Otherwise, returns NULL.
    )pbdoc", py::arg("ctx"));


    m.def("llama_get_embeddings_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_embeddings_ith, R"pbdoc(
        Get the embeddings for the ith token. For positive indices, Equivalent to:

        llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
        Negative indicies can be used to access embeddings in reverse order, -1 is th=
    )pbdoc", py::arg("ctx"), py::arg("seq_id"));

    m.def("llama_token_get_text", (const char* (*)(const struct llama_model *, llama_token)) &llama_token_get_text, "", py::arg("model"), py::arg("token"));
    m.def("llama_token_get_score", (float (*)(const struct llama_model *, llama_token)) &llama_token_get_score, "", py::arg("model"), py::arg("token"));

    m.def("llama_token_get_attr", (enum llama_token_attr (*)(const struct llama_model *, llama_token)) &llama_token_get_attr, "", py::arg("model"), py::arg("token"));
    m.def("llama_token_is_eog", (bool (*)(const struct llama_model *, llama_token)) &llama_token_is_eog, "", py::arg("model"), py::arg("token"));
    m.def("llama_token_is_control", (bool (*)(const struct llama_model *, llama_token)) &llama_token_is_control, "", py::arg("model"), py::arg("token"));

    m.def("llama_token_bos", (llama_token (*)(const struct llama_model *)) &llama_token_bos, "", py::arg("model"));
    m.def("llama_token_eos", (llama_token (*)(const struct llama_model *)) &llama_token_eos, "", py::arg("model"));
    m.def("llama_token_cls", (llama_token (*)(const struct llama_model *)) &llama_token_cls, "", py::arg("model"));
    m.def("llama_token_sep", (llama_token (*)(const struct llama_model *)) &llama_token_sep, "", py::arg("model"));
    m.def("llama_token_nl",  (llama_token (*)(const struct llama_model *)) &llama_token_nl,  "", py::arg("model"));
    m.def("llama_token_pad", (llama_token (*)(const struct llama_model *)) &llama_token_pad, "", py::arg("model"));

    m.def("llama_add_bos_token", (int32_t (*)(const struct llama_model *)) &llama_add_bos_token, "", py::arg("model"));
    m.def("llama_add_eos_token", (int32_t (*)(const struct llama_model *)) &llama_add_eos_token, "", py::arg("model"));

    m.def("llama_token_prefix", (llama_token (*)(const struct llama_model *)) &llama_token_prefix, "", py::arg("model"));
    m.def("llama_token_middle", (llama_token (*)(const struct llama_model *)) &llama_token_middle, "", py::arg("model"));
    m.def("llama_token_suffix", (llama_token (*)(const struct llama_model *)) &llama_token_suffix, "", py::arg("model"));
    m.def("llama_token_eot", (llama_token (*)(const struct llama_model *)) &llama_token_eot, "", py::arg("model"));

    m.def("llama_tokenize", (int32_t (*)(const struct llama_model *, const char*, int32_t, llama_token*, int32_t, bool, bool)) &llama_tokenize, "", py::arg("model"), py::arg("text"), py::arg("text_len"), py::arg("tokens"), py::arg("n_tokens_max"), py::arg("add_special"), py::arg("parse_special"), py::return_value_policy::reference_internal);
    m.def("llama_token_to_piece", (int32_t (*)(const struct llama_model *, llama_token, char*, int32_t, int32_t, bool)) &llama_token_to_piece, "", py::arg("model"), py::arg("token"), py::arg("buf"), py::arg("length"), py::arg("lstrip"), py::arg("special"));
    m.def("llama_detokenize", (int32_t (*)(const struct llama_model *, const llama_token*, int32_t, char*, int32_t, bool, bool)) &llama_detokenize, "", py::arg("model"), py::arg("tokens"), py::arg("n_tokens"), py::arg("text"), py::arg("text_len_max"), py::arg("remove_special"), py::arg("unparse_special"));

    m.def("llama_chat_apply_template", (int32_t (*)(const struct llama_model *, const char*, const struct llama_chat_message*, size_t, bool, char*, int32_t)) &llama_chat_apply_template, "", py::arg("model"), py::arg("tmpl"), py::arg("chat"), py::arg("n_msg"), py::arg("add_ass"), py::arg("buf"), py::arg("length"));

    py::class_<llama_sampler_i, std::shared_ptr<llama_sampler_i>> (m, "llama_sampler_i", "")
        .def( py::init( [](){ return new llama_sampler_i(); } ) );
        // .def_readwrite("name", &llama_sampler_i::name)
        // .def_readwrite("accept", &llama_sampler_i::accept);
        // const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        // void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        // void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        // void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        // struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        // void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

    py::class_<llama_sampler, std::shared_ptr<llama_sampler>> (m, "llama_sampler", "")
        .def( py::init( [](){ return new llama_sampler(); } ) )
        .def_readwrite("iface", &llama_sampler::iface)
        .def_readwrite("ctx", &llama_sampler::ctx);

    m.def("llama_sampler_name", (const char * (*)(const struct llama_sampler *)) &llama_sampler_name, "", py::arg("smpl"));
    m.def("llama_sampler_accept", (void (*)(struct llama_sampler *, llama_token)) &llama_sampler_accept, "", py::arg("smpl"),  py::arg("token"));
    m.def("llama_sampler_apply", (void (*)(struct llama_sampler *, llama_token_data_array *)) &llama_sampler_apply, "", py::arg("smpl"),  py::arg("cur_p"));
    m.def("llama_sampler_reset", (void (*)(struct llama_sampler *)) &llama_sampler_apply, "", py::arg("smpl"));
    m.def("llama_sampler_clone", (struct llama_sampler * (*)(const struct llama_sampler *)) &llama_sampler_clone, "", py::arg("smpl"));
    m.def("llama_sampler_free", (void (*)(const struct llama_sampler *)) &llama_sampler_free, "", py::arg("smpl"));
    m.def("llama_sampler_chain_init", (struct llama_sampler *  (*)(struct llama_sampler_chain_params)) &llama_sampler_chain_init, "", py::arg("params"));
    m.def("llama_sampler_chain_add", (void (*)(struct llama_sampler *, struct llama_sampler *)) &llama_sampler_chain_add, "", py::arg("chain"), py::arg("smpl"));
    m.def("llama_sampler_chain_get", (struct llama_sampler * (*)(const struct llama_sampler *, int32_t)) &llama_sampler_chain_get, "", py::arg("chain"), py::arg("i"));
    m.def("llama_sampler_chain_n", (int (*)(const struct llama_sampler *)) &llama_sampler_chain_n, "", py::arg("chain"));
    m.def("llama_sampler_init_greedy", (struct llama_sampler * (*)(void)) &llama_sampler_init_greedy);
    m.def("llama_sampler_init_dist", (struct llama_sampler * (*)(uint32_t)) &llama_sampler_init_dist, "", py::arg("seed"));

    m.def("llama_sampler_init_softmax", (struct llama_sampler * (*)(void)) &llama_sampler_init_softmax);
    m.def("llama_sampler_init_top_k", (struct llama_sampler * (*)(uint32_t)) &llama_sampler_init_top_k, "", py::arg("k"));
    m.def("llama_sampler_init_top_p", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_top_p, "", py::arg("p"), py::arg("min_keep"));
    m.def("llama_sampler_init_min_p", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_min_p, "", py::arg("p"), py::arg("min_keep"));
    m.def("llama_sampler_init_tail_free", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_tail_free, "", py::arg("z"), py::arg("min_keep"));

    m.def("llama_sampler_init_typical", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_typical, "", py::arg("p"), py::arg("min_keep"));
    m.def("llama_sampler_init_temp", (struct llama_sampler * (*)(float)) &llama_sampler_init_temp, "", py::arg("t"));

    m.def("llama_sampler_init_temp_ext", (struct llama_sampler * (*)(float, float, float)) &llama_sampler_init_temp_ext, "", py::arg("t"), py::arg("delta"), py::arg("exponent"));

    m.def("llama_sampler_init_mirostat", (struct llama_sampler * (*)(int32_t, uint32_t, float, float, int32_t)) &llama_sampler_init_mirostat, "", py::arg("n_vocab"), py::arg("seed"), py::arg("tau"), py::arg("eta"), py::arg("m"));

    m.def("llama_sampler_init_mirostat_v2", (struct llama_sampler * (*)(uint32_t, float, float)) &llama_sampler_init_mirostat_v2, "", py::arg("seed"), py::arg("tau"), py::arg("eta"));

    m.def("llama_sampler_init_grammar", (struct llama_sampler * (*)(const struct llama_model *, const char *, const char *)) &llama_sampler_init_grammar, "", py::arg("model"), py::arg("grammar_str"), py::arg("grammar_root"));

    m.def("llama_sampler_init_penalties", (struct llama_sampler * (*)(int32_t, llama_token, llama_token, int32_t, float, float, float, bool, bool)) &llama_sampler_init_penalties, "", py::arg("n_vocab"), py::arg("special_eos_id"), py::arg("linefeed_id"), py::arg("penalty_last_n"), py::arg("penalty_repeat"), py::arg("epenalty_freq"), py::arg("penalty_present"), py::arg("penalize_nl"), py::arg("ignore_eos"));

    m.def("llama_sampler_init_logit_bias", (struct llama_sampler * (*)(int32_t, int32_t, const llama_logit_bias *)) &llama_sampler_init_logit_bias, "", py::arg("n_vocab"), py::arg("n_logit_bias"), py::arg("logit_bias"));

    m.def("llama_sampler_sample", (llama_token (*)(struct llama_sampler *, struct llama_context *, int32_t)) &llama_sampler_sample, "", py::arg("smpl"), py::arg("ctx"), py::arg("idx"));

    m.def("llama_split_path", (int (*)(char *, int, const char *, int, int)) &llama_split_path, "Build a split GGUF final path for this chunk.", py::arg("split_path"), py::arg("maxlen"), py::arg("path_prefix"), py::arg("split_no"), py::arg("split_count"));
    m.def("llama_split_prefix", (int (*)(char *, int, const char *, int, int)) &llama_split_prefix, "Extract the path prefix from the split_path if and only if the split_no and split_count match.", py::arg("split_prefix"), py::arg("maxlen"), py::arg("split_path"), py::arg("split_no"), py::arg("split_count"));

    // Print system information
    m.def("llama_print_system_info", (const char * (*)(void)) &llama_print_system_info, "");

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);

    // -----------------------------------------------------------------------
    // common.h

    py::class_<gpt_params, std::shared_ptr<gpt_params>> (m, "gpt_params", "")
        .def( py::init( [](){ return new gpt_params(); } ) )
        .def( py::init( [](gpt_params const &o){ return new gpt_params(o); } ) )
        .def_readwrite("n_predict", &gpt_params::n_predict)
        .def_readwrite("n_ctx", &gpt_params::n_ctx)
        .def_readwrite("n_batch", &gpt_params::n_batch)
        .def_readwrite("n_ubatch", &gpt_params::n_ubatch)
        .def_readwrite("n_keep", &gpt_params::n_keep)
        .def_readwrite("n_draft", &gpt_params::n_draft)
        .def_readwrite("n_chunks", &gpt_params::n_chunks)
        .def_readwrite("n_parallel", &gpt_params::n_parallel)
        .def_readwrite("n_sequences", &gpt_params::n_sequences)
        .def_readwrite("p_split", &gpt_params::p_split)
        .def_readwrite("n_gpu_layers", &gpt_params::n_gpu_layers)
        .def_readwrite("n_gpu_layers_draft", &gpt_params::n_gpu_layers_draft)
        .def_readwrite("main_gpu", &gpt_params::main_gpu)
        .def_readwrite("grp_attn_n", &gpt_params::grp_attn_n)
        .def_readwrite("grp_attn_w", &gpt_params::grp_attn_w)
        .def_readwrite("n_print", &gpt_params::n_print)
        .def_readwrite("rope_freq_base", &gpt_params::rope_freq_base)
        .def_readwrite("rope_freq_scale", &gpt_params::rope_freq_scale)
        .def_readwrite("yarn_ext_factor", &gpt_params::yarn_ext_factor)
        .def_readwrite("yarn_attn_factor", &gpt_params::yarn_attn_factor)
        .def_readwrite("yarn_beta_fast", &gpt_params::yarn_beta_fast)
        .def_readwrite("yarn_beta_slow", &gpt_params::yarn_beta_slow)
        .def_readwrite("yarn_orig_ctx", &gpt_params::yarn_orig_ctx)
        .def_readwrite("defrag_thold", &gpt_params::defrag_thold)
        .def_readwrite("numa", &gpt_params::numa)
        .def_readwrite("split_mode", &gpt_params::split_mode)
        .def_readwrite("rope_scaling_type", &gpt_params::rope_scaling_type)
        .def_readwrite("pooling_type", &gpt_params::pooling_type)
        .def_readwrite("attention_type", &gpt_params::attention_type)
        .def_readwrite("sparams", &gpt_params::sparams)
        .def_readwrite("model", &gpt_params::model)
        .def_readwrite("model_draft", &gpt_params::model_draft)
        .def_readwrite("model_alias", &gpt_params::model_alias)
        .def_readwrite("model_url", &gpt_params::model_url)
        .def_readwrite("hf_token", &gpt_params::hf_token)
        .def_readwrite("hf_repo", &gpt_params::hf_repo)
        .def_readwrite("hf_file", &gpt_params::hf_file)
        .def_readwrite("prompt", &gpt_params::prompt)
        .def_readwrite("prompt_file", &gpt_params::prompt_file)
        .def_readwrite("path_prompt_cache", &gpt_params::path_prompt_cache)
        .def_readwrite("input_prefix", &gpt_params::input_prefix)
        .def_readwrite("input_suffix", &gpt_params::input_suffix)
        .def_readwrite("logdir", &gpt_params::logdir)
        .def_readwrite("lookup_cache_static", &gpt_params::lookup_cache_static)
        .def_readwrite("lookup_cache_dynamic", &gpt_params::lookup_cache_dynamic)
        .def_readwrite("logits_file", &gpt_params::logits_file)
        .def_readwrite("rpc_servers", &gpt_params::rpc_servers)
        .def_readwrite("in_files", &gpt_params::in_files)
        .def_readwrite("antiprompt", &gpt_params::antiprompt)
        .def_readwrite("kv_overrides", &gpt_params::kv_overrides)
        .def_readwrite("lora_adapters", &gpt_params::lora_adapters)
        .def_readwrite("control_vectors", &gpt_params::control_vectors)
        .def_readwrite("verbosity", &gpt_params::verbosity)
        .def_readwrite("control_vector_layer_start", &gpt_params::control_vector_layer_start)
        .def_readwrite("control_vector_layer_end", &gpt_params::control_vector_layer_end)
        .def_readwrite("ppl_stride", &gpt_params::ppl_stride)
        .def_readwrite("ppl_output_type", &gpt_params::ppl_output_type)
        .def_readwrite("hellaswag", &gpt_params::hellaswag)
        .def_readwrite("hellaswag_tasks", &gpt_params::hellaswag_tasks)
        .def_readwrite("winogrande", &gpt_params::winogrande)
        .def_readwrite("winogrande_tasks", &gpt_params::winogrande_tasks)
        .def_readwrite("multiple_choice", &gpt_params::multiple_choice)
        .def_readwrite("multiple_choice_tasks", &gpt_params::multiple_choice_tasks)
        .def_readwrite("kl_divergence", &gpt_params::kl_divergence)
        .def_readwrite("usage", &gpt_params::usage)
        .def_readwrite("use_color", &gpt_params::use_color)
        .def_readwrite("special", &gpt_params::special)
        .def_readwrite("interactive", &gpt_params::interactive)
        .def_readwrite("interactive_first", &gpt_params::interactive_first)
        .def_readwrite("conversation", &gpt_params::conversation)
        .def_readwrite("prompt_cache_all", &gpt_params::prompt_cache_all)
        .def_readwrite("prompt_cache_ro", &gpt_params::prompt_cache_ro)
        .def_readwrite("escape", &gpt_params::escape)
        .def_readwrite("multiline_input", &gpt_params::multiline_input)
        .def_readwrite("simple_io", &gpt_params::simple_io)
        .def_readwrite("cont_batching", &gpt_params::cont_batching)
        .def_readwrite("flash_attn", &gpt_params::flash_attn)
        .def_readwrite("input_prefix_bos", &gpt_params::input_prefix_bos)
        // .def_readwrite("ignore_eos", &gpt_params::ignore_eos)
        .def_readwrite("logits_all", &gpt_params::logits_all)
        .def_readwrite("use_mmap", &gpt_params::use_mmap)
        .def_readwrite("use_mlock", &gpt_params::use_mlock)
        .def_readwrite("verbose_prompt", &gpt_params::verbose_prompt)
        .def_readwrite("display_prompt", &gpt_params::display_prompt)
        .def_readwrite("infill", &gpt_params::infill)
        .def_readwrite("dump_kv_cache", &gpt_params::dump_kv_cache)
        .def_readwrite("no_kv_offload", &gpt_params::no_kv_offload)
        .def_readwrite("warmup", &gpt_params::warmup)
        .def_readwrite("check_tensors", &gpt_params::check_tensors)
        .def_readwrite("cache_type_k", &gpt_params::cache_type_k)
        .def_readwrite("cache_type_v", &gpt_params::cache_type_v)
        .def_readwrite("mmproj", &gpt_params::mmproj)
        .def_readwrite("image", &gpt_params::image)
        .def_readwrite("embedding", &gpt_params::embedding)
        .def_readwrite("embd_normalize", &gpt_params::embd_normalize)
        .def_readwrite("embd_out", &gpt_params::embd_out)
        .def_readwrite("embd_sep", &gpt_params::embd_sep)
        .def_readwrite("port", &gpt_params::port)
        .def_readwrite("timeout_read", &gpt_params::timeout_read)
        .def_readwrite("timeout_write", &gpt_params::timeout_write)
        .def_readwrite("n_threads_http", &gpt_params::n_threads_http)
        .def_readwrite("hostname", &gpt_params::hostname)
        .def_readwrite("public_path", &gpt_params::public_path)
        .def_readwrite("chat_template", &gpt_params::chat_template)
        .def_readwrite("system_prompt", &gpt_params::system_prompt)
        .def_readwrite("enable_chat_template", &gpt_params::enable_chat_template)
        .def_readwrite("api_keys", &gpt_params::api_keys)
        .def_readwrite("ssl_file_key", &gpt_params::ssl_file_key)
        .def_readwrite("ssl_file_cert", &gpt_params::ssl_file_cert)
        .def_readwrite("endpoint_slots", &gpt_params::endpoint_slots)
        .def_readwrite("endpoint_metrics", &gpt_params::endpoint_metrics)
        .def_readwrite("log_json", &gpt_params::log_json)
        .def_readwrite("slot_save_path", &gpt_params::slot_save_path)
        .def_readwrite("slot_prompt_similarity", &gpt_params::slot_prompt_similarity)
        .def_readwrite("is_pp_shared", &gpt_params::is_pp_shared)
        .def_readwrite("n_pp", &gpt_params::n_pp)
        .def_readwrite("n_tg", &gpt_params::n_tg)
        .def_readwrite("n_pl", &gpt_params::n_pl)
        .def_readwrite("context_files", &gpt_params::context_files)
        .def_readwrite("chunk_size", &gpt_params::chunk_size)
        .def_readwrite("chunk_separator", &gpt_params::chunk_separator)
        .def_readwrite("n_junk", &gpt_params::n_junk)
        .def_readwrite("i_pos", &gpt_params::i_pos)
        .def_readwrite("out_file", &gpt_params::out_file)
        .def_readwrite("n_out_freq", &gpt_params::n_out_freq)
        .def_readwrite("n_save_freq", &gpt_params::n_save_freq)
        .def_readwrite("i_chunk", &gpt_params::i_chunk)
        .def_readwrite("process_output", &gpt_params::process_output)
        .def_readwrite("compute_ppl", &gpt_params::compute_ppl)
        .def_readwrite("n_pca_batch", &gpt_params::n_pca_batch)
        .def_readwrite("n_pca_iterations", &gpt_params::n_pca_iterations)
        .def_readwrite("cvector_dimre_method", &gpt_params::cvector_dimre_method)
        .def_readwrite("cvector_outfile", &gpt_params::cvector_outfile)
        .def_readwrite("cvector_positive_file", &gpt_params::cvector_positive_file)
        .def_readwrite("cvector_negative_file", &gpt_params::cvector_negative_file)
        .def_readwrite("spm_infill", &gpt_params::spm_infill)
        .def_readwrite("lora_outfile", &gpt_params::lora_outfile)
        .def("assign", (struct gpt_params & (gpt_params::*)(const struct gpt_params &)) &gpt_params::operator=, "C++: gpt_params::operator=(const struct gpt_params &) --> struct gpt_params &", py::return_value_policy::automatic, py::arg(""));


    m.def("llama_token_to_piece", (std::string (*)(const struct llama_context *, llama_token, bool)) &llama_token_to_piece, "", py::arg("ctx"), py::arg("token"), py::arg("special") = true);

    // overloaded
    // m.def("llama_tokenize", [](const struct llama_context * ctx, const std::string & text, bool add_special, bool parse_special = false) -> std::vector<llama_token> {
    //     std::vector<llama_token> tokens_list;
    //     tokens_list = llama_tokenize(ctx, text, add_special, parse_special);
    //     return tokens_list;
    // });
    m.def("llama_tokenize", (std::vector<llama_token> (*)(const struct llama_context *, const std::string &, bool, bool)) &llama_tokenize, "", py::arg("ctx"), py::arg("text"), py::arg("add_special"), py::arg("parse_special") = false, py::return_value_policy::reference_internal);
    m.def("llama_tokenize", (std::vector<llama_token> (*)(const struct llama_model *, const std::string &, bool, bool)) &llama_tokenize, "", py::arg("model"), py::arg("text"), py::arg("add_special"), py::arg("parse_special") = false, py::return_value_policy::reference_internal);

    // m.def("gpt_params_parse_from_env", (void (*)(struct gpt_params &)) &gpt_params_parse_from_env, "", py::arg("params"));
    // m.def("gpt_params_handle_model_default", (void (*)(struct gpt_params &)) &gpt_params_handle_model_default, "C++: gpt_params_handle_model_default(struct gpt_params &) --> void", py::arg("params"));
    m.def("gpt_params_get_system_info", (std::string (*)(const struct gpt_params &)) &gpt_params_get_system_info, "C++: gpt_params_get_system_info(const struct gpt_params &) --> std::string", py::arg("params"));

    m.def("string_split", (class std::vector<std::string> (*)(std::string, char)) &string_split, "C++: string_split(std::string, char) --> class std::vector<std::string>", py::arg("input"), py::arg("separator"));
    m.def("string_strip", (std::string (*)(const std::string &)) &string_strip, "C++: string_strip(const std::string &) --> std::string", py::arg("str"));
    m.def("string_get_sortable_timestamp", (std::string (*)()) &string_get_sortable_timestamp, "C++: string_get_sortable_timestamp() --> std::string");
    m.def("string_parse_kv_override", (bool (*)(const char *, class std::vector<struct llama_model_kv_override> &)) &string_parse_kv_override, "C++: string_parse_kv_override(const char *, class std::vector<struct llama_model_kv_override> &) --> bool", py::arg("data"), py::arg("overrides"));
    m.def("string_process_escapes", (void (*)(std::string &)) &string_process_escapes, "C++: string_process_escapes(std::string &) --> void", py::arg("input"));

    m.def("fs_validate_filename", (bool (*)(const std::string &)) &fs_validate_filename, "C++: fs_validate_filename(const std::string &) --> bool", py::arg("filename"));
    m.def("fs_create_directory_with_parents", (bool (*)(const std::string &)) &fs_create_directory_with_parents, "C++: fs_create_directory_with_parents(const std::string &) --> bool", py::arg("path"));
    m.def("fs_get_cache_directory", (std::string (*)()) &fs_get_cache_directory, "C++: fs_get_cache_directory() --> std::string");
    m.def("fs_get_cache_file", (std::string (*)(const std::string &)) &fs_get_cache_file, "C++: fs_get_cache_file(const std::string &) --> std::string", py::arg("filename"));

    m.def("llama_init_from_gpt_params", (class std::tuple<struct llama_model *, struct llama_context *> (*)(struct gpt_params &)) &llama_init_from_gpt_params, "C++: llama_init_from_gpt_params(struct gpt_params &) --> class std::tuple<struct llama_model *, struct llama_context *>", py::arg("params"));

    m.def("llama_model_params_from_gpt_params", (struct llama_model_params (*)(const struct gpt_params &)) &llama_model_params_from_gpt_params, "C++: llama_model_params_from_gpt_params(const struct gpt_params &) --> struct llama_model_params", py::arg("params"));

    m.def("llama_context_params_from_gpt_params", (struct llama_context_params (*)(const struct gpt_params &)) &llama_context_params_from_gpt_params, "C++: llama_context_params_from_gpt_params(const struct gpt_params &) --> struct llama_context_params", py::arg("params"));

    m.def("llama_batch_clear", (void (*)(struct llama_batch &)) &llama_batch_clear, "C++: llama_batch_clear(struct llama_batch &) --> void", py::arg("batch"));
    m.def("llama_batch_add", (void (*)(struct llama_batch &, int, int, const class std::vector<int> &, bool)) &llama_batch_add, "C++: llama_batch_add(struct llama_batch &, int, int, const class std::vector<int> &, bool) --> void", py::arg("batch"), py::arg("id"), py::arg("pos"), py::arg("seq_ids"), py::arg("logits"));

    py::class_<llama_chat_msg, std::shared_ptr<llama_chat_msg>> (m, "llama_chat_msg", "")
        .def( py::init( [](){ return new llama_chat_msg(); } ) )
        .def( py::init( [](llama_chat_msg const &o){ return new llama_chat_msg(o); } ) )
        .def_readwrite("role", &llama_chat_msg::role)
        .def_readwrite("content", &llama_chat_msg::content)
        .def("assign", (struct llama_chat_msg & (llama_chat_msg::*)(const struct llama_chat_msg &)) &llama_chat_msg::operator=, "", py::return_value_policy::automatic, py::arg(""));

    m.def("llama_chat_verify_template", (bool (*)(const std::string &)) &llama_chat_verify_template, "C++: llama_chat_verify_template(const std::string &) --> bool", py::arg("tmpl"));

    m.def("llama_kv_cache_dump_view", [](const struct llama_kv_cache_view & a0) -> void { return llama_kv_cache_dump_view(a0); }, "", py::arg("view"));
    m.def("llama_kv_cache_dump_view", (void (*)(const struct llama_kv_cache_view &, int)) &llama_kv_cache_dump_view, "C++: llama_kv_cache_dump_view(const struct llama_kv_cache_view &, int) --> void", py::arg("view"), py::arg("row_size"));

    m.def("llama_kv_cache_dump_view_seqs", [](const struct llama_kv_cache_view & a0) -> void { return llama_kv_cache_dump_view_seqs(a0); }, "", py::arg("view"));
    m.def("llama_kv_cache_dump_view_seqs", (void (*)(const struct llama_kv_cache_view &, int)) &llama_kv_cache_dump_view_seqs, "C++: llama_kv_cache_dump_view_seqs(const struct llama_kv_cache_view &, int) --> void", py::arg("view"), py::arg("row_size"));

    m.def("llama_embd_normalize", [](const float * a0, float * a1, int const & a2) -> void { return llama_embd_normalize(a0, a1, a2); }, "", py::arg("inp"), py::arg("out"), py::arg("n"));
    m.def("llama_embd_normalize", (void (*)(const float *, float *, int, int)) &llama_embd_normalize, "C++: llama_embd_normalize(const float *, float *, int, int) --> void", py::arg("inp"), py::arg("out"), py::arg("n"), py::arg("embd_norm"));

    m.def("llama_embd_similarity_cos", (float (*)(const float *, const float *, int)) &llama_embd_similarity_cos, "C++: llama_embd_similarity_cos(const float *, const float *, int) --> float", py::arg("embd1"), py::arg("embd2"), py::arg("n"));

    py::class_<llama_control_vector_data, std::shared_ptr<llama_control_vector_data>> (m, "llama_control_vector_data", "")
        .def( py::init( [](){ return new llama_control_vector_data(); } ) )
        .def( py::init( [](llama_control_vector_data const &o){ return new llama_control_vector_data(o); } ) )
        .def_readwrite("n_embd", &llama_control_vector_data::n_embd)
        .def_readwrite("data", &llama_control_vector_data::data)
        .def("assign", (struct llama_control_vector_data & (llama_control_vector_data::*)(const struct llama_control_vector_data &)) &llama_control_vector_data::operator=, "", py::return_value_policy::automatic, py::arg(""));

    py::class_<llama_control_vector_load_info, std::shared_ptr<llama_control_vector_load_info>> (m, "llama_control_vector_load_info", "")
        .def( py::init( [](){ return new llama_control_vector_load_info(); } ) )
        .def( py::init( [](llama_control_vector_load_info const &o){ return new llama_control_vector_load_info(o); } ) )
        .def_readwrite("strength", &llama_control_vector_load_info::strength)
        .def_readwrite("fname", &llama_control_vector_load_info::fname)
        .def("assign", (struct llama_control_vector_load_info & (llama_control_vector_load_info::*)(const struct llama_control_vector_load_info &)) &llama_control_vector_load_info::operator=, "C++: llama_control_vector_load_info::operator=(const struct llama_control_vector_load_info &) --> struct llama_control_vector_load_info &", py::return_value_policy::automatic, py::arg(""));

    m.def("llama_control_vector_load", (struct llama_control_vector_data (*)(const class std::vector<struct llama_control_vector_load_info> &)) &llama_control_vector_load, "C++: llama_control_vector_load(const class std::vector<struct llama_control_vector_load_info> &) --> struct llama_control_vector_data", py::arg("load_infos"));

    m.def("yaml_dump_vector_float", (void (*)(struct __sFILE *, const char *, const class std::vector<float> &)) &yaml_dump_vector_float, "C++: yaml_dump_vector_float(struct __sFILE *, const char *, const class std::vector<float> &) --> void", py::arg("stream"), py::arg("prop_name"), py::arg("data"));
    m.def("yaml_dump_vector_int", (void (*)(struct __sFILE *, const char *, const class std::vector<int> &)) &yaml_dump_vector_int, "C++: yaml_dump_vector_int(struct __sFILE *, const char *, const class std::vector<int> &) --> void", py::arg("stream"), py::arg("prop_name"), py::arg("data"));
    m.def("yaml_dump_string_multiline", (void (*)(struct __sFILE *, const char *, const char *)) &yaml_dump_string_multiline, "C++: yaml_dump_string_multiline(struct __sFILE *, const char *, const char *) --> void", py::arg("stream"), py::arg("prop_name"), py::arg("data"));

}