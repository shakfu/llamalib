#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <llama.h>
#include <ggml.h>

// #include <vector>

namespace py = pybind11;


PYBIND11_MODULE(pbllama, m) {
    m.doc() = "pbllama: pybind11 llama.cpp wrapper"; // optional module docstring

    struct llama_model {};
    struct llama_context {};
    struct llama_grammar {};

    // m.def("backend_init", &llama_backend_init, "Initialize the llama + ggml backend.\n\nCall once at the start of a program.");
    // m.def("backend_free", &llama_backend_free, "Call once at the end of the program - currently only used for MPI.");

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
        .export_values();

    py::enum_<enum llama_rope_type>(m, "llama_rope_type", py::arithmetic(), "")
        .value("LLAMA_ROPE_TYPE_NONE", LLAMA_ROPE_TYPE_NONE)
        .value("LLAMA_ROPE_TYPE_NORM", LLAMA_ROPE_TYPE_NORM)
        .value("LLAMA_ROPE_TYPE_NEOX", LLAMA_ROPE_TYPE_NEOX)
        .value("LLAMA_ROPE_TYPE_GLM", LLAMA_ROPE_TYPE_GLM)
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
        .def_readwrite("sorted", &llama_token_data_array::sorted);

    py::class_<llama_batch, std::shared_ptr<llama_batch>> (m, "llama_batch", "")
        .def( py::init( [](){ return new llama_batch(); } ) )
        .def_readwrite("n_tokens", &llama_batch::n_tokens)
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
        .def_readwrite("seed", &llama_context_params::seed)
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


    py::enum_<enum llama_gretype>(m, "llama_gretype", py::arithmetic(), "")
        .value("LLAMA_GRETYPE_END", LLAMA_GRETYPE_END)
        .value("LLAMA_GRETYPE_ALT", LLAMA_GRETYPE_ALT)
        .value("LLAMA_GRETYPE_RULE_REF", LLAMA_GRETYPE_RULE_REF)
        .value("LLAMA_GRETYPE_CHAR", LLAMA_GRETYPE_CHAR)
        .value("LLAMA_GRETYPE_CHAR_NOT", LLAMA_GRETYPE_CHAR_NOT)
        .value("LLAMA_GRETYPE_CHAR_RNG_UPPER", LLAMA_GRETYPE_CHAR_RNG_UPPER)
        .value("LLAMA_GRETYPE_CHAR_ALT", LLAMA_GRETYPE_CHAR_ALT)
        .value("LLAMA_GRETYPE_CHAR_ANY", LLAMA_GRETYPE_CHAR_ANY)
        .export_values();


    py::class_<llama_grammar_element, std::shared_ptr<llama_grammar_element>> (m, "llama_grammar_element", "")
        .def( py::init( [](){ return new llama_grammar_element(); } ) )
        .def_readwrite("type", &llama_grammar_element::type)
        .def_readwrite("value", &llama_grammar_element::value);

    py::class_<llama_timings, std::shared_ptr<llama_timings>> (m, "llama_timings", "")
        .def( py::init( [](){ return new llama_timings(); } ) )
        .def_readwrite("t_start_ms", &llama_timings::t_start_ms)
        .def_readwrite("t_end_ms", &llama_timings::t_end_ms)
        .def_readwrite("t_load_ms", &llama_timings::t_load_ms)
        .def_readwrite("t_sample_ms", &llama_timings::t_sample_ms)
        .def_readwrite("t_p_eval_ms", &llama_timings::t_p_eval_ms)
        .def_readwrite("t_eval_ms", &llama_timings::t_eval_ms)
        .def_readwrite("n_sample", &llama_timings::n_sample)
        .def_readwrite("n_p_eval", &llama_timings::n_p_eval)
        .def_readwrite("n_eval", &llama_timings::n_eval);

    py::class_<llama_chat_message, std::shared_ptr<llama_chat_message>> (m, "llama_chat_message", "")
        .def( py::init( [](){ return new llama_chat_message(); } ) )
        .def_readwrite("role", &llama_chat_message::role)
        .def_readwrite("content", &llama_chat_message::content);


    m.def("llama_model_default_params", (struct llama_model_params (*)()) &llama_model_default_params, "C++: llama_model_default_params() --> struct llama_model_params");
    m.def("llama_context_default_params", (struct llama_context_params (*)()) &llama_context_default_params, "C++: llama_context_default_params() --> struct llama_context_params");
    m.def("llama_model_quantize_default_params", (struct llama_model_quantize_params (*)()) &llama_model_quantize_default_params, "C++: llama_model_quantize_default_params() --> struct llama_model_quantize_params");
    m.def("llama_backend_init", (void (*)()) &llama_backend_init, "C++: llama_backend_init() --> void");
    m.def("llama_numa_init", (void (*)(enum ggml_numa_strategy)) &llama_numa_init, "C++: llama_numa_init(enum ggml_numa_strategy) --> void", py::arg("numa"));
    m.def("llama_backend_free", (void (*)()) &llama_backend_free, "C++: llama_backend_free() --> void");
    m.def("llama_load_model_from_file", (struct llama_model (*)(const char *, struct llama_model_params *)) &llama_load_model_from_file, "Load a model from file", py::arg("path_model"), py::arg("params"));
    m.def("llama_free_model", (void (*)(struct llama_model *)) &llama_free_model, "Free a model", py::arg("model"));
    m.def("llama_new_context_with_model", (struct llama_context (*)(struct llama_model *, struct llama_context_params)) &llama_new_context_with_model, "New context with model", py::arg("model"), py::arg("params"));
    m.def("llama_free", (void (*)(struct llama_context *)) &llama_free, "Free context", py::arg("ctx"));
    m.def("llama_time_us", (int64_t (*)()) &llama_time_us, "C++: llama_time_us() --> int");
    m.def("llama_max_devices", (size_t (*)()) &llama_max_devices, "C++: llama_max_devices() --> int");
    m.def("llama_supports_mmap", (bool (*)()) &llama_supports_mmap, "C++: llama_supports_mmap() --> bool");
    m.def("llama_supports_mlock", (bool (*)()) &llama_supports_mlock, "C++: llama_supports_mlock() --> bool");
    m.def("llama_supports_gpu_offload", (bool (*)()) &llama_supports_gpu_offload, "C++: llama_supports_gpu_offload() --> bool");
    m.def("llama_get_model", (const struct llama_model (*)(const struct llama_context *)) &llama_get_model, "get model from context", py::arg("ctx"));

    m.def("llama_n_ctx", (uint32_t (*)(const struct llama_context *)) &llama_n_ctx, "get n_ctx from context", py::arg("ctx"));
    m.def("llama_n_batch", (uint32_t (*)(const struct llama_context *)) &llama_n_batch, "get n_batch from context", py::arg("ctx"));
    m.def("llama_n_ubatch", (uint32_t (*)(const struct llama_context *)) &llama_n_ubatch, "get n_ubatch from context", py::arg("ctx"));
    m.def("llama_n_seq_max", (uint32_t (*)(const struct llama_context *)) &llama_n_seq_max, "get n_seq_max from context", py::arg("ctx"));

    m.def("get_llama_pooling_type", (enum llama_pooling_type (*)(const struct llama_context *)) &llama_pooling_type, "get pooling_type from context", py::arg("ctx"));

    m.def("get_llama_vocab_type", (enum llama_vocab_type (*)(const struct llama_model *)) &llama_vocab_type, "get vocab_type from model", py::arg("model"));
    m.def("get_llama_rope_type", (enum llama_rope_type (*)(const struct llama_model *)) &llama_rope_type, "get rope_type from model", py::arg("model"));

    m.def("llama_n_vocab", (int32_t (*)(const struct llama_model *)) &llama_n_vocab, "get n_vocab from model", py::arg("model"));
    m.def("llama_n_ctx_train", (int32_t (*)(const struct llama_model *)) &llama_n_ctx_train, "get n_ctx_train from model", py::arg("model"));
    m.def("llama_n_embd", (int32_t (*)(const struct llama_model *)) &llama_n_embd, "get n_embed from model", py::arg("model"));
    m.def("llama_n_layer", (int32_t (*)(const struct llama_model *)) &llama_n_layer, "get n_layer from model", py::arg("model"));

    m.def("llama_rope_freq_scale_train", (float (*)(const struct llama_model *)) &llama_rope_freq_scale_train, "get rope_freq_scale_train from model", py::arg("model"));

    m.def("llama_model_meta_val_str", (int32_t (*)(const struct llama_model *, const char *, char *, size_t)) &llama_model_meta_val_str, "get meta_val_str from model", py::arg("model"), py::arg("key"), py::arg("buf"), py::arg("buf_size"));
    m.def("llama_model_meta_count", (int32_t (*)(const struct llama_model *)) &llama_model_meta_count, "get meta_count from model", py::arg("model"));
    m.def("llama_model_meta_key_by_index", (int32_t (*)(const struct llama_model *, int32_t, char *, size_t)) &llama_model_meta_key_by_index, "get meta_key_by_index from model", py::arg("model"), py::arg("i"), py::arg("buf"), py::arg("buf_size"));
    m.def("llama_model_meta_val_str_by_index", (int32_t (*)(const struct llama_model *, int32_t, char *, size_t)) &llama_model_meta_val_str_by_index, "get meta_val_str_by_index from model", py::arg("model"), py::arg("i"), py::arg("buf"), py::arg("buf_size"));
    m.def("llama_model_desc", (int32_t (*)(const struct llama_model *, char *, size_t)) &llama_model_desc, "get model_desc from model", py::arg("model"), py::arg("buf"), py::arg("buf_size"));

    m.def("llama_model_size", (uint64_t (*)(const struct llama_model *)) &llama_model_size, "get model_size from model", py::arg("model"));
    m.def("llama_model_n_params", (uint64_t (*)(const struct llama_model *)) &llama_model_n_params, "get model_n_params from model", py::arg("model"));

    m.def("llama_get_model_tensor", (struct ggml_tensor* (*)(struct llama_model *)) &llama_get_model_tensor, "get model tensor from model", py::arg("model"));

    // struct ggml_tensor* llama_get_model_tensor(struct llama_model* model, const char* name);

    m.def("llama_model_has_encoder", (bool (*)(const struct llama_model *)) &llama_model_has_encoder, "model has encoder?", py::arg("model"));

    m.def("llama_model_decoder_start_token", (llama_token (*)(const struct llama_model *)) &llama_model_decoder_start_token, "get decoder_start_token from model", py::arg("model"));

    m.def("llama_model_quantize", (int (*)(const char *, const char *, const struct llama_model_quantize_params *)) &llama_model_quantize, "C++: llama_model_quantize(const char *, const char *, const struct llama_model_quantize_params *) --> int", py::arg("fname_inp"), py::arg("fname_out"), py::arg("params"));

    // m.def("llama_lora_adapter_init", (struct llama_lora_adapter (*)(const struct llama_model *, const char *)) &llama_lora_adapter_init, "", py::arg("model"), py::arg("path_lora"));

    // int32_t llama_lora_adapter_set(struct llama_context* ctx, struct llama_lora_adapter* adapter, float scale);
    // int32_t llama_lora_adapter_remove(struct llama_context* ctx, struct llama_lora_adapter* adapter);
    // void llama_lora_adapter_clear(struct llama_context* ctx);
    // void llama_lora_adapter_free(struct llama_lora_adapter* adapter);

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
    m.def("llama_get_logits", (float* (*)(const struct llama_context *)) &llama_get_logits, "", py::arg("ctx"));
    m.def("llama_get_logits_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_logits_ith, "", py::arg("ctx"), py::arg("i"));

    m.def("llama_get_embeddings", (float* (*)(const struct llama_context *)) &llama_get_embeddings, "", py::arg("ctx"));
    m.def("llama_get_embeddings_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_embeddings_ith, "", py::arg("ctx"), py::arg("i"));
    m.def("llama_get_embeddings_seq", (float* (*)(const struct llama_context *, llama_seq_id)) &llama_get_embeddings_seq, "", py::arg("ctx"), py::arg("seq_id"));

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

    m.def("llama_tokenize", (int32_t (*)(const struct llama_model *, const char*, int32_t, llama_token*, int32_t, bool, bool)) &llama_tokenize, "", py::arg("model"), py::arg("text"), py::arg("text_len"), py::arg("tokens"), py::arg("n_tokens_max"), py::arg("add_special"), py::arg("parse_special"));
    m.def("llama_token_to_piece", (int32_t (*)(const struct llama_model *, llama_token, char*, int32_t, int32_t, bool)) &llama_token_to_piece, "", py::arg("model"), py::arg("token"), py::arg("buf"), py::arg("length"), py::arg("lstrip"), py::arg("special"));
    m.def("llama_detokenize", (int32_t (*)(const struct llama_model *, const llama_token*, int32_t, char*, int32_t, bool, bool)) &llama_detokenize, "", py::arg("model"), py::arg("tokens"), py::arg("n_tokens"), py::arg("text"), py::arg("text_len_max"), py::arg("remove_special"), py::arg("unparse_special"));

    m.def("llama_chat_apply_template", (int32_t (*)(const struct llama_model *, const char*, const struct llama_chat_message*, size_t, bool, char*, int32_t)) &llama_chat_apply_template, "", py::arg("model"), py::arg("tmpl"), py::arg("chat"), py::arg("n_msg"), py::arg("add_ass"), py::arg("buf"), py::arg("length"));

    // struct llama_grammar* llama_grammar_init(const llama_grammar_element** rules, size_t n_rules, size_t start_rule_index);
    m.def("llama_grammar_init", [](std::vector<llama_grammar_element> rules, size_t start_rule_index) -> struct llama_grammar * {
        std::vector<const llama_grammar_element *> elems;
        elems.reserve(rules.size());
        for (auto rule : rules) {
            elems.push_back(&rule);
        }
        return (struct llama_grammar *)llama_grammar_init(elems.data(), elems.size(), start_rule_index);
    });

    m.def("llama_grammar_free", (void (*)(struct llama_grammar *)) &llama_grammar_free, "", py::arg("grammar"));
    m.def("llama_grammar_copy", (struct llama_grammar* (*)(const struct llama_grammar *)) &llama_grammar_copy, "", py::arg("grammar"));
    m.def("llama_grammar_sample", (void (*)(const struct llama_grammar *, const struct llama_context*, llama_token_data_array*)) &llama_grammar_sample, "", py::arg("grammar"), py::arg("ctx"), py::arg("candidates"));
    m.def("llama_grammar_accept_token", (void (*)(const struct llama_grammar *, const struct llama_context*, llama_token)) &llama_grammar_accept_token, "", py::arg("grammar"), py::arg("ctx"), py::arg("token"));

    m.def("llama_set_rng_seed", (void (*)(const struct llama_context *, uint32_t)) &llama_set_rng_seed, "", py::arg("ctx"), py::arg("seed"));

    m.def("llama_sample_repetition_penalties", (void (*)(const struct llama_context *, llama_token_data_array *, const llama_token *, size_t, float, float, float)) &llama_sample_repetition_penalties, "", py::arg("ctx"), py::arg("candidates"), py::arg("last_tokens"), py::arg("penalty_last_n"), py::arg("penalty_repeat"), py::arg("penalty_freq"), py::arg("penalty_present"));
    m.def("llama_sample_apply_guidance", (void (*)(const struct llama_context *, float *, float *, float)) &llama_sample_apply_guidance, "", py::arg("ctx"), py::arg("logits"), py::arg("logits_guidance"), py::arg("scale"));
    m.def("llama_sample_softmax", (void (*)(const struct llama_context *, llama_token_data_array *)) &llama_sample_softmax, "", py::arg("ctx"), py::arg("candidates"));
    m.def("llama_sample_top_k", (void (*)(const struct llama_context *, llama_token_data_array *, int32_t, size_t)) &llama_sample_top_k, "", py::arg("ctx"), py::arg("candidates"), py::arg("k"), py::arg("min_keep"));
    m.def("llama_sample_top_p", (void (*)(const struct llama_context *, llama_token_data_array *, int32_t, size_t)) &llama_sample_top_p, "", py::arg("ctx"), py::arg("candidates"), py::arg("k"), py::arg("min_keep"));
    m.def("llama_sample_min_p", (void (*)(const struct llama_context *, llama_token_data_array *, int32_t, size_t)) &llama_sample_min_p, "", py::arg("ctx"), py::arg("candidates"), py::arg("k"), py::arg("min_keep"));
    m.def("llama_sample_tail_free", (void (*)(const struct llama_context *, llama_token_data_array *, float, size_t)) &llama_sample_tail_free, "", py::arg("ctx"), py::arg("candidates"), py::arg("z"), py::arg("min_keep"));
    m.def("llama_sample_typical", (void (*)(const struct llama_context *, llama_token_data_array *, float, size_t)) &llama_sample_typical, "", py::arg("ctx"), py::arg("candidates"), py::arg("p"), py::arg("min_keep"));
    m.def("llama_sample_entropy", (void (*)(const struct llama_context *, llama_token_data_array *, float, float, float)) &llama_sample_entropy, "", py::arg("ctx"), py::arg("candidates_p"), py::arg("min_temp"), py::arg("max_temp"), py::arg("exponent_val"));
    m.def("llama_sample_temp", (void (*)(const struct llama_context *, llama_token_data_array *, float)) &llama_sample_temp, "", py::arg("ctx"), py::arg("candidates"), py::arg("temp"));
    m.def("llama_sample_token_mirostat", (llama_token (*)(const struct llama_context *, llama_token_data_array *, float, float, int32_t, float*)) &llama_sample_token_mirostat, "", py::arg("ctx"), py::arg("candidates"), py::arg("tau"), py::arg("eta"), py::arg("m"), py::arg("mu"));
    m.def("llama_sample_token_mirostat_v2", (llama_token (*)(const struct llama_context *, llama_token_data_array *, float, float, float*)) &llama_sample_token_mirostat_v2, "", py::arg("ctx"), py::arg("candidates"), py::arg("tau"), py::arg("eta"), py::arg("mu"));
    m.def("llama_sample_token_greedy", (llama_token (*)(const struct llama_context *, llama_token_data_array *)) &llama_sample_token_greedy, "", py::arg("ctx"), py::arg("candidates"));
    m.def("llama_sample_token", (llama_token (*)(const struct llama_context *, llama_token_data_array *)) &llama_sample_token, "", py::arg("ctx"), py::arg("candidates"));

    m.def("llama_split_path", (int (*)(char *, int, const char *, int, int)) &llama_split_path, "Build a split GGUF final path for this chunk.", py::arg("split_path"), py::arg("maxlen"), py::arg("path_prefix"), py::arg("split_no"), py::arg("split_count"));
    m.def("llama_split_prefix", (int (*)(char *, int, const char *, int, int)) &llama_split_prefix, "Extract the path prefix from the split_path if and only if the split_no and split_count match.", py::arg("split_prefix"), py::arg("maxlen"), py::arg("split_path"), py::arg("split_no"), py::arg("split_count"));

    m.def("llama_print_system_info", (const char * (*)()) &llama_print_system_info, "C++: llama_print_system_info() --> const char *", py::return_value_policy::automatic);

}