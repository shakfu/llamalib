#include <pybind11/pybind11.h>

#include <llama.h>

namespace py = pybind11;


// typedef struct llama_model* t_model;


PYBIND11_MODULE(pbllama, m) {
    m.doc() = "pybind11 pyllama wrapper"; // optional module docstring

    struct llama_model {};
    struct llama_context {};

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

// void llama_free(struct llama_context* ctx);

    m.def("llama_time_us", (int (*)()) &llama_time_us, "C++: llama_time_us() --> int");

    m.def("llama_max_devices", (int (*)()) &llama_max_devices, "C++: llama_max_devices() --> int");

    m.def("llama_supports_mmap", (bool (*)()) &llama_supports_mmap, "C++: llama_supports_mmap() --> bool");

    m.def("llama_supports_mlock", (bool (*)()) &llama_supports_mlock, "C++: llama_supports_mlock() --> bool");

    m.def("llama_supports_gpu_offload", (bool (*)()) &llama_supports_gpu_offload, "C++: llama_supports_gpu_offload() --> bool");

    m.def("llama_model_quantize", (int (*)(const char *, const char *, const struct llama_model_quantize_params *)) &llama_model_quantize, "C++: llama_model_quantize(const char *, const char *, const struct llama_model_quantize_params *) --> int", pybind11::arg("fname_inp"), pybind11::arg("fname_out"), pybind11::arg("params"));

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


    m.def("llama_kv_cache_view_free", (void (*)(struct llama_kv_cache_view *)) &llama_kv_cache_view_free, "C++: llama_kv_cache_view_free(struct llama_kv_cache_view *) --> void", py::arg("view"));

    m.def("llama_batch_get_one", (struct llama_batch (*)(int *, int, int, int)) &llama_batch_get_one, "C++: llama_batch_get_one(int *, int, int, int) --> struct llama_batch", py::arg("tokens"), py::arg("n_tokens"), py::arg("pos_0"), py::arg("seq_id"));

    m.def("llama_batch_init", (struct llama_batch (*)(int, int, int)) &llama_batch_init, "C++: llama_batch_init(int, int, int) --> struct llama_batch", py::arg("n_tokens"), py::arg("embd"), py::arg("n_seq_max"));

    m.def("llama_batch_free", (void (*)(struct llama_batch)) &llama_batch_free, "C++: llama_batch_free(struct llama_batch) --> void", py::arg("batch"));

    m.def("llama_split_path", (int (*)(char *, int, const char *, int, int)) &llama_split_path, "Build a split GGUF final path for this chunk.", py::arg("split_path"), py::arg("maxlen"), py::arg("path_prefix"), py::arg("split_no"), py::arg("split_count"));

    m.def("llama_split_prefix", (int (*)(char *, int, const char *, int, int)) &llama_split_prefix, "Extract the path prefix from the split_path if and only if the split_no and split_count match.", py::arg("split_prefix"), py::arg("maxlen"), py::arg("split_path"), py::arg("split_no"), py::arg("split_count"));

    m.def("llama_print_system_info", (const char * (*)()) &llama_print_system_info, "C++: llama_print_system_info() --> const char *", py::return_value_policy::automatic);

}