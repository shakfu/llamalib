#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <llama.h>

#include <memory>

namespace nb = nanobind;


NB_MODULE(nbllama, m) {
    m.doc() = "nanobind pyllama wrapper"; // optional module docstring

    struct llama_model {};
    struct llama_context {};


    nb::enum_<enum llama_vocab_type>(m, "llama_vocab_type")
        .value("LLAMA_VOCAB_TYPE_NONE", LLAMA_VOCAB_TYPE_NONE)
        .value("LLAMA_VOCAB_TYPE_SPM", LLAMA_VOCAB_TYPE_SPM)
        .value("LLAMA_VOCAB_TYPE_BPE", LLAMA_VOCAB_TYPE_BPE)
        .value("LLAMA_VOCAB_TYPE_WPM", LLAMA_VOCAB_TYPE_WPM)
        .value("LLAMA_VOCAB_TYPE_UGM", LLAMA_VOCAB_TYPE_UGM)
        .export_values();

    nb::enum_<enum llama_vocab_pre_type>(m, "llama_vocab_pre_type")
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

    nb::enum_<enum llama_rope_type>(m, "llama_rope_type")
        .value("LLAMA_ROPE_TYPE_NONE", LLAMA_ROPE_TYPE_NONE)
        .value("LLAMA_ROPE_TYPE_NORM", LLAMA_ROPE_TYPE_NORM)
        .value("LLAMA_ROPE_TYPE_NEOX", LLAMA_ROPE_TYPE_NEOX)
        .value("LLAMA_ROPE_TYPE_GLM", LLAMA_ROPE_TYPE_GLM)
        .export_values();

    nb::enum_<enum llama_token_type>(m, "llama_token_type")
        .value("LLAMA_TOKEN_TYPE_UNDEFINED", LLAMA_TOKEN_TYPE_UNDEFINED)
        .value("LLAMA_TOKEN_TYPE_NORMAL", LLAMA_TOKEN_TYPE_NORMAL)
        .value("LLAMA_TOKEN_TYPE_UNKNOWN", LLAMA_TOKEN_TYPE_UNKNOWN)
        .value("LLAMA_TOKEN_TYPE_CONTROL", LLAMA_TOKEN_TYPE_CONTROL)
        .value("LLAMA_TOKEN_TYPE_USER_DEFINED", LLAMA_TOKEN_TYPE_USER_DEFINED)
        .value("LLAMA_TOKEN_TYPE_UNUSED", LLAMA_TOKEN_TYPE_UNUSED)
        .value("LLAMA_TOKEN_TYPE_BYTE", LLAMA_TOKEN_TYPE_BYTE)
        .export_values();

    nb::enum_<enum llama_token_attr>(m, "llama_token_attr")
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

    nb::enum_<enum llama_ftype>(m, "llama_ftype")
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

    nb::enum_<enum llama_rope_scaling_type>(m, "llama_rope_scaling_type")
        .value("LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED", LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED)
        .value("LLAMA_ROPE_SCALING_TYPE_NONE", LLAMA_ROPE_SCALING_TYPE_NONE)
        .value("LLAMA_ROPE_SCALING_TYPE_LINEAR", LLAMA_ROPE_SCALING_TYPE_LINEAR)
        .value("LLAMA_ROPE_SCALING_TYPE_YARN", LLAMA_ROPE_SCALING_TYPE_YARN)
        .value("LLAMA_ROPE_SCALING_TYPE_MAX_VALUE", LLAMA_ROPE_SCALING_TYPE_MAX_VALUE)
        .export_values();

    nb::enum_<enum llama_pooling_type>(m, "llama_pooling_type")
        .value("LLAMA_POOLING_TYPE_UNSPECIFIED", LLAMA_POOLING_TYPE_UNSPECIFIED)
        .value("LLAMA_POOLING_TYPE_NONE", LLAMA_POOLING_TYPE_NONE)
        .value("LLAMA_POOLING_TYPE_MEAN", LLAMA_POOLING_TYPE_MEAN)
        .value("LLAMA_POOLING_TYPE_CLS", LLAMA_POOLING_TYPE_CLS)
        .value("LLAMA_POOLING_TYPE_LAST", LLAMA_POOLING_TYPE_LAST)
        .export_values();

    nb::enum_<enum llama_attention_type>(m, "llama_attention_type")
        .value("LLAMA_ATTENTION_TYPE_UNSPECIFIED", LLAMA_ATTENTION_TYPE_UNSPECIFIED)
        .value("LLAMA_ATTENTION_TYPE_CAUSAL", LLAMA_ATTENTION_TYPE_CAUSAL)
        .value("LLAMA_ATTENTION_TYPE_NON_CAUSAL", LLAMA_ATTENTION_TYPE_NON_CAUSAL)
        .export_values();

    nb::enum_<enum llama_split_mode>(m, "llama_split_mode")
        .value("LLAMA_SPLIT_MODE_NONE", LLAMA_SPLIT_MODE_NONE)
        .value("LLAMA_SPLIT_MODE_LAYER", LLAMA_SPLIT_MODE_LAYER)
        .value("LLAMA_SPLIT_MODE_ROW", LLAMA_SPLIT_MODE_ROW)
        .export_values();


    nb::class_<llama_token_data>(m, "llama_token_data")
        .def(nb::init<>())
        .def(nb::init<llama_token,float,float>())
        .def_rw("id", &llama_token_data::id)         // token id
        .def_rw("logit", &llama_token_data::logit)   // log-odds of the token
        .def_rw("p", &llama_token_data::p);          // probability of the token

    nb::class_<llama_token_data_array> (m, "llama_token_data_array")
        .def(nb::init<>())
        .def_rw("size", &llama_token_data_array::size)
        .def_rw("sorted", &llama_token_data_array::sorted);


    nb::class_<llama_batch> (m, "llama_batch")
        .def(nb::init<>())
        .def_rw("n_tokens", &llama_batch::n_tokens)
        .def_rw("all_pos_0", &llama_batch::all_pos_0)
        .def_rw("all_pos_1", &llama_batch::all_pos_1)
        .def_rw("all_seq_id", &llama_batch::all_seq_id);
    
    nb::enum_<enum llama_model_kv_override_type>(m, "llama_model_kv_override_type")
        .value("LLAMA_KV_OVERRIDE_TYPE_INT", LLAMA_KV_OVERRIDE_TYPE_INT)
        .value("LLAMA_KV_OVERRIDE_TYPE_FLOAT", LLAMA_KV_OVERRIDE_TYPE_FLOAT)
        .value("LLAMA_KV_OVERRIDE_TYPE_BOOL", LLAMA_KV_OVERRIDE_TYPE_BOOL)
        .value("LLAMA_KV_OVERRIDE_TYPE_STR", LLAMA_KV_OVERRIDE_TYPE_STR)
        .export_values();

    nb::class_<llama_model_kv_override> (m, "llama_model_kv_override")
        .def(nb::init<>())
        .def_rw("tag", &llama_model_kv_override::tag);
    
    nb::class_<llama_model_params> (m, "llama_model_params")
        .def(nb::init<>())
        .def_rw("n_gpu_layers", &llama_model_params::n_gpu_layers)
        .def_rw("split_mode", &llama_model_params::split_mode)
        .def_rw("main_gpu", &llama_model_params::main_gpu)
        .def_rw("vocab_only", &llama_model_params::vocab_only)
        .def_rw("use_mmap", &llama_model_params::use_mmap)
        .def_rw("use_mlock", &llama_model_params::use_mlock)
        .def_rw("check_tensors", &llama_model_params::check_tensors);
    
    nb::class_<llama_context_params> (m, "llama_context_params")
        .def(nb::init<>())
        .def_rw("seed", &llama_context_params::seed)
        .def_rw("n_ctx", &llama_context_params::n_ctx)
        .def_rw("n_batch", &llama_context_params::n_batch)
        .def_rw("n_ubatch", &llama_context_params::n_ubatch)
        .def_rw("n_seq_max", &llama_context_params::n_seq_max)
        .def_rw("n_threads", &llama_context_params::n_threads)
        .def_rw("n_threads_batch", &llama_context_params::n_threads_batch)
        .def_rw("rope_scaling_type", &llama_context_params::rope_scaling_type)
        .def_rw("pooling_type", &llama_context_params::pooling_type)
        .def_rw("attention_type", &llama_context_params::attention_type)
        .def_rw("rope_freq_base", &llama_context_params::rope_freq_base)
        .def_rw("rope_freq_scale", &llama_context_params::rope_freq_scale)
        .def_rw("yarn_ext_factor", &llama_context_params::yarn_ext_factor)
        .def_rw("yarn_attn_factor", &llama_context_params::yarn_attn_factor)
        .def_rw("yarn_beta_fast", &llama_context_params::yarn_beta_fast)
        .def_rw("yarn_beta_slow", &llama_context_params::yarn_beta_slow)
        .def_rw("yarn_orig_ctx", &llama_context_params::yarn_orig_ctx)
        .def_rw("defrag_thold", &llama_context_params::defrag_thold)
        .def_rw("type_k", &llama_context_params::type_k)
        .def_rw("type_v", &llama_context_params::type_v)
        .def_rw("logits_all", &llama_context_params::logits_all)
        .def_rw("embeddings", &llama_context_params::embeddings)
        .def_rw("offload_kqv", &llama_context_params::offload_kqv)
        .def_rw("flash_attn", &llama_context_params::flash_attn);

    nb::class_<llama_model_quantize_params> (m, "llama_model_quantize_params")
        .def(nb::init<>())
        .def_rw("nthread", &llama_model_quantize_params::nthread)
        .def_rw("ftype", &llama_model_quantize_params::ftype)
        .def_rw("output_tensor_type", &llama_model_quantize_params::output_tensor_type)
        .def_rw("token_embedding_type", &llama_model_quantize_params::token_embedding_type)
        .def_rw("allow_requantize", &llama_model_quantize_params::allow_requantize)
        .def_rw("quantize_output_tensor", &llama_model_quantize_params::quantize_output_tensor)
        .def_rw("only_copy", &llama_model_quantize_params::only_copy)
        .def_rw("pure", &llama_model_quantize_params::pure);


    nb::enum_<enum llama_gretype>(m, "llama_gretype")
        .value("LLAMA_GRETYPE_END", LLAMA_GRETYPE_END)
        .value("LLAMA_GRETYPE_ALT", LLAMA_GRETYPE_ALT)
        .value("LLAMA_GRETYPE_RULE_REF", LLAMA_GRETYPE_RULE_REF)
        .value("LLAMA_GRETYPE_CHAR", LLAMA_GRETYPE_CHAR)
        .value("LLAMA_GRETYPE_CHAR_NOT", LLAMA_GRETYPE_CHAR_NOT)
        .value("LLAMA_GRETYPE_CHAR_RNG_UPPER", LLAMA_GRETYPE_CHAR_RNG_UPPER)
        .value("LLAMA_GRETYPE_CHAR_ALT", LLAMA_GRETYPE_CHAR_ALT)
        .value("LLAMA_GRETYPE_CHAR_ANY", LLAMA_GRETYPE_CHAR_ANY)
        .export_values();


    nb::class_<llama_grammar_element> (m, "llama_grammar_element")
        .def(nb::init<>())
        .def_rw("type", &llama_grammar_element::type)
        .def_rw("value", &llama_grammar_element::value);

    nb::class_<llama_timings> (m, "llama_timings")
        .def(nb::init<>())
        .def_rw("t_start_ms", &llama_timings::t_start_ms)
        .def_rw("t_end_ms", &llama_timings::t_end_ms)
        .def_rw("t_load_ms", &llama_timings::t_load_ms)
        .def_rw("t_sample_ms", &llama_timings::t_sample_ms)
        .def_rw("t_p_eval_ms", &llama_timings::t_p_eval_ms)
        .def_rw("t_eval_ms", &llama_timings::t_eval_ms)
        .def_rw("n_sample", &llama_timings::n_sample)
        .def_rw("n_p_eval", &llama_timings::n_p_eval)
        .def_rw("n_eval", &llama_timings::n_eval);

    nb::class_<llama_chat_message> (m, "llama_chat_message")
        .def(nb::init<>())
        .def_rw("role", &llama_chat_message::role)
        .def_rw("content", &llama_chat_message::content);


    m.def("llama_model_default_params", (struct llama_model_params (*)()) &llama_model_default_params, "C++: llama_model_default_params() --> struct llama_model_params");
    m.def("llama_context_default_params", (struct llama_context_params (*)()) &llama_context_default_params, "C++: llama_context_default_params() --> struct llama_context_params");
    m.def("llama_model_quantize_default_params", (struct llama_model_quantize_params (*)()) &llama_model_quantize_default_params, "C++: llama_model_quantize_default_params() --> struct llama_model_quantize_params");
    m.def("llama_backend_init", (void (*)()) &llama_backend_init, "C++: llama_backend_init() --> void");
    m.def("llama_numa_init", (void (*)(enum ggml_numa_strategy)) &llama_numa_init, "C++: llama_numa_init(enum ggml_numa_strategy) --> void", nb::arg("numa"));
    m.def("llama_backend_free", (void (*)()) &llama_backend_free, "C++: llama_backend_free() --> void");
    m.def("llama_load_model_from_file", (struct llama_model (*)(const char *, struct llama_model_params *)) &llama_load_model_from_file, "Load a model from file", nb::arg("path_model"), nb::arg("params"));
    m.def("llama_free_model", (void (*)(struct llama_model *)) &llama_free_model, "Free a model", nb::arg("model"));
    m.def("llama_new_context_with_model", (struct llama_context (*)(struct llama_model *, struct llama_context_params)) &llama_new_context_with_model, "New context with model", nb::arg("model"), nb::arg("params"));
    m.def("llama_free", (void (*)(struct llama_context *)) &llama_free, "Free context", nb::arg("ctx"));
    m.def("llama_time_us", (int64_t (*)()) &llama_time_us, "C++: llama_time_us() --> int");
    m.def("llama_max_devices", (size_t (*)()) &llama_max_devices, "C++: llama_max_devices() --> int");
    m.def("llama_supports_mmap", (bool (*)()) &llama_supports_mmap, "C++: llama_supports_mmap() --> bool");
    m.def("llama_supports_mlock", (bool (*)()) &llama_supports_mlock, "C++: llama_supports_mlock() --> bool");
    m.def("llama_supports_gpu_offload", (bool (*)()) &llama_supports_gpu_offload, "C++: llama_supports_gpu_offload() --> bool");
    m.def("llama_get_model", (const struct llama_model (*)(const struct llama_context *)) &llama_get_model, "get model from context", nb::arg("ctx"));



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
