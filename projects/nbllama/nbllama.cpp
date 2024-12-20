#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/ndarray.h>

#include "llamalib.h"

namespace nb = nanobind;

// forward declarations
struct llama_model {};
struct llama_context {};
struct llama_lora_adapter {};
struct ggml_threadpool {};

// ggml > ggml-backend > llama > common > sampling > arg > llamalib


NB_MODULE(nbllama, m) {
    m.doc() = "nanobind nbllama wrapper";
    m.attr("__version__") = "0.0.1";

    // -----------------------------------------------------------------------
    // attributes

    m.attr("LLAMA_DEFAULT_SEED") = 0xFFFFFFFF;


    // -----------------------------------------------------------------------
    // high-level api
    
    m.def("simple_prompt", &simple_prompt, 
        "Return a textual response after prompting a given model.", 
        nb::arg("model"),
        nb::arg("prompt"),
        nb::arg("n_predict") = 512,
        nb::arg("n_ctx") = 2048,
        nb::arg("disable_log") = true,
        nb::arg("n_threads") = 4
    );

    // -----------------------------------------------------------------------
    // ggml.h

    nb::enum_<ggml_numa_strategy>(m, "ggml_numa_strategy", "")
        .value("GGML_NUMA_STRATEGY_DISABLED", GGML_NUMA_STRATEGY_DISABLED)
        .value("GGML_NUMA_STRATEGY_DISTRIBUTE", GGML_NUMA_STRATEGY_DISTRIBUTE)
        .value("GGML_NUMA_STRATEGY_ISOLATE", GGML_NUMA_STRATEGY_ISOLATE)
        .value("GGML_NUMA_STRATEGY_NUMACTL", GGML_NUMA_STRATEGY_NUMACTL)
        .value("GGML_NUMA_STRATEGY_MIRROR", GGML_NUMA_STRATEGY_MIRROR)
        .value("GGML_NUMA_STRATEGY_COUNT", GGML_NUMA_STRATEGY_COUNT);

    m.def("ggml_time_us", (int64_t (*)(void)) &ggml_time_us);


    // -----------------------------------------------------------------------
    // llama.h
    
    nb::class_<llama_model> (m, "llama_model", "")
        .def(nb::init<>());

    nb::class_<llama_context> (m, "llama_context", "")
        .def(nb::init<>());

    nb::class_<llama_lora_adapter> (m, "llama_lora_adapter", "")
        .def(nb::init<>());


    nb::enum_<enum llama_vocab_type>(m, "llama_vocab_type")
        .value("LLAMA_VOCAB_TYPE_NONE", LLAMA_VOCAB_TYPE_NONE)
        .value("LLAMA_VOCAB_TYPE_SPM", LLAMA_VOCAB_TYPE_SPM)
        .value("LLAMA_VOCAB_TYPE_BPE", LLAMA_VOCAB_TYPE_BPE)
        .value("LLAMA_VOCAB_TYPE_WPM", LLAMA_VOCAB_TYPE_WPM)
        .value("LLAMA_VOCAB_TYPE_UGM", LLAMA_VOCAB_TYPE_UGM)
        .value("LLAMA_VOCAB_TYPE_RWKV", LLAMA_VOCAB_TYPE_RWKV)
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
        .value("LLAMA_VOCAB_PRE_TYPE_BLOOM", LLAMA_VOCAB_PRE_TYPE_BLOOM)
        .value("LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH", LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH)
        .value("LLAMA_VOCAB_PRE_TYPE_EXAONE", LLAMA_VOCAB_PRE_TYPE_EXAONE)
        .value("LLAMA_VOCAB_PRE_TYPE_CHAMELEON", LLAMA_VOCAB_PRE_TYPE_CHAMELEON)
        .export_values();

    nb::enum_<enum llama_rope_type>(m, "llama_rope_type")
        .value("LLAMA_ROPE_TYPE_NONE", LLAMA_ROPE_TYPE_NONE)
        .value("LLAMA_ROPE_TYPE_NORM", LLAMA_ROPE_TYPE_NORM)
        .value("LLAMA_ROPE_TYPE_NEOX", LLAMA_ROPE_TYPE_NEOX)
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
        .value("LLAMA_FTYPE_MOSTLY_TQ1_0", LLAMA_FTYPE_MOSTLY_TQ1_0)
        .value("LLAMA_FTYPE_MOSTLY_TQ2_0", LLAMA_FTYPE_MOSTLY_TQ2_0)
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
        .value("LLAMA_POOLING_TYPE_RANK", LLAMA_POOLING_TYPE_RANK)
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
        .def_rw("selected", &llama_token_data_array::selected)
        .def_rw("sorted", &llama_token_data_array::sorted);
        // llama_token  *  token;
        // float        *  embd;
        // llama_pos    *  pos;
        // int32_t      *  n_seq_id;
        // llama_seq_id ** seq_id;
        // int8_t       *  logits; // TODO: rename this to "output"


    nb::class_<llama_batch> (m, "llama_batch")
        .def(nb::init<>())
        .def_rw("n_tokens", &llama_batch::n_tokens)
        // .def_rw("token", &llama_batch::token)
        // .def_rw("embd", &llama_batch::embd)
        // .def_rw("pos", &llama_batch::pos)
        // .def_rw("n_seq_id", &llama_batch::n_seq_id)
        // .def_rw("seq_id", &llama_batch::seq_id)
        // FIXME: this is WRONG!!
        .def("set_last_logits_to_true", [](llama_batch& self) {
            self.logits[self.n_tokens - 1] = true;
        })
        // .def("get_logits", [](llama_batch& self) -> nb::ndarray<int8_t> {
        //     return to_array<int8_t>(self.logits, self.n_tokens);
        // })
        ;
    
    nb::enum_<enum llama_model_kv_override_type>(m, "llama_model_kv_override_type")
        .value("LLAMA_KV_OVERRIDE_TYPE_INT", LLAMA_KV_OVERRIDE_TYPE_INT)
        .value("LLAMA_KV_OVERRIDE_TYPE_FLOAT", LLAMA_KV_OVERRIDE_TYPE_FLOAT)
        .value("LLAMA_KV_OVERRIDE_TYPE_BOOL", LLAMA_KV_OVERRIDE_TYPE_BOOL)
        .value("LLAMA_KV_OVERRIDE_TYPE_STR", LLAMA_KV_OVERRIDE_TYPE_STR)
        .export_values();

    nb::class_<llama_model_kv_override> (m, "llama_model_kv_override")
        .def(nb::init<>())
        .def_rw("tag", &llama_model_kv_override::tag);
        // ..
    
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
        .def_rw("flash_attn", &llama_context_params::flash_attn)
        .def_rw("no_perf", &llama_context_params::no_perf);

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
        // ..

    nb::class_<llama_logit_bias> (m, "llama_logit_bias", "")
        .def(nb::init<>());
        // .def_rw("token", &llama_model_quantize_params::token)
        // .def_rw("bias", &llama_model_quantize_params::bias);

    nb::class_<llama_sampler_chain_params> (m, "llama_sampler_chain_params", "")
        .def(nb::init<>())
        .def_rw("no_perf", &llama_sampler_chain_params::no_perf);

    nb::class_<llama_chat_message> (m, "llama_chat_message")
        .def(nb::init<>())
        .def_rw("role", &llama_chat_message::role)
        .def_rw("content", &llama_chat_message::content);

    m.def("llama_model_default_params", (struct llama_model_params (*)()) &llama_model_default_params, "C++: llama_model_default_params() --> struct llama_model_params");
    m.def("llama_context_default_params", (struct llama_context_params (*)()) &llama_context_default_params, "C++: llama_context_default_params() --> struct llama_context_params");
    m.def("llama_sampler_chain_default_params", (struct llama_sampler_chain_params (*)()) &llama_sampler_chain_default_params);
    m.def("llama_model_quantize_default_params", (struct llama_model_quantize_params (*)()) &llama_model_quantize_default_params, "C++: llama_model_quantize_default_params() --> struct llama_model_quantize_params");
    
    m.def("llama_backend_init", (void (*)()) &llama_backend_init, "C++: llama_backend_init() --> void");
    m.def("llama_numa_init", (void (*)(enum ggml_numa_strategy)) &llama_numa_init, "C++: llama_numa_init(enum ggml_numa_strategy) --> void", nb::arg("numa"));
    
    m.def("llama_attach_threadpool", (void (*)(struct llama_context *, ggml_threadpool_t, ggml_threadpool_t)) &llama_attach_threadpool, "", nb::arg("ctx"), nb::arg("threadpool"), nb::arg("threadpool_batch"));
    m.def("llama_detach_threadpool", (void (*)(struct llama_context *)) &llama_detach_threadpool, "", nb::arg("ctx"));

    m.def("llama_backend_free", (void (*)()) &llama_backend_free, "C++: llama_backend_free() --> void");
    m.def("llama_load_model_from_file", (struct llama_model * (*)(const char *, struct llama_model_params *)) &llama_load_model_from_file, "Load a model from file", nb::arg("path_model"), nb::arg("params"), nb::rv_policy::reference);
    m.def("llama_free_model", (void (*)(struct llama_model *)) &llama_free_model, "Free a model", nb::arg("model"));
   
    m.def("llama_new_context_with_model", (struct llama_context * (*)(struct llama_model *, struct llama_context_params)) &llama_new_context_with_model, "New context with model", nb::arg("model"), nb::arg("params"), nb::rv_policy::reference);
   
    m.def("llama_free", (void (*)(struct llama_context *)) &llama_free, "Free context", nb::arg("ctx"));
    m.def("llama_time_us", (int64_t (*)()) &llama_time_us, "C++: llama_time_us() --> int");
    m.def("llama_max_devices", (size_t (*)()) &llama_max_devices, "C++: llama_max_devices() --> int");
    m.def("llama_supports_mmap", (bool (*)()) &llama_supports_mmap, "C++: llama_supports_mmap() --> bool");
    m.def("llama_supports_mlock", (bool (*)()) &llama_supports_mlock, "C++: llama_supports_mlock() --> bool");
    m.def("llama_supports_gpu_offload", (bool (*)()) &llama_supports_gpu_offload, "C++: llama_supports_gpu_offload() --> bool");

    m.def("llama_n_ctx", (uint32_t (*)(const struct llama_context *)) &llama_n_ctx, "get n_ctx from context", nb::arg("ctx"));
    m.def("llama_n_batch", (uint32_t (*)(const struct llama_context *)) &llama_n_batch, "get n_batch from context", nb::arg("ctx"));
    m.def("llama_n_ubatch", (uint32_t (*)(const struct llama_context *)) &llama_n_ubatch, "get n_ubatch from context", nb::arg("ctx"));
    m.def("llama_n_seq_max", (uint32_t (*)(const struct llama_context *)) &llama_n_seq_max, "get n_seq_max from context", nb::arg("ctx"));

    m.def("llama_n_vocab", (int32_t (*)(const struct llama_model *)) &llama_n_vocab, "get n_vocab from model", nb::arg("model"));
    m.def("llama_n_ctx_train", (int32_t (*)(const struct llama_model *)) &llama_n_ctx_train, "get n_ctx_train from model", nb::arg("model"));
    m.def("llama_n_embd", (int32_t (*)(const struct llama_model *)) &llama_n_embd, "get n_embed from model", nb::arg("model"));
    m.def("llama_n_layer", (int32_t (*)(const struct llama_model *)) &llama_n_layer, "get n_layer from model", nb::arg("model"));
    m.def("llama_n_head", (int32_t (*)(const struct llama_model *)) &llama_n_head, "get n_head from model", nb::arg("model"));

    m.def("llama_get_model", (const struct llama_model * (*)(const struct llama_context *)) &llama_get_model, "get model from context", nb::arg("ctx"));
    m.def("get_llama_pooling_type", (enum llama_pooling_type (*)(const struct llama_context *)) &llama_pooling_type, "get pooling_type from context", nb::arg("ctx"));
    m.def("get_llama_vocab_type", (enum llama_vocab_type (*)(const struct llama_model *)) &llama_vocab_type, "get vocab_type from model", nb::arg("model"));
    m.def("get_llama_rope_type", (enum llama_rope_type (*)(const struct llama_model *)) &llama_rope_type, "get rope_type from model", nb::arg("model"));

    m.def("llama_rope_freq_scale_train", (float (*)(const struct llama_model *)) &llama_rope_freq_scale_train, "get rope_freq_scale_train from model", nb::arg("model"));

    m.def("llama_model_meta_val_str", (int32_t (*)(const struct llama_model *, const char *, const char *, size_t)) &llama_model_meta_val_str, "get meta_val_str from model", nb::arg("model"), nb::arg("key"), nb::arg("buf"), nb::arg("buf_size"));
    m.def("llama_model_meta_count", (int32_t (*)(const struct llama_model *)) &llama_model_meta_count, "get meta_count from model", nb::arg("model"));
    m.def("llama_model_meta_key_by_index", (int32_t (*)(const struct llama_model *, int32_t, const char *, size_t)) &llama_model_meta_key_by_index, "get meta_key_by_index from model", nb::arg("model"), nb::arg("i"), nb::arg("buf"), nb::arg("buf_size"));
    m.def("llama_model_meta_val_str_by_index", (int32_t (*)(const struct llama_model *, int32_t, const char *, size_t)) &llama_model_meta_val_str_by_index, "get meta_val_str_by_index from model", nb::arg("model"), nb::arg("i"), nb::arg("buf"), nb::arg("buf_size"));
    m.def("llama_model_desc", (int32_t (*)(const struct llama_model *, const char *, size_t)) &llama_model_desc, "get model_desc from model", nb::arg("model"), nb::arg("buf"), nb::arg("buf_size"));

    m.def("llama_model_size", (uint64_t (*)(const struct llama_model *)) &llama_model_size, "get model_size from model", nb::arg("model"));
    m.def("llama_model_n_params", (uint64_t (*)(const struct llama_model *)) &llama_model_n_params, "get model_n_params from model", nb::arg("model"));

    m.def("llama_get_model_tensor", (struct ggml_tensor* (*)(struct llama_model *)) &llama_get_model_tensor, "get model tensor from model", nb::arg("model"));
    // m.def("llama_get_model_tensor", (struct ggml_tensor* (*)(struct llama_model *, const char *)) &llama_get_model_tensor, "get named model tensor from model", nb::arg("model"), nb::arg("name"));

    m.def("llama_model_has_encoder", (bool (*)(const struct llama_model *)) &llama_model_has_encoder, "model has encoder?", nb::arg("model"));
    m.def("llama_model_has_decoder", (bool (*)(const struct llama_model *)) &llama_model_has_decoder, "model has decoder?", nb::arg("model"));

    m.def("llama_model_decoder_start_token", (llama_token (*)(const struct llama_model *)) &llama_model_decoder_start_token, "get decoder_start_token from model", nb::arg("model"));

    m.def("llama_model_is_recurrent", (bool (*)(const struct llama_model *)) &llama_model_is_recurrent, "check if model is recurrent", nb::arg("model"));

    m.def("llama_model_quantize", (int (*)(const char *, const char *, const struct llama_model_quantize_params *)) &llama_model_quantize, "C++: llama_model_quantize(const char *, const char *, const struct llama_model_quantize_params *) --> int", nb::arg("fname_inp"), nb::arg("fname_out"), nb::arg("params"));

    m.def("llama_lora_adapter_init", (struct llama_lora_adapter (*)(const struct llama_model *, const char *)) &llama_lora_adapter_init, "", nb::arg("model"), nb::arg("path_lora"));
    m.def("llama_lora_adapter_set", (int32_t (*)(struct llama_context*, struct llama_lora_adapter*, float)) &llama_lora_adapter_set, "", nb::arg("ctx"), nb::arg("adapter"), nb::arg("scale"));
    m.def("llama_lora_adapter_remove", (int32_t (*)(struct llama_context*, struct llama_lora_adapter*)) &llama_lora_adapter_remove, "", nb::arg("ctx"), nb::arg("adapter"));
    m.def("llama_lora_adapter_clear", (void (*)(struct llama_context*)) &llama_lora_adapter_clear, "", nb::arg("ctx"));
    m.def("llama_lora_adapter_free", (void (*)(struct llama_lora_adapter*)) &llama_lora_adapter_free, "", nb::arg("adapter"));

    m.def("llama_control_vector_apply", (int32_t (*)(struct llama_context * , const float*, size_t, int32_t, int32_t, int32_t)) &llama_model_quantize, "", nb::arg("lctx"), nb::arg("data"), nb::arg("len"), nb::arg("n_embd"), nb::arg("il_start"), nb::arg("il_end"));

    // KV cache
    
    nb::class_<llama_kv_cache_view_cell>(m, "llama_kv_cache_view_cell")
        .def(nb::init<>())
        .def_rw("pos", &llama_kv_cache_view_cell::pos);

    nb::class_<llama_kv_cache_view>(m, "llama_kv_cache_view", "")
        .def(nb::init<>())
        .def_rw("n_cells", &llama_kv_cache_view::n_cells)
        .def_rw("n_seq_max", &llama_kv_cache_view::n_seq_max)
        .def_rw("token_count", &llama_kv_cache_view::token_count)
        .def_rw("used_cells", &llama_kv_cache_view::used_cells)
        .def_rw("max_contiguous", &llama_kv_cache_view::max_contiguous)
        .def_rw("max_contiguous_idx", &llama_kv_cache_view::max_contiguous_idx);

    m.def("llama_kv_cache_view_init", (struct llama_kv_cache_view (*)(const struct llama_context *, int32_t)) &llama_kv_cache_view_init, "", nb::arg("ctx"), nb::arg("n_seq_max"));
    m.def("llama_kv_cache_view_free", (void (*)(struct llama_kv_cache_view *)) &llama_kv_cache_view_free, "", nb::arg("view"));
    m.def("llama_kv_cache_view_update", (void (*)(const struct llama_context *, struct llama_kv_cache_view *)) &llama_kv_cache_view_update, "", nb::arg("ctx"), nb::arg("view"));
    m.def("llama_get_kv_cache_token_count", (int32_t (*)(const struct llama_context *)) &llama_get_kv_cache_token_count, "", nb::arg("ctx"));
    m.def("llama_get_kv_cache_used_cells", (int32_t (*)(const struct llama_context *)) &llama_get_kv_cache_used_cells, "", nb::arg("ctx"));
    m.def("llama_kv_cache_clear", (void (*)(const struct llama_context *)) &llama_kv_cache_clear, "", nb::arg("ctx"));
    m.def("llama_kv_cache_seq_rm", (bool (*)(const struct llama_context *, llama_seq_id, llama_pos, llama_pos)) &llama_kv_cache_seq_rm, "", nb::arg("ctx"), nb::arg("seq_id"), nb::arg("p0"), nb::arg("p1"));
    m.def("llama_kv_cache_seq_cp", (void (*)(const struct llama_context *, llama_seq_id, llama_seq_id, llama_pos, llama_pos)) &llama_kv_cache_seq_cp, "", nb::arg("ctx"), nb::arg("seq_id_src"), nb::arg("seq_id_dst"), nb::arg("p0"), nb::arg("p1"));
    m.def("llama_kv_cache_seq_keep", (void (*)(const struct llama_context *, llama_seq_id)) &llama_kv_cache_seq_keep, "", nb::arg("ctx"), nb::arg("seq_id"));
    m.def("llama_kv_cache_seq_add", (void (*)(const struct llama_context *, llama_seq_id, llama_pos, llama_pos, llama_pos)) &llama_kv_cache_seq_add, "", nb::arg("ctx"), nb::arg("seq_id"), nb::arg("p0"), nb::arg("p1"), nb::arg("delta"));
    m.def("llama_kv_cache_seq_div", (void (*)(const struct llama_context *, llama_seq_id, llama_pos, llama_pos, int)) &llama_kv_cache_seq_div, "", nb::arg("ctx"), nb::arg("seq_id"), nb::arg("p0"), nb::arg("p1"), nb::arg("d"));
    m.def("llama_kv_cache_seq_pos_max", (llama_pos (*)(const struct llama_context *, llama_seq_id)) &llama_kv_cache_seq_pos_max, "", nb::arg("ctx"), nb::arg("seq_id"));
    m.def("llama_kv_cache_defrag", (void (*)(const struct llama_context *)) &llama_kv_cache_defrag, "", nb::arg("ctx"));
    m.def("llama_kv_cache_update", (void (*)(const struct llama_context *)) &llama_kv_cache_update, "", nb::arg("ctx"));

    // State / sessions

    m.def("llama_state_get_size", (size_t (*)(const struct llama_context *)) &llama_state_get_size, "", nb::arg("ctx"));
    m.def("llama_state_get_data", (size_t (*)(const struct llama_context *, uint8_t *, size_t)) &llama_state_get_data, "", nb::arg("ctx"), nb::arg("dst"), nb::arg("size"));
    m.def("llama_state_set_data", (size_t (*)(const struct llama_context *, const uint8_t *, size_t)) &llama_state_set_data, "", nb::arg("ctx"), nb::arg("dst"), nb::arg("size"));
    m.def("llama_state_load_file", (size_t (*)(const struct llama_context *, const char *, llama_token *, size_t, size_t *)) &llama_state_load_file, "", nb::arg("ctx"), nb::arg("path_session"), nb::arg("tokens_out"), nb::arg("n_token_capacity"), nb::arg("n_token_count_out"));
    m.def("llama_state_save_file", (size_t (*)(const struct llama_context *, const char *, const llama_token *, size_t)) &llama_state_save_file, "", nb::arg("ctx"), nb::arg("path_session"), nb::arg("tokens"), nb::arg("n_token_count"));
    m.def("llama_state_seq_get_size", (size_t (*)(const struct llama_context *, llama_seq_id)) &llama_state_seq_get_size, "", nb::arg("ctx"), nb::arg("seq_id"));
    m.def("llama_state_seq_get_data", (size_t (*)(const struct llama_context *, uint8_t *, size_t, llama_seq_id)) &llama_state_seq_get_data, "", nb::arg("ctx"), nb::arg("dst"), nb::arg("size"), nb::arg("seq_id"));
    m.def("llama_state_seq_set_data", (size_t (*)(const struct llama_context *, const uint8_t *, size_t, llama_seq_id)) &llama_state_seq_set_data, "", nb::arg("ctx"), nb::arg("src"), nb::arg("size"), nb::arg("dest_seq_id"));
    m.def("llama_state_seq_save_file", (size_t (*)(const struct llama_context *, const char *, llama_seq_id, const llama_token *, size_t)) &llama_state_seq_save_file, "", nb::arg("ctx"), nb::arg("filepath"), nb::arg("seq_id"), nb::arg("tokens"), nb::arg("n_token_count"));
    m.def("llama_state_seq_load_file", (size_t (*)(const struct llama_context *, const char *, llama_seq_id, llama_token *, size_t, size_t *)) &llama_state_seq_load_file, "", nb::arg("ctx"), nb::arg("filepath"), nb::arg("dest_seq_id"), nb::arg("tokens_out"), nb::arg("n_token_capacity"), nb::arg("n_token_count_out"));

    // decoding

    m.def("llama_batch_get_one", (struct llama_batch (*)(int *, int)) &llama_batch_get_one, "C++: llama_batch_get_one(int *, int) --> struct llama_batch", nb::arg("tokens"), nb::arg("n_tokens"));
    m.def("llama_batch_init", (struct llama_batch (*)(int, int, int)) &llama_batch_init, "C++: llama_batch_init(int, int, int) --> struct llama_batch", nb::arg("n_tokens"), nb::arg("embd"), nb::arg("n_seq_max"));
    m.def("llama_batch_free", (void (*)(struct llama_batch)) &llama_batch_free, "C++: llama_batch_free(struct llama_batch) --> void", nb::arg("batch"));

    m.def("llama_encode", (int32_t (*)(const struct llama_context *, struct llama_batch)) &llama_encode, "", nb::arg("ctx"), nb::arg("batch"));
    m.def("llama_decode", (int32_t (*)(const struct llama_context *, struct llama_batch)) &llama_decode, "", nb::arg("ctx"), nb::arg("batch"));

    m.def("llama_set_n_threads", (void (*)(const struct llama_context *, uint32_t, uint32_t)) &llama_set_n_threads, "", nb::arg("ctx"), nb::arg("n_threads"), nb::arg("n_threads_batch"));
    m.def("llama_n_threads", (uint32_t (*)(const struct llama_context *)) &llama_n_threads, "", nb::arg("ctx"));
    m.def("llama_n_threads_batch", (uint32_t (*)(const struct llama_context *)) &llama_n_threads_batch, "", nb::arg("ctx"));

    m.def("llama_set_embeddings", (void (*)(const struct llama_context *, bool)) &llama_set_embeddings, "", nb::arg("ctx"), nb::arg("embeddings"));

    m.def("llama_set_causal_attn", (void (*)(const struct llama_context *, bool)) &llama_set_causal_attn, "", nb::arg("ctx"), nb::arg("causal_attn"));

    void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);    

    m.def("llama_synchronize", (void (*)(const struct llama_context *)) &llama_synchronize, "", nb::arg("ctx"));
    
    m.def("llama_get_logits", (float* (*)(const struct llama_context *)) &llama_get_logits, "", nb::arg("ctx"));
    m.def("llama_get_logits_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_logits_ith, "", nb::arg("ctx"), nb::arg("i"));

    m.def("llama_get_embeddings", (float* (*)(const struct llama_context *)) &llama_get_embeddings, "", nb::arg("ctx"));
    m.def("llama_get_embeddings_ith", (float* (*)(const struct llama_context *, int32_t)) &llama_get_embeddings_ith, "", nb::arg("ctx"), nb::arg("i"));
    m.def("llama_get_embeddings_seq", (float* (*)(const struct llama_context *, llama_seq_id)) &llama_get_embeddings_seq, "", nb::arg("ctx"), nb::arg("seq_id"));

    m.def("llama_token_get_text", (const char* (*)(const struct llama_model *, llama_token)) &llama_token_get_text, "", nb::arg("model"), nb::arg("token"));
    m.def("llama_token_get_score", (float (*)(const struct llama_model *, llama_token)) &llama_token_get_score, "", nb::arg("model"), nb::arg("token"));

    m.def("llama_token_get_attr", (enum llama_token_attr (*)(const struct llama_model *, llama_token)) &llama_token_get_attr, "", nb::arg("model"), nb::arg("token"));
    m.def("llama_token_is_eog", (bool (*)(const struct llama_model *, llama_token)) &llama_token_is_eog, "", nb::arg("model"), nb::arg("token"));
    m.def("llama_token_is_control", (bool (*)(const struct llama_model *, llama_token)) &llama_token_is_control, "", nb::arg("model"), nb::arg("token"));

    m.def("llama_token_bos", (llama_token (*)(const struct llama_model *)) &llama_token_bos, "", nb::arg("model"));
    m.def("llama_token_eos", (llama_token (*)(const struct llama_model *)) &llama_token_eos, "", nb::arg("model"));
    m.def("llama_token_eot", (llama_token (*)(const struct llama_model *)) &llama_token_eot, "", nb::arg("model"));
    m.def("llama_token_cls", (llama_token (*)(const struct llama_model *)) &llama_token_cls, "", nb::arg("model"));
    m.def("llama_token_sep", (llama_token (*)(const struct llama_model *)) &llama_token_sep, "", nb::arg("model"));
    m.def("llama_token_nl",  (llama_token (*)(const struct llama_model *)) &llama_token_nl,  "", nb::arg("model"));
    m.def("llama_token_pad", (llama_token (*)(const struct llama_model *)) &llama_token_pad, "", nb::arg("model"));

    m.def("llama_add_bos_token", (int32_t (*)(const struct llama_model *)) &llama_add_bos_token, "", nb::arg("model"));
    m.def("llama_add_eos_token", (int32_t (*)(const struct llama_model *)) &llama_add_eos_token, "", nb::arg("model"));

    m.def("llama_token_fim_pre", (llama_token (*)(const struct llama_model *)) &llama_token_fim_pre, "", nb::arg("model"));
    m.def("llama_token_fim_mid", (llama_token (*)(const struct llama_model *)) &llama_token_fim_mid, "", nb::arg("model"));
    m.def("llama_token_fim_suf", (llama_token (*)(const struct llama_model *)) &llama_token_fim_suf, "", nb::arg("model"));
    m.def("llama_token_fim_pad", (llama_token (*)(const struct llama_model *)) &llama_token_fim_pad, "", nb::arg("model"));
    m.def("llama_token_fim_rep", (llama_token (*)(const struct llama_model *)) &llama_token_fim_rep, "", nb::arg("model"));
    m.def("llama_token_fim_sep", (llama_token (*)(const struct llama_model *)) &llama_token_fim_sep, "", nb::arg("model"));


    m.def("llama_tokenize", (int32_t (*)(const struct llama_model *, const char*, int32_t, llama_token*, int32_t, bool, bool)) &llama_tokenize, "", nb::arg("model"), nb::arg("text"), nb::arg("text_len"), nb::arg("tokens"), nb::arg("n_tokens_max"), nb::arg("add_special"), nb::arg("parse_special"));
    // m.def("llama_token_to_piece", (int32_t (*)(const struct llama_model *, llama_token, char*, int32_t, int32_t, bool)) &llama_token_to_piece, "", nb::arg("model"), nb::arg("token"), nb::arg("buf"), nb::arg("length"), nb::arg("lstrip"), nb::arg("special"));
    // m.def("llama_detokenize", (int32_t (*)(const struct llama_model *, const llama_token*, int32_t, char*, int32_t, bool, bool)) &llama_detokenize, "", nb::arg("model"), nb::arg("tokens"), nb::arg("n_tokens"), nb::arg("text"), nb::arg("text_len_max"), nb::arg("remove_special"), nb::arg("unparse_special"));
    
    // Chat templates

    // m.def("llama_chat_apply_template", (int32_t (*)(const struct llama_model *, const char*, const struct llama_chat_message*, size_t, bool, char*, int32_t)) &llama_chat_apply_template, "", nb::arg("model"), nb::arg("tmpl"), nb::arg("chat"), nb::arg("n_msg"), nb::arg("add_ass"), nb::arg("buf"), nb::arg("length"));

    nb::class_<llama_sampler_i> (m, "llama_sampler_i", "")
        .def(nb::init<>());
        // .def_readwrite("name", &llama_sampler_i::name)
        // .def_readwrite("accept", &llama_sampler_i::accept);
        // const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        // void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        // void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        // void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        // struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        // void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

    nb::class_<llama_sampler> (m, "llama_sampler", "")
        .def(nb::init<>())
        .def_rw("iface", &llama_sampler::iface)
        .def_rw("ctx", &llama_sampler::ctx);


    m.def("llama_sampler_name", (const char * (*)(const struct llama_sampler *)) &llama_sampler_name, "", nb::arg("smpl"));
    m.def("llama_sampler_accept", (void (*)(struct llama_sampler *, llama_token)) &llama_sampler_accept, "", nb::arg("smpl"),  nb::arg("token"));
    m.def("llama_sampler_apply", (void (*)(struct llama_sampler *, llama_token_data_array *)) &llama_sampler_apply, "", nb::arg("smpl"),  nb::arg("cur_p"));
    m.def("llama_sampler_reset", (void (*)(struct llama_sampler *)) &llama_sampler_apply, "", nb::arg("smpl"));
    m.def("llama_sampler_clone", (struct llama_sampler * (*)(const struct llama_sampler *)) &llama_sampler_clone, "", nb::arg("smpl"), nb::rv_policy::reference);
    m.def("llama_sampler_free", (void (*)(const struct llama_sampler *)) &llama_sampler_free, "", nb::arg("smpl"));
    
    m.def("llama_sampler_chain_init", (struct llama_sampler *  (*)(struct llama_sampler_chain_params)) &llama_sampler_chain_init, "", nb::arg("params"), nb::rv_policy::reference);
    
    m.def("llama_sampler_chain_add", (void (*)(struct llama_sampler *, struct llama_sampler *)) &llama_sampler_chain_add, "", nb::arg("chain"), nb::arg("smpl"));
    m.def("llama_sampler_chain_get", (struct llama_sampler * (*)(const struct llama_sampler *, int32_t)) &llama_sampler_chain_get, "", nb::arg("chain"), nb::arg("i"), nb::rv_policy::reference);
    m.def("llama_sampler_chain_n", (int (*)(const struct llama_sampler *)) &llama_sampler_chain_n, "", nb::arg("chain"));
    
    m.def("llama_sampler_chain_remove", (struct llama_sampler * (*)(const struct llama_sampler *, int32_t)) &llama_sampler_chain_remove, "", nb::arg("chain"), nb::arg("i"), nb::rv_policy::reference);    
    
    m.def("llama_sampler_init_greedy", (struct llama_sampler * (*)(void)) &llama_sampler_init_greedy, nb::rv_policy::reference);
    m.def("llama_sampler_init_dist", (struct llama_sampler * (*)(uint32_t)) &llama_sampler_init_dist, "", nb::arg("seed"), nb::rv_policy::reference);

    // DEPRECATED
    // m.def("llama_sampler_init_softmax", (struct llama_sampler * (*)(void)) &llama_sampler_init_softmax);
    m.def("llama_sampler_init_top_k", (struct llama_sampler * (*)(uint32_t)) &llama_sampler_init_top_k, "", nb::arg("k"));
    m.def("llama_sampler_init_top_p", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_top_p, "", nb::arg("p"), nb::arg("min_keep"));
    m.def("llama_sampler_init_min_p", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_min_p, "", nb::arg("p"), nb::arg("min_keep"));
    m.def("llama_sampler_init_tail_free", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_tail_free, "", nb::arg("z"), nb::arg("min_keep"));

    m.def("llama_sampler_init_typical", (struct llama_sampler * (*)(float, size_t)) &llama_sampler_init_typical, "", nb::arg("p"), nb::arg("min_keep"));
    m.def("llama_sampler_init_temp", (struct llama_sampler * (*)(float)) &llama_sampler_init_temp, "", nb::arg("t"));
    m.def("llama_sampler_init_temp_ext", (struct llama_sampler * (*)(float, float, float)) &llama_sampler_init_temp_ext, "", nb::arg("t"), nb::arg("delta"), nb::arg("exponent"));
    m.def("llama_sampler_init_xtc", (struct llama_sampler * (*)(float, float, float, uint32_t)) &llama_sampler_init_xtc, "", nb::arg("p"), nb::arg("t"), nb::arg("min_keep"), nb::arg("seed")); 
    m.def("llama_sampler_init_mirostat", (struct llama_sampler * (*)(int32_t, uint32_t, float, float, int32_t)) &llama_sampler_init_mirostat, "", nb::arg("n_vocab"), nb::arg("seed"), nb::arg("tau"), nb::arg("eta"), nb::arg("m"));
    m.def("llama_sampler_init_mirostat_v2", (struct llama_sampler * (*)(uint32_t, float, float)) &llama_sampler_init_mirostat_v2, "", nb::arg("seed"), nb::arg("tau"), nb::arg("eta"));
    m.def("llama_sampler_init_grammar", (struct llama_sampler * (*)(const struct llama_model *, const char *, const char *)) &llama_sampler_init_grammar, "", nb::arg("model"), nb::arg("grammar_str"), nb::arg("grammar_root"));
    m.def("llama_sampler_init_penalties", (struct llama_sampler * (*)(int32_t, llama_token, llama_token, int32_t, float, float, float, bool, bool)) &llama_sampler_init_penalties, "", nb::arg("n_vocab"), nb::arg("special_eos_id"), nb::arg("linefeed_id"), nb::arg("penalty_last_n"), nb::arg("penalty_repeat"), nb::arg("epenalty_freq"), nb::arg("penalty_present"), nb::arg("penalize_nl"), nb::arg("ignore_eos"));
    m.def("llama_sampler_init_logit_bias", (struct llama_sampler * (*)(int32_t, int32_t, const llama_logit_bias *)) &llama_sampler_init_logit_bias, "", nb::arg("n_vocab"), nb::arg("n_logit_bias"), nb::arg("logit_bias"));
    m.def("llama_sampler_init_infill", (struct llama_sampler * (*)(const struct llama_model *)) &llama_sampler_init_infill, "", nb::arg("model"));

    m.def("llama_sampler_sample", (llama_token (*)(struct llama_sampler *, struct llama_context *, int32_t)) &llama_sampler_sample, "", nb::arg("smpl"), nb::arg("ctx"), nb::arg("idx"));

    // m.def("llama_split_path", (int (*)(char *, int, const char *, int, int)) &llama_split_path, "Build a split GGUF final path for this chunk.", nb::arg("split_path"), nb::arg("maxlen"), nb::arg("path_prefix"), nb::arg("split_no"), nb::arg("split_count"));
    // m.def("llama_split_prefix", (int (*)(char *, int, const char *, int, int)) &llama_split_prefix, "Extract the path prefix from the split_path if and only if the split_no and split_count match.", nb::arg("split_prefix"), nb::arg("maxlen"), nb::arg("split_path"), nb::arg("split_no"), nb::arg("split_count"));

    // Print system information
    m.def("llama_print_system_info", (const char * (*)(void)) &llama_print_system_info, "");

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    m.def("llama_log_set", (void (*)(ggml_log_callback log_callback, void * user_data)) &llama_log_set, "", nb::arg("log_callback"), nb::arg("user_data"));

    // Performance utils

    nb::class_<llama_perf_context_data> (m, "llama_perf_context_data", "")
        .def(nb::init<>())
        .def_rw("t_start_ms", &llama_perf_context_data::t_start_ms)
        .def_rw("t_load_ms", &llama_perf_context_data::t_load_ms)
        .def_rw("t_p_eval_ms", &llama_perf_context_data::t_p_eval_ms)
        .def_rw("t_eval_ms", &llama_perf_context_data::t_eval_ms)
        .def_rw("n_p_eval", &llama_perf_context_data::n_p_eval)
        .def_rw("n_eval", &llama_perf_context_data::n_eval);

    nb::class_<llama_perf_sampler_data> (m, "llama_perf_sampler_data", "")
        .def(nb::init<>())
        .def_rw("t_sample_ms", &llama_perf_sampler_data::t_sample_ms)
        .def_rw("n_sample", &llama_perf_sampler_data::n_sample);

    m.def("llama_perf_context", (struct llama_perf_context_data (*)(const struct llama_context *)) &llama_perf_context, "", nb::arg("ctx"));
    m.def("llama_perf_context_print", (void (*)(const struct llama_context *)) &llama_perf_context_print, "", nb::arg("ctx"));
    m.def("llama_perf_context_reset", (void (*)(struct llama_context *)) &llama_perf_context_reset, "", nb::arg("ctx"));

    m.def("llama_perf_sampler", (struct llama_perf_sampler_data (*)(const struct llama_sampler *)) &llama_perf_sampler, "", nb::arg("chain"));
    m.def("llama_perf_sampler_print", (void (*)(const struct llama_sampler *)) &llama_perf_sampler_print, "", nb::arg("chain"));
    m.def("llama_perf_sampler_reset", (void (*)(struct llama_sampler *)) &llama_perf_sampler_reset, "", nb::arg("chain"));

    m.def("llama_perf_dump_yaml", (const char * (*)(FILE *, const struct llama_context *)) &llama_perf_dump_yaml, "", nb::arg("stream"), nb::arg("ctx"));

    // -----------------------------------------------------------------------
    // common.h

    nb::class_<common_params> (m, "common_params", "")
        .def(nb::init<>())
        .def_rw("n_predict", &common_params::n_predict)
        .def_rw("n_ctx", &common_params::n_ctx)
        .def_rw("n_batch", &common_params::n_batch)
        .def_rw("n_ubatch", &common_params::n_ubatch)
        .def_rw("n_keep", &common_params::n_keep)
        .def_rw("n_draft", &common_params::n_draft)
        .def_rw("n_chunks", &common_params::n_chunks)
        .def_rw("n_parallel", &common_params::n_parallel)
        .def_rw("n_sequences", &common_params::n_sequences)
        .def_rw("p_split", &common_params::p_split)
        .def_rw("n_gpu_layers", &common_params::n_gpu_layers)
        .def_rw("n_gpu_layers_draft", &common_params::n_gpu_layers_draft)
        .def_rw("main_gpu", &common_params::main_gpu)
        .def_rw("grp_attn_n", &common_params::grp_attn_n)
        .def_rw("grp_attn_w", &common_params::grp_attn_w)
        .def_rw("n_print", &common_params::n_print)
        .def_rw("rope_freq_base", &common_params::rope_freq_base)
        .def_rw("rope_freq_scale", &common_params::rope_freq_scale)
        .def_rw("yarn_ext_factor", &common_params::yarn_ext_factor)
        .def_rw("yarn_attn_factor", &common_params::yarn_attn_factor)
        .def_rw("yarn_beta_fast", &common_params::yarn_beta_fast)
        .def_rw("yarn_beta_slow", &common_params::yarn_beta_slow)
        .def_rw("yarn_orig_ctx", &common_params::yarn_orig_ctx)
        .def_rw("defrag_thold", &common_params::defrag_thold)
        .def_rw("numa", &common_params::numa)
        .def_rw("split_mode", &common_params::split_mode)
        .def_rw("rope_scaling_type", &common_params::rope_scaling_type)
        .def_rw("pooling_type", &common_params::pooling_type)
        .def_rw("attention_type", &common_params::attention_type)
        .def_rw("sparams", &common_params::sparams)
        .def_rw("model", &common_params::model)
        .def_rw("model_draft", &common_params::model_draft)
        .def_rw("model_alias", &common_params::model_alias)
        .def_rw("model_url", &common_params::model_url)
        .def_rw("hf_token", &common_params::hf_token)
        .def_rw("hf_repo", &common_params::hf_repo)
        .def_rw("hf_file", &common_params::hf_file)
        .def_rw("prompt", &common_params::prompt)
        .def_rw("prompt_file", &common_params::prompt_file)
        .def_rw("path_prompt_cache", &common_params::path_prompt_cache)
        .def_rw("input_prefix", &common_params::input_prefix)
        .def_rw("input_suffix", &common_params::input_suffix)
        .def_rw("logdir", &common_params::logdir)
        .def_rw("lookup_cache_static", &common_params::lookup_cache_static)
        .def_rw("lookup_cache_dynamic", &common_params::lookup_cache_dynamic)
        .def_rw("logits_file", &common_params::logits_file)
        .def_rw("rpc_servers", &common_params::rpc_servers)
        .def_rw("in_files", &common_params::in_files)
        .def_rw("antiprompt", &common_params::antiprompt)
        .def_rw("kv_overrides", &common_params::kv_overrides)
        .def_rw("lora_adapters", &common_params::lora_adapters)
        .def_rw("control_vectors", &common_params::control_vectors)
        .def_rw("verbosity", &common_params::verbosity)
        .def_rw("control_vector_layer_start", &common_params::control_vector_layer_start)
        .def_rw("control_vector_layer_end", &common_params::control_vector_layer_end)
        .def_rw("ppl_stride", &common_params::ppl_stride)
        .def_rw("ppl_output_type", &common_params::ppl_output_type)
        .def_rw("hellaswag", &common_params::hellaswag)
        .def_rw("hellaswag_tasks", &common_params::hellaswag_tasks)
        .def_rw("winogrande", &common_params::winogrande)
        .def_rw("winogrande_tasks", &common_params::winogrande_tasks)
        .def_rw("multiple_choice", &common_params::multiple_choice)
        .def_rw("multiple_choice_tasks", &common_params::multiple_choice_tasks)
        .def_rw("kl_divergence", &common_params::kl_divergence)
        .def_rw("usage", &common_params::usage)
        .def_rw("use_color", &common_params::use_color)
        .def_rw("special", &common_params::special)
        .def_rw("interactive", &common_params::interactive)
        .def_rw("interactive_first", &common_params::interactive_first)
        .def_rw("conversation", &common_params::conversation)
        .def_rw("prompt_cache_all", &common_params::prompt_cache_all)
        .def_rw("prompt_cache_ro", &common_params::prompt_cache_ro)
        .def_rw("escape", &common_params::escape)
        .def_rw("multiline_input", &common_params::multiline_input)
        .def_rw("simple_io", &common_params::simple_io)
        .def_rw("cont_batching", &common_params::cont_batching)
        .def_rw("flash_attn", &common_params::flash_attn)
        .def_rw("no_perf", &common_params::flash_attn)
        .def_rw("ctx_shift", &common_params::flash_attn)
        .def_rw("input_prefix_bos", &common_params::input_prefix_bos)
        // .def_rw("ignore_eos", &common_params::ignore_eos)
        .def_rw("logits_all", &common_params::logits_all)
        .def_rw("use_mmap", &common_params::use_mmap)
        .def_rw("use_mlock", &common_params::use_mlock)
        .def_rw("verbose_prompt", &common_params::verbose_prompt)
        .def_rw("display_prompt", &common_params::display_prompt)
        // .def_rw("infill", &common_params::infill)
        .def_rw("dump_kv_cache", &common_params::dump_kv_cache)
        .def_rw("no_kv_offload", &common_params::no_kv_offload)
        .def_rw("warmup", &common_params::warmup)
        .def_rw("check_tensors", &common_params::check_tensors)
        .def_rw("cache_type_k", &common_params::cache_type_k)
        .def_rw("cache_type_v", &common_params::cache_type_v)
        .def_rw("mmproj", &common_params::mmproj)
        .def_rw("image", &common_params::image)
        .def_rw("embedding", &common_params::embedding)
        .def_rw("embd_normalize", &common_params::embd_normalize)
        .def_rw("embd_out", &common_params::embd_out)
        .def_rw("embd_sep", &common_params::embd_sep)
        .def_rw("reranking", &common_params::reranking)
        .def_rw("port", &common_params::port)
        .def_rw("timeout_read", &common_params::timeout_read)
        .def_rw("timeout_write", &common_params::timeout_write)
        .def_rw("n_threads_http", &common_params::n_threads_http)
        .def_rw("n_cache_reuse", &common_params::n_cache_reuse)
        .def_rw("hostname", &common_params::hostname)
        .def_rw("public_path", &common_params::public_path)
        .def_rw("chat_template", &common_params::chat_template)
        .def_rw("enable_chat_template", &common_params::enable_chat_template)
        .def_rw("api_keys", &common_params::api_keys)
        .def_rw("ssl_file_key", &common_params::ssl_file_key)
        .def_rw("ssl_file_cert", &common_params::ssl_file_cert)
        .def_rw("webui", &common_params::webui)
        .def_rw("endpoint_slots", &common_params::endpoint_slots)
        .def_rw("endpoint_metrics", &common_params::endpoint_metrics)
        .def_rw("endpoint_props", &common_params::endpoint_props)
        .def_rw("log_json", &common_params::log_json)
        .def_rw("slot_save_path", &common_params::slot_save_path)
        .def_rw("slot_prompt_similarity", &common_params::slot_prompt_similarity)
        .def_rw("is_pp_shared", &common_params::is_pp_shared)
        .def_rw("n_pp", &common_params::n_pp)
        .def_rw("n_tg", &common_params::n_tg)
        .def_rw("n_pl", &common_params::n_pl)
        .def_rw("context_files", &common_params::context_files)
        .def_rw("chunk_size", &common_params::chunk_size)
        .def_rw("chunk_separator", &common_params::chunk_separator)
        .def_rw("n_junk", &common_params::n_junk)
        .def_rw("i_pos", &common_params::i_pos)
        .def_rw("out_file", &common_params::out_file)
        .def_rw("n_out_freq", &common_params::n_out_freq)
        .def_rw("n_save_freq", &common_params::n_save_freq)
        .def_rw("i_chunk", &common_params::i_chunk)
        .def_rw("process_output", &common_params::process_output)
        .def_rw("compute_ppl", &common_params::compute_ppl)
        .def_rw("n_pca_batch", &common_params::n_pca_batch)
        .def_rw("n_pca_iterations", &common_params::n_pca_iterations)
        .def_rw("cvector_dimre_method", &common_params::cvector_dimre_method)
        .def_rw("cvector_outfile", &common_params::cvector_outfile)
        .def_rw("cvector_positive_file", &common_params::cvector_positive_file)
        .def_rw("cvector_negative_file", &common_params::cvector_negative_file)
        .def_rw("spm_infill", &common_params::spm_infill)
        .def_rw("lora_outfile", &common_params::lora_outfile);

    // overloaded

    m.def("common_token_to_piece", (std::string (*)(const struct llama_context *, llama_token, bool)) &common_token_to_piece, "", nb::arg("ctx"), nb::arg("token"), nb::arg("special") = true);

    m.def("common_tokenize", (std::vector<llama_token> (*)(const struct llama_context *, const std::string &, bool, bool)) &common_tokenize, "", nb::arg("ctx"), nb::arg("text"), nb::arg("add_special"), nb::arg("parse_special") = false);
    m.def("common_tokenize", (std::vector<llama_token> (*)(const struct llama_model *, const std::string &, bool, bool)) &common_tokenize, "", nb::arg("model"), nb::arg("text"), nb::arg("add_special"), nb::arg("parse_special") = false);

    // m.def("common_params_parse_from_env", (void (*)(struct common_params &)) &common_params_parse_from_env, "", nb::arg("params"));
    // m.def("common_params_handle_model_default", (void (*)(struct common_params &)) &common_params_handle_model_default, "C++: common_params_handle_model_default(struct common_params &) --> void", nb::arg("params"));
    m.def("common_params_get_system_info", (std::string (*)(const struct common_params &)) &common_params_get_system_info, "C++: common_params_get_system_info(const struct common_params &) --> std::string", nb::arg("params"));

    // m.def("string_split", (class std::vector<std::string> (*)(std::string, char)) &string_split, "C++: string_split(std::string, char) --> class std::vector<std::string>", nb::arg("input"), nb::arg("separator"));
    m.def("string_strip", (std::string (*)(const std::string &)) &string_strip, "C++: string_strip(const std::string &) --> std::string", nb::arg("str"));
    m.def("string_get_sortable_timestamp", (std::string (*)()) &string_get_sortable_timestamp, "C++: string_get_sortable_timestamp() --> std::string");
    m.def("string_parse_kv_override", (bool (*)(const char *, class std::vector<struct llama_model_kv_override> &)) &string_parse_kv_override, "C++: string_parse_kv_override(const char *, class std::vector<struct llama_model_kv_override> &) --> bool", nb::arg("data"), nb::arg("overrides"));
    m.def("string_process_escapes", (void (*)(std::string &)) &string_process_escapes, "C++: string_process_escapes(std::string &) --> void", nb::arg("input"));

    m.def("fs_validate_filename", (bool (*)(const std::string &)) &fs_validate_filename, "C++: fs_validate_filename(const std::string &) --> bool", nb::arg("filename"));
    m.def("fs_create_directory_with_parents", (bool (*)(const std::string &)) &fs_create_directory_with_parents, "C++: fs_create_directory_with_parents(const std::string &) --> bool", nb::arg("path"));
    m.def("fs_get_cache_directory", (std::string (*)()) &fs_get_cache_directory, "C++: fs_get_cache_directory() --> std::string");
    m.def("fs_get_cache_file", (std::string (*)(const std::string &)) &fs_get_cache_file, "C++: fs_get_cache_file(const std::string &) --> std::string", nb::arg("filename"));

    m.def("common_init_from_params", (class std::tuple<struct llama_model *, struct llama_context *> (*)(struct common_params &)) &common_init_from_params, "", nb::arg("params"));

    m.def("common_model_params_to_llama", (struct llama_model_params (*)(const struct common_params &)) &common_model_params_to_llama, "C++: common_model_params_to_llama(const struct common_params &) --> struct llama_model_params", nb::arg("params"));

    m.def("common_context_params_to_llama", (struct llama_context_params (*)(const struct common_params &)) &common_context_params_to_llama, "C++: common_context_params_to_llama(const struct common_params &) --> struct llama_context_params", nb::arg("params"));

    m.def("common_batch_clear", (void (*)(struct llama_batch &)) &common_batch_clear, "C++: common_batch_clear(struct llama_batch &) --> void", nb::arg("batch"));
    m.def("common_batch_add", (void (*)(struct llama_batch &, int, int, const class std::vector<int> &, bool)) &common_batch_add, "C++: common_batch_add(struct llama_batch &, int, int, const class std::vector<int> &, bool) --> void", nb::arg("batch"), nb::arg("id"), nb::arg("pos"), nb::arg("seq_ids"), nb::arg("logits"));

    nb::class_<common_chat_msg> (m, "common_chat_msg", "")
        .def(nb::init<>())
        .def_rw("role", &common_chat_msg::role)
        .def_rw("content", &common_chat_msg::content);

    m.def("common_chat_verify_template", (bool (*)(const std::string &)) &common_chat_verify_template, "C++: common_chat_verify_template(const std::string &) --> bool", nb::arg("tmpl"));

    m.def("common_kv_cache_dump_view", [](const struct llama_kv_cache_view & a0) -> void { return common_kv_cache_dump_view(a0); }, "", nb::arg("view"));
    m.def("common_kv_cache_dump_view", (void (*)(const struct llama_kv_cache_view &, int)) &common_kv_cache_dump_view, "C++: common_kv_cache_dump_view(const struct llama_kv_cache_view &, int) --> void", nb::arg("view"), nb::arg("row_size"));

    m.def("common_kv_cache_dump_view_seqs", [](const struct llama_kv_cache_view & a0) -> void { return common_kv_cache_dump_view_seqs(a0); }, "", nb::arg("view"));
    m.def("common_kv_cache_dump_view_seqs", (void (*)(const struct llama_kv_cache_view &, int)) &common_kv_cache_dump_view_seqs, "C++: common_kv_cache_dump_view_seqs(const struct llama_kv_cache_view &, int) --> void", nb::arg("view"), nb::arg("row_size"));

    m.def("common_embd_normalize", [](const float * a0, float * a1, int const & a2) -> void { return common_embd_normalize(a0, a1, a2); }, "", nb::arg("inp"), nb::arg("out"), nb::arg("n"));
    m.def("common_embd_normalize", (void (*)(const float *, float *, int, int)) &common_embd_normalize, "C++: common_embd_normalize(const float *, float *, int, int) --> void", nb::arg("inp"), nb::arg("out"), nb::arg("n"), nb::arg("embd_norm"));

    m.def("common_embd_similarity_cos", (float (*)(const float *, const float *, int)) &common_embd_similarity_cos, "C++: common_embd_similarity_cos(const float *, const float *, int) --> float", nb::arg("embd1"), nb::arg("embd2"), nb::arg("n"));

    nb::class_<common_control_vector_data> (m, "common_control_vector_data", "")
        .def(nb::init<>())
        .def_rw("n_embd", &common_control_vector_data::n_embd)
        .def_rw("data", &common_control_vector_data::data);

    nb::class_<common_control_vector_load_info> (m, "common_control_vector_load_info", "")
        .def(nb::init<>())
        .def_rw("strength", &common_control_vector_load_info::strength)
        .def_rw("fname", &common_control_vector_load_info::fname);

    m.def("common_control_vector_load", (struct common_control_vector_data (*)(const class std::vector<struct common_control_vector_load_info> &)) &common_control_vector_load, "C++: common_control_vector_load(const class std::vector<struct common_control_vector_load_info> &) --> struct common_control_vector_data", nb::arg("load_infos"));

    m.def("yaml_dump_vector_float", (void (*)(struct __sFILE *, const char *, const class std::vector<float> &)) &yaml_dump_vector_float, "C++: yaml_dump_vector_float(struct __sFILE *, const char *, const class std::vector<float> &) --> void", nb::arg("stream"), nb::arg("prop_name"), nb::arg("data"));
    m.def("yaml_dump_vector_int", (void (*)(struct __sFILE *, const char *, const class std::vector<int> &)) &yaml_dump_vector_int, "C++: yaml_dump_vector_int(struct __sFILE *, const char *, const class std::vector<int> &) --> void", nb::arg("stream"), nb::arg("prop_name"), nb::arg("data"));
    m.def("yaml_dump_string_multiline", (void (*)(struct __sFILE *, const char *, const char *)) &yaml_dump_string_multiline, "C++: yaml_dump_string_multiline(struct __sFILE *, const char *, const char *) --> void", nb::arg("stream"), nb::arg("prop_name"), nb::arg("data"));

    // -----------------------------------------------------------------------
    // sampling.h


    // -----------------------------------------------------------------------
    // arg.h

    nb::enum_<enum llama_example>(m, "llama_example", "")
        .value("LLAMA_EXAMPLE_COMMON", LLAMA_EXAMPLE_COMMON)
        .value("LLAMA_EXAMPLE_SPECULATIVE", LLAMA_EXAMPLE_SPECULATIVE)
        .value("LLAMA_EXAMPLE_MAIN", LLAMA_EXAMPLE_MAIN)
        .value("LLAMA_EXAMPLE_INFILL", LLAMA_EXAMPLE_INFILL)
        .value("LLAMA_EXAMPLE_EMBEDDING", LLAMA_EXAMPLE_EMBEDDING)
        .value("LLAMA_EXAMPLE_PERPLEXITY", LLAMA_EXAMPLE_PERPLEXITY)
        .value("LLAMA_EXAMPLE_RETRIEVAL", LLAMA_EXAMPLE_RETRIEVAL)
        .value("LLAMA_EXAMPLE_PASSKEY", LLAMA_EXAMPLE_PASSKEY)
        .value("LLAMA_EXAMPLE_IMATRIX", LLAMA_EXAMPLE_IMATRIX)
        .value("LLAMA_EXAMPLE_BENCH", LLAMA_EXAMPLE_BENCH)
        .value("LLAMA_EXAMPLE_SERVER", LLAMA_EXAMPLE_SERVER)
        .value("LLAMA_EXAMPLE_CVECTOR_GENERATOR", LLAMA_EXAMPLE_CVECTOR_GENERATOR)
        .value("LLAMA_EXAMPLE_EXPORT_LORA", LLAMA_EXAMPLE_EXPORT_LORA)
        .value("LLAMA_EXAMPLE_LLAVA", LLAMA_EXAMPLE_LLAVA)
        .value("LLAMA_EXAMPLE_LOOKUP", LLAMA_EXAMPLE_LOOKUP)
        .value("LLAMA_EXAMPLE_PARALLEL", LLAMA_EXAMPLE_PARALLEL)
        .value("LLAMA_EXAMPLE_COUNT", LLAMA_EXAMPLE_COUNT)
        .export_values();

    nb::class_<common_arg> (m, "common_arg", "")
        // .def( nb::init( [](){ return new common_arg(); } ) )
        .def_rw("examples", &common_arg::examples)
        .def_rw("args", &common_arg::args)
        .def_rw("value_hint", &common_arg::value_hint)
        .def_rw("value_hint_2", &common_arg::value_hint_2)
        .def_rw("env", &common_arg::env)
        .def_rw("help", &common_arg::help)
        .def_rw("is_sparam", &common_arg::is_sparam);
        // .def_rw("handler_void", &common_arg::handler_void)
        // .def_rw("handler_string", &common_arg::handler_string)
        // .def_rw("handler_str_str", &common_arg::handler_str_str)
        // .def_rw("handler_int", &common_arg::handler_int);

    nb::class_<common_params_context> (m, "common_params_context", "")
        .def(nb::init<common_params &>())
        .def_rw("ex", &common_params_context::ex)
        // .def_rw("params", &common_params_context::params)
        .def_rw("options", &common_params_context::options);
        // void(*print_usage)(int, char **) = nullptr;


    m.def("common_params_parse", [](std::vector<std::string> args, common_params & params, enum llama_example example) -> bool {
        void(*print_usage)(int, char **) = NULL;
        std::vector<char*> cstrings;
        cstrings.reserve(args.size());

        for(size_t i = 0; i < args.size(); ++i)
            cstrings.push_back(const_cast<char*>(args[i].c_str()));

        if (cstrings.empty()) {
            return common_params_parse(0, nullptr, params, example, print_usage);
        } else {
            return common_params_parse(cstrings.size(), &cstrings[0], params, example, print_usage);
        }
    }, "",  nb::arg("args"), nb::arg("params"), nb::arg("example"));

    m.def("common_params_parser_init", (std::vector<common_arg> (*)(common_params &, llama_example)) &common_params_parser_init, "", nb::arg("params"), nb::arg("ex"));

}
