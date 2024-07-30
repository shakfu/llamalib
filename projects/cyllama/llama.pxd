from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t


cdef extern from "ggml.h":

    ctypedef bint (*ggml_abort_callback)(void * data)

    ctypedef enum ggml_type:
        GGML_TYPE_F32     = 0
        GGML_TYPE_F16     = 1
        GGML_TYPE_Q4_0    = 2
        GGML_TYPE_Q4_1    = 3
        # // GGML_TYPE_Q4_2 = 4 support has been removed
        # // GGML_TYPE_Q4_3 = 5 support has been removed
        GGML_TYPE_Q5_0    = 6
        GGML_TYPE_Q5_1    = 7
        GGML_TYPE_Q8_0    = 8
        GGML_TYPE_Q8_1    = 9
        GGML_TYPE_Q2_K    = 10
        GGML_TYPE_Q3_K    = 11
        GGML_TYPE_Q4_K    = 12
        GGML_TYPE_Q5_K    = 13
        GGML_TYPE_Q6_K    = 14
        GGML_TYPE_Q8_K    = 15
        GGML_TYPE_IQ2_XXS = 16
        GGML_TYPE_IQ2_XS  = 17
        GGML_TYPE_IQ3_XXS = 18
        GGML_TYPE_IQ1_S   = 19
        GGML_TYPE_IQ4_NL  = 20
        GGML_TYPE_IQ3_S   = 21
        GGML_TYPE_IQ2_S   = 22
        GGML_TYPE_IQ4_XS  = 23
        GGML_TYPE_I8      = 24
        GGML_TYPE_I16     = 25
        GGML_TYPE_I32     = 26
        GGML_TYPE_I64     = 27
        GGML_TYPE_F64     = 28
        GGML_TYPE_IQ1_M   = 29
        GGML_TYPE_BF16    = 30
        GGML_TYPE_Q4_0_4_4 = 31
        GGML_TYPE_Q4_0_4_8 = 32
        GGML_TYPE_Q4_0_8_8 = 33
        GGML_TYPE_COUNT

    ctypedef enum ggml_op:
        GGML_OP_NONE = 0

        GGML_OP_DUP
        GGML_OP_ADD
        GGML_OP_ADD1
        GGML_OP_ACC
        GGML_OP_SUB
        GGML_OP_MUL
        GGML_OP_DIV
        GGML_OP_SQR
        GGML_OP_SQRT
        GGML_OP_LOG
        GGML_OP_SUM
        GGML_OP_SUM_ROWS
        GGML_OP_MEAN
        GGML_OP_ARGMAX
        GGML_OP_REPEAT
        GGML_OP_REPEAT_BACK
        GGML_OP_CONCAT
        GGML_OP_SILU_BACK
        GGML_OP_NORM # normalize
        GGML_OP_RMS_NORM
        GGML_OP_RMS_NORM_BACK
        GGML_OP_GROUP_NORM

        GGML_OP_MUL_MAT
        GGML_OP_MUL_MAT_ID
        GGML_OP_OUT_PROD

        GGML_OP_SCALE
        GGML_OP_SET
        GGML_OP_CPY
        GGML_OP_CONT
        GGML_OP_RESHAPE
        GGML_OP_VIEW
        GGML_OP_PERMUTE
        GGML_OP_TRANSPOSE
        GGML_OP_GET_ROWS
        GGML_OP_GET_ROWS_BACK
        GGML_OP_DIAG
        GGML_OP_DIAG_MASK_INF
        GGML_OP_DIAG_MASK_ZERO
        GGML_OP_SOFT_MAX
        GGML_OP_SOFT_MAX_BACK
        GGML_OP_ROPE
        GGML_OP_ROPE_BACK
        GGML_OP_CLAMP
        GGML_OP_CONV_TRANSPOSE_1D
        GGML_OP_IM2COL
        GGML_OP_CONV_TRANSPOSE_2D
        GGML_OP_POOL_1D
        GGML_OP_POOL_2D
        GGML_OP_UPSCALE # nearest interpolate
        GGML_OP_PAD
        GGML_OP_ARANGE
        GGML_OP_TIMESTEP_EMBEDDING
        GGML_OP_ARGSORT
        GGML_OP_LEAKY_RELU

        GGML_OP_FLASH_ATTN_EXT
        GGML_OP_FLASH_ATTN_BACK
        GGML_OP_SSM_CONV
        GGML_OP_SSM_SCAN
        GGML_OP_WIN_PART
        GGML_OP_WIN_UNPART
        GGML_OP_GET_REL_POS
        GGML_OP_ADD_REL_POS

        GGML_OP_UNARY

        GGML_OP_MAP_UNARY
        GGML_OP_MAP_BINARY

        GGML_OP_MAP_CUSTOM1_F32
        GGML_OP_MAP_CUSTOM2_F32
        GGML_OP_MAP_CUSTOM3_F32

        GGML_OP_MAP_CUSTOM1
        GGML_OP_MAP_CUSTOM2
        GGML_OP_MAP_CUSTOM3

        GGML_OP_CROSS_ENTROPY_LOSS
        GGML_OP_CROSS_ENTROPY_LOSS_BACK

        GGML_OP_COUNT



    ctypedef struct ggml_backend_buffer

    DEF GGML_MAX_DIMS = 4
    DEF GGML_MAX_NAME = 64
    DEF GGML_MAX_OP_PARAMS = 64
    DEF GGML_MAX_SRC = 10

    # n-dimensional tensor
    ctypedef struct ggml_tensor:
        ggml_type type

        # GGML_DEPRECATED(enum ggml_backend_type backend, "use the buffer type to find the storage location of the tensor")

        ggml_backend_buffer * buffer

        int64_t ne[GGML_MAX_DIMS]  # number of elements
        size_t  nb[GGML_MAX_DIMS]  # stride in bytes:
                                   # nb[0] = ggml_type_size(type)
                                   # nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   # nb[i] = nb[i-1] * ne[i-1]

        # compute data
        ggml_op op

        # op params - allocated as int32_t for alignment
        # int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]
        int32_t op_params[16] # 64 / 4

        int32_t flags

        ggml_tensor * grad
        ggml_tensor * src[GGML_MAX_SRC]

        # source tensor and offset for views
        ggml_tensor * view_src
        size_t               view_offs

        void * data

        char name[GGML_MAX_NAME]

        void * extra # extra things e.g. for ggml-cuda.cu

        # char padding[4]


cdef extern from "ggml-backend.h":
    ctypedef bint (*ggml_backend_sched_eval_callback)(ggml_tensor * t, bint ask, void * user_data)



cdef extern from "llama.h":
    ctypedef struct llama_model
    ctypedef struct llama_context

    ctypedef int32_t llama_pos
    ctypedef int32_t llama_token
    ctypedef int32_t llama_seq_id

    ctypedef enum llama_vocab_type:
        LLAMA_VOCAB_TYPE_NONE
        LLAMA_VOCAB_TYPE_SPM
        LLAMA_VOCAB_TYPE_BPE
        LLAMA_VOCAB_TYPE_WPM
        LLAMA_VOCAB_TYPE_UGM

    ctypedef enum llama_vocab_pre_type:
        LLAMA_VOCAB_PRE_TYPE_DEFAULT
        LLAMA_VOCAB_PRE_TYPE_LLAMA3
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER
        LLAMA_VOCAB_PRE_TYPE_FALCON
        LLAMA_VOCAB_PRE_TYPE_MPT
        LLAMA_VOCAB_PRE_TYPE_STARCODER
        LLAMA_VOCAB_PRE_TYPE_GPT2
        LLAMA_VOCAB_PRE_TYPE_REFACT
        LLAMA_VOCAB_PRE_TYPE_COMMAND_R
        LLAMA_VOCAB_PRE_TYPE_STABLELM2
        LLAMA_VOCAB_PRE_TYPE_QWEN2
        LLAMA_VOCAB_PRE_TYPE_OLMO
        LLAMA_VOCAB_PRE_TYPE_DBRX
        LLAMA_VOCAB_PRE_TYPE_SMAUG
        LLAMA_VOCAB_PRE_TYPE_PORO
        LLAMA_VOCAB_PRE_TYPE_CHATGLM3
        LLAMA_VOCAB_PRE_TYPE_CHATGLM4
        LLAMA_VOCAB_PRE_TYPE_VIKING
        LLAMA_VOCAB_PRE_TYPE_JAIS
        LLAMA_VOCAB_PRE_TYPE_TEKKEN
        LLAMA_VOCAB_PRE_TYPE_SMOLLM
        LLAMA_VOCAB_PRE_TYPE_CODESHELL

    ctypedef enum llama_rope_type:
        LLAMA_ROPE_TYPE_NONE = -1
        LLAMA_ROPE_TYPE_NORM =  0
        LLAMA_ROPE_TYPE_NEOX =  2
        LLAMA_ROPE_TYPE_GLM  =  4

    ctypedef enum llama_token_type:
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0
        LLAMA_TOKEN_TYPE_NORMAL       = 1
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2
        LLAMA_TOKEN_TYPE_CONTROL      = 3
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4
        LLAMA_TOKEN_TYPE_UNUSED       = 5
        LLAMA_TOKEN_TYPE_BYTE         = 6

    ctypedef enum llama_token_attr:
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

    ctypedef enum llama_ftype:
        LLAMA_FTYPE_ALL_F32              = 0
        LLAMA_FTYPE_MOSTLY_F16           = 1
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3
        # LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4  # tok_embeddings.weight and output.weight are F16
        # LLAMA_FTYPE_MOSTLY_Q4_2       = 5     # support has been removed
        # LLAMA_FTYPE_MOSTLY_Q4_3       = 6     # support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31
        LLAMA_FTYPE_MOSTLY_BF16          = 32
        LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33
        LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34
        LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35
        LLAMA_FTYPE_GUESSED              = 1024

    ctypedef enum llama_rope_scaling_type:
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = 2

    ctypedef enum llama_pooling_type:
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1
        LLAMA_POOLING_TYPE_NONE = 0
        LLAMA_POOLING_TYPE_MEAN = 1
        LLAMA_POOLING_TYPE_CLS  = 2
        LLAMA_POOLING_TYPE_LAST = 3

    ctypedef enum llama_attention_type:
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

    ctypedef enum llama_split_mode:
        LLAMA_SPLIT_MODE_NONE    = 0
        LLAMA_SPLIT_MODE_LAYER   = 1
        LLAMA_SPLIT_MODE_ROW     = 2

    ctypedef struct llama_token_data:
        llama_token id
        float logit
        float p

    ctypedef struct llama_token_data_array:
        llama_token_data * data
        size_t size
        bint sorted

    ctypedef bint (*llama_progress_callback)(float progress, void * user_data)

    ctypedef struct llama_batch:
        int32_t n_tokens

        llama_token  *  token
        float        *  embd
        llama_pos    *  pos
        int32_t      *  n_seq_id
        llama_seq_id ** seq_id
        int8_t       *  logits   # TODO: rename this to "output"

        llama_pos    all_pos_0  # used if pos == NULL
        llama_pos    all_pos_1  # used if pos == NULL
        llama_seq_id all_seq_id # used if seq_id == NULL

    ctypedef enum llama_model_kv_override_type:
        LLAMA_KV_OVERRIDE_TYPE_INT
        LLAMA_KV_OVERRIDE_TYPE_FLOAT
        LLAMA_KV_OVERRIDE_TYPE_BOOL
        LLAMA_KV_OVERRIDE_TYPE_STR

    ctypedef struct llama_model_kv_override: # FLATTENED nested union enum
        llama_model_kv_override_type tag
        char key[128]
        int64_t val_i64
        double  val_f64
        bint    val_bool
        char    val_str[128]

    ctypedef struct llama_model_params:
        int32_t n_gpu_layers
        llama_split_mode split_mode
        int32_t main_gpu
        const float * tensor_split
        const char * rpc_servers
        llama_progress_callback progress_callback
        void * progress_callback_user_data
        const llama_model_kv_override * kv_overrides
        bint vocab_only
        bint use_mmap
        bint use_mlock
        bint check_tensors

    ctypedef struct llama_context_params:
        uint32_t seed              # RNG seed, -1 for random
        uint32_t n_ctx             # text context, 0 = from model
        uint32_t n_batch           # logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch          # physical maximum batch size
        uint32_t n_seq_max         # max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_threads         # number of threads to use for generation
        uint32_t n_threads_batch   # number of threads to use for batch processing

        llama_rope_scaling_type rope_scaling_type # RoPE scaling type
        llama_pooling_type      pooling_type      # whether to pool (sum) embedding results by sequence id
        llama_attention_type    attention_type    # attention type to use for embeddings

        float    rope_freq_base   # RoPE base frequency, 0 = from model
        float    rope_freq_scale  # RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor  # YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor # YaRN magnitude scaling factor
        float    yarn_beta_fast   # YaRN low correction dim
        float    yarn_beta_slow   # YaRN high correction dim
        uint32_t yarn_orig_ctx    # YaRN original context size
        float    defrag_thold     # defragment the KV cache if holes/size > thold, < 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml_type type_k # data type for K cache [EXPERIMENTAL]
        ggml_type type_v # data type for V cache [EXPERIMENTAL]

        # Keep the booleans together to avoid misalignment during copy-by-value.
        bint logits_all  # the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
        bint embeddings  # if true, extract embeddings (together with logits)
        bint offload_kqv # whether to offload the KQV ops (including the KV cache) to GPU
        bint flash_attn  # whether to use flash attention [EXPERIMENTAL]

        # Abort callback
        # if it returns true, execution of llama_decode() will be aborted
        # currently works only with CPU execution
        ggml_abort_callback abort_callback
        void * abort_callback_data


    ctypedef struct llama_model_quantize_params:
        int32_t nthread                     # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        llama_ftype ftype                   # quantize to this llama_ftype
        ggml_type output_tensor_type        # output tensor type
        ggml_type token_embedding_type      # itoken embeddings tensor type
        bint allow_requantize               # allow quantizing non-f32/f16 tensors
        bint quantize_output_tensor         # quantize output.weight
        bint only_copy                      # only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bint pure                           # quantize all tensors to the default type
        bint keep_split                     # quantize to the same number of shards
        void * imatrix                      # pointer to importance matrix data
        void * kv_overrides                 # pointer to vector containing overrides

    ctypedef struct llama_grammar

    ctypedef enum llama_gretype:
        LLAMA_GRETYPE_END            = 0
        LLAMA_GRETYPE_ALT            = 1
        LLAMA_GRETYPE_RULE_REF       = 2
        LLAMA_GRETYPE_CHAR           = 3
        LLAMA_GRETYPE_CHAR_NOT       = 4
        LLAMA_GRETYPE_CHAR_RNG_UPPER = 5
        LLAMA_GRETYPE_CHAR_ALT       = 6
        LLAMA_GRETYPE_CHAR_ANY       = 7

    ctypedef struct llama_grammar_element:
        llama_gretype type
        uint32_t value

    ctypedef struct llama_timings:
        double t_start_ms
        double t_end_ms
        double t_load_ms
        double t_sample_ms
        double t_p_eval_ms
        double t_eval_ms

        int32_t n_sample
        int32_t n_p_eval
        int32_t n_eval

    ctypedef struct llama_chat_message:
        const char * role
        const char * content

    ctypedef struct llama_lora_adapter


    # -------------------------------------------------------------------------
    # functions

    cdef llama_model_params llama_model_default_params()
    cdef llama_context_params llama_context_default_params()
    cdef llama_model_quantize_params llama_model_quantize_default_params()



