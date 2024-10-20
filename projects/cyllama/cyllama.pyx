# distutils: language = c++
from libc.stdint cimport uint8_t, int32_t
from libc.stdlib cimport malloc, calloc, realloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport llama_cpp

import os
from typing import Optional, Sequence


# enums
# -----------------------------------------------------------------------------

cpdef enum llama_vocab_type:
    LLAMA_VOCAB_TYPE_NONE # For models without vocab
    LLAMA_VOCAB_TYPE_SPM  # LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  # GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  # BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  # T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV # RWKV tokenizer based on greedy tokenization

cpdef enum llama_rope_type:
    LLAMA_ROPE_TYPE_NONE = -1
    LLAMA_ROPE_TYPE_NORM =  0
    LLAMA_ROPE_TYPE_NEOX =  2

cpdef enum llama_token_attr:
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3 # SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9


# llama-cpp wrappers
# -----------------------------------------------------------------------------



def ask(str prompt, str model, n_predict=512, n_ctx=2048, disable_log=True, n_threads=4) -> str:
    """ask/prompt a llama model"""

    cdef str result = llama_cpp.simple_prompt(
        model.encode(),
        prompt.encode(),
        n_predict,
        n_ctx,
        disable_log,
        n_threads).decode()
    return result.strip()


cdef class LoraAdapter:
    cdef llama_cpp.llama_lora_adapter * ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self.ptr = NULL
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.ptr_owner is True:
            llama_cpp.llama_lora_adapter_free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef LoraAdapter from_ptr(llama_cpp.llama_lora_adapter *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef LoraAdapter wrapper = LoraAdapter.__new__(LoraAdapter)
        wrapper.ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper


cdef class GGMLTensor:
    cdef llama_cpp.ggml_tensor * ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self.ptr = NULL
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.ptr_owner is True:
            free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    # Get a llama model tensor
    # struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name);


    @staticmethod
    cdef GGMLTensor from_ptr(llama_cpp.ggml_tensor *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef GGMLTensor wrapper = GGMLTensor.__new__(GGMLTensor)
        wrapper.ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef GGMLTensor create():
        cdef llama_cpp.ggml_tensor *ptr = <llama_cpp.ggml_tensor *>malloc(sizeof(llama_cpp.ggml_tensor))
        if ptr is NULL:
            raise MemoryError
        # ptr.a = 0
        # ptr.b = 0
        return GGMLTensor.from_ptr(ptr, owner=True)


cdef class SamplerChainParams:
    cdef llama_cpp.llama_sampler_chain_params p

    def __init__(self):
        self.p = llama_cpp.llama_sampler_chain_default_params()

    @staticmethod
    cdef SamplerChainParams from_instance(llama_cpp.llama_sampler_chain_params params):
        cdef SamplerChainParams wrapper = SamplerChainParams.__new__(SamplerChainParams)
        wrapper.p = params
        return wrapper

    @property
    def no_perf(self) -> bool:
        """whether to measure performance timings."""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value


cdef class LlamaSampler:
    """cython wrapper for llama_cpp.llama_sampler."""
    cdef llama_cpp.llama_sampler * ptr
    cdef SamplerChainParams params
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, params: Optional[SamplerChainParams] = None):
        if not params:
            self.ptr = llama_cpp.llama_sampler_chain_init(self.params.p)
        else:
            self.ptr = llama_cpp.llama_sampler_chain_init(params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to init Sampler")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_sampler_free(self.ptr)
            self.ptr = NULL

    def name(self) -> str:
        """Get sampler name"""
        return llama_cpp.llama_sampler_name(self.ptr).decode()

    def accept(self, llama_cpp.llama_token token):
        """Accept llama token"""
        llama_cpp.llama_sampler_accept(self.ptr, token)

    # cdef void llama_sampler_apply (llama_sampler * smpl, llama_token_data_array * cur_p)
    
    def reset(self):
        """Reset sampler"""
        llama_cpp.llama_sampler_reset(self.ptr)

    def clone(self) -> LlamaSampler:
        """clone sampler"""
        cdef llama_cpp.llama_sampler * smplr = llama_cpp.llama_sampler_clone(self.ptr)
        cdef LlamaSampler wrapper = LlamaSampler.__new__(LlamaSampler)
        wrapper.ptr = smplr
        return wrapper

    def get_seed(self) -> int:
        """Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise"""
        return llama_cpp.llama_sampler_get_seed(self.ptr)

    def add_greedy(self):
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_greedy())

    def add_dist(self, llama_cpp.uint32_t seed):
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_dist(seed))

    def add_softmax(self):
        """Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_softmax())

    def add_top_k(self, int32_t k):
        """Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_top_k(k))

    def add_top_p(self, float p, size_t min_keep):
        """Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_top_p(p, min_keep))

    def add_min_p(self, float p, size_t min_keep):
        """Minimum P sampling as described in https:#github.com/ggerganov/llama.cpp/pull/3841"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_min_p(p, min_keep))

    def add_tail_free(self, float z, size_t min_keep):
        """Tail Free Sampling described in https:#www.trentonbricken.com/Tail-Free-Sampling/."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_tail_free(z, min_keep))

    # XXX: should add_typical + add_temp be combined?
    def add_typical(self, float p, size_t min_keep):
        """Locally Typical Sampling implementation described in the paper https:#arxiv.org/abs/2202.00666."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_typical(p, min_keep))

    def add_temp(self, float t):
        """Locally Typical Sampling implementation described in the paper https:#arxiv.org/abs/2202.00666."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_temp(t))

    def add_temp_ext(self, float t, float delta, float exponent):
        """Dynamic temperature implementation described in the paper https:#arxiv.org/abs/2309.02772."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_temp_ext(t, delta, exponent))

    def add_xtc(self, float p, float t, size_t min_keep, llama_cpp.uint32_t seed):
        """XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_xtc(p, t, min_keep, seed))

    # XXX: docstring incorrect
    def add_mirostat(self, int n_vocab, llama_cpp.uint32_t seed, float tau, float eta, int m):
        """Mirostat 1.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.
    
        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau:     The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta:     The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        m:       The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
        mu:      Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
        """
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m))

    def add_mirostat_v2(self, llama_cpp.uint32_t seed, float tau, float eta):
        """Mirostat 2.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.
        
        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau:  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        mu: Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
        """
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_mirostat_v2(seed, tau, eta))

    def add_grammar(self, LlamaModel model, str grammar_str, str grammar_root):
        """Add grammer chain link"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_grammar(
                model.ptr, grammar_str.encode(), grammar_root.encode()))

    def add_penalties(self,      int n_vocab,         # llama_n_vocab()
               llama_cpp.llama_token special_eos_id,  # llama_token_eos()
               llama_cpp.llama_token linefeed_id,     # llama_token_nl()
                                 int penalty_last_n,  # last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float penalty_repeat,  # 1.0 = disabled
                               float penalty_freq,    # 0.0 = disabled
                               float penalty_present, # 0.0 = disabled
                                bint penalize_nl,     # consider newlines as a repeatable token
                                bint ignore_eos):     # ignore the end-of-sequence token

        """Add penalties chain link"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_penalties(
                n_vocab, 
                special_eos_id,
                linefeed_id,
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
                penalize_nl,
                ignore_eos,
            ))

    # XXX FIXME:
    # def add_logit_bias(self, int n_vocab, int n_logit_bias, logit_bias: list[LogitBias]):
    #     """Add grammer chain link"""
    #     cdef vector[llama_cpp.logit_bias] vec
    #     llama_cpp.llama_sampler_chain_add(
    #         self.ptr, llama_cpp.llama_sampler_init_logit_bias(
    #             n_vocab, n_logit_bias, vec.data()))

    def add_infill(self, LlamaModel model):
        """This sampler is meant to be used for fill-in-the-middle infilling
        
        it's supposed to be used after top_k + top_p sampling
        
        1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
        2. combine probs of tokens that have the same prefix
        
        example:
        
        - before:
          "hel":   0.5
          "hell":  0.2
          "hello": 0.1
          "dummy": 0.1
        
        - after:
          "hel":   0.8
          "dummy": 0.1
        
        3. discard non-EOG tokens with low prob
        4. if no tokens are left -> pick EOT
        """
        llama_cpp.llama_sampler_chain_add(self.ptr,
            llama_cpp.llama_sampler_init_infill(model.ptr))


    def sample(self, LlamaContext ctx, int idx) -> int:
        """Sample and accept a token from the idx-th output of the last evaluation
        
        Shorthand for:
        
           const auto * logits = llama_get_logits_ith(ctx, idx)
           llama_token_data_array cur_p = { ... init from logits ... }
           llama_sampler_apply(smpl, &cur_p)
           return cur_p.data[cur_p.selected].id
        
        At this point, this is mostly a convenience function.
        """
        return llama_cpp.llama_sampler_sample(self.ptr, ctx.ptr, idx)



    # @staticmethod
    # cdef LlamaSampler init_greedy():
    #     cdef LlamaSampler wrapper = LlamaSampler.__new__(LlamaSampler)
    #     wrapper.ptr = llama_cpp.llama_sampler_init_greedy()
    #     return wrapper

    # def chain_add(self, smplr: LlamaSampler):
    #     smplr.owner = False
    #     llama_cpp.llama_sampler_chain_add(self.ptr, smplr.ptr)

    # def chain_add_greedy(self):
    #     self.chain_add(LlamaSampler.init_greedy())


cdef class CpuParams:
    cdef llama_cpp.cpu_params *ptr
    cdef object owner

    @staticmethod
    cdef CpuParams from_ptr(llama_cpp.cpu_params *p, object parent):
        cdef CpuParams wrapper = CpuParams.__new__(CpuParams)
        wrapper.ptr = p
        wrapper.owner = parent
        return wrapper

    @property
    def n_threads(self) -> int:
        """number of threads."""
        return self.ptr.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.ptr.n_threads = value

    # @property
    # def cpumask(self) -> list[bool]:
    #     """CPU affinity mask."""
    #     return self.ptr.cpumask

    # @cpumask.setter
    # def cpumask(self, value: list[bool]):
    #     self.ptr.cpumask = value

    @property
    def mask_valid(self) -> bool:
        """Default: any CPU."""
        return self.ptr.mask_valid

    @mask_valid.setter
    def mask_valid(self, value: bool):
        self.ptr.mask_valid = value

    @property
    def priority(self) -> llama_cpp.ggml_sched_priority:
        """Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)."""
        return self.ptr.priority

    @priority.setter
    def priority(self, value: llama_cpp.ggml_sched_priority):
        self.ptr.priority = value

    # @property
    # def llama_cpp.ggml_sched_strict_cpu strict_cpu(self):
    #     """Use strict CPU placement."""
    #     return self.ptr.strict_cpu

    # @strict_cpu.setter
    # def strict_cpu(self, llama_cpp.ggml_sched_strict_cpu value):
    #     self.ptr.strict_cpu = value

    # @property
    # def poll(self) -> llama_cpp.ggml_sched_poll:
    #     """Use strict CPU placement"""
    #     return self.ptr.poll

    # @poll.setter
    # def poll(self, value: llama_cpp.ggml_sched_poll):
    #     self.ptr.poll = value


cdef class CommonParams:
    cdef llama_cpp.common_params p
    cdef public CpuParams cpuparams
    cdef public CpuParams cpuparams_batch
    cdef public CpuParams draft_cpuparams
    cdef public CpuParams draft_cpuparams_batch

    @staticmethod
    cdef CommonParams from_instance(llama_cpp.common_params p):
        cdef CommonParams wrapper = CommonParams.__new__(CommonParams)
        wrapper.p = p
        wrapper.cpuparams = CpuParams.from_ptr(&wrapper.p.cpuparams, wrapper)
        wrapper.cpuparams_batch = CpuParams.from_ptr(&wrapper.p.cpuparams_batch, wrapper)
        wrapper.draft_cpuparams = CpuParams.from_ptr(&wrapper.p.draft_cpuparams, wrapper)
        wrapper.draft_cpuparams_batch = CpuParams.from_ptr(&wrapper.p.draft_cpuparams_batch, wrapper)
        return wrapper

    def __cinit__(self):
        self.cpuparams = CpuParams.from_ptr(&self.p.cpuparams, self)
        self.cpuparams_batch = CpuParams.from_ptr(&self.p.cpuparams_batch, self)
        self.draft_cpuparams = CpuParams.from_ptr(&self.p.draft_cpuparams, self)
        self.draft_cpuparams_batch = CpuParams.from_ptr(&self.p.draft_cpuparams_batch, self)

    @property
    def n_predict(self) -> int:
        """new tokens to predict."""
        return self.p.n_predict

    @n_predict.setter
    def n_predict(self, value: int):
        self.p.n_predict = value

    @property
    def n_ctx(self) -> int:
        """context size."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_batch(self) -> int:
        """logical batch size for prompt processing (must be >=32)."""
        return self.p.n_batch

    @n_batch.setter
    def n_batch(self, value: int):
        self.p.n_batch = value

    @property
    def n_ubatch(self) -> int:
        """physical batch size for prompt processing (must be >=32)."""
        return self.p.n_ubatch

    @n_ubatch.setter
    def n_ubatch(self, value: int):
        self.p.n_ubatch = value

    @property
    def n_keep(self) -> int:
        """number of tokens to keep from initial prompt."""
        return self.p.n_keep

    @n_keep.setter
    def n_keep(self, value: int):
        self.p.n_keep = value

    @property
    def n_draft(self) -> int:
        """number of tokens to draft during speculative decoding."""
        return self.p.n_draft

    @n_draft.setter
    def n_draft(self, value: int):
        self.p.n_draft = value

    @property
    def n_chunks(self) -> int:
        """max number of chunks to process (-1 = unlimited)."""
        return self.p.n_chunks

    @n_chunks.setter
    def n_chunks(self, value: int):
        self.p.n_chunks = value

    @property
    def n_parallel(self) -> int:
        """number of parallel sequences to decode."""
        return self.p.n_parallel

    @n_parallel.setter
    def n_parallel(self, value: int):
        self.p.n_parallel = value

    @property
    def n_sequences(self) -> int:
        """number of sequences to decode."""
        return self.p.n_sequences

    @n_sequences.setter
    def n_sequences(self, value: int):
        self.p.n_sequences = value

    @property
    def p_split(self) -> float:
        """speculative decoding split probability."""
        return self.p.p_split

    @p_split.setter
    def p_split(self, value: float):
        self.p.p_split = value

    @property
    def n_gpu_layers(self) -> int:
        """number of layers to store in VRAM (-1 - use default)."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def n_gpu_layers_draft(self) -> int:
        """number of layers to store in VRAM for the draft model (-1 - use default)."""
        return self.p.n_gpu_layers_draft

    @n_gpu_layers_draft.setter
    def n_gpu_layers_draft(self, value: int):
        self.p.n_gpu_layers_draft = value

    @property
    def tensor_split(self) -> list[float]:
        """how split tensors should be distributed across GPUs."""
        result = []
        for i in range(128):
            result.append(self.p.tensor_split[i])
        return result

    @tensor_split.setter
    def tensor_split(self, value: list[float]):
        assert len(value) == 128, "tensor must of length 128"
        for i in range(128):
            self.p.tensor_split[i] = value[i]

    @property
    def grp_attn_n(self) -> int:
        """group-attention factor."""
        return self.p.grp_attn_n

    @grp_attn_n.setter
    def grp_attn_n(self, value: int):
        self.p.grp_attn_n = value

    @property
    def grp_attn_w(self) -> int:
        """group-attention width."""
        return self.p.grp_attn_w

    @grp_attn_w.setter
    def grp_attn_w(self, value: int):
        self.p.grp_attn_w = value

    @property
    def n_print(self) -> int:
        """print token count every n tokens (-1 = disabled)."""
        return self.p.n_print

    @n_print.setter
    def n_print(self, value: int):
        self.p.n_print = value

    @property
    def rope_freq_base(self) -> float:
        """RoPE base frequency."""
        return self.p.rope_freq_base

    @rope_freq_base.setter
    def rope_freq_base(self, value: float):
        self.p.rope_freq_base = value

    @property
    def rope_freq_scale(self) -> float:
        """RoPE frequency scaling factor."""
        return self.p.rope_freq_scale

    @rope_freq_scale.setter
    def rope_freq_scale(self, value: float):
        self.p.rope_freq_scale = value

    @property
    def yarn_ext_factor(self) -> float:
        """YaRN extrapolation mix factor."""
        return self.p.yarn_ext_factor

    @yarn_ext_factor.setter
    def yarn_ext_factor(self, value: float):
        self.p.yarn_ext_factor = value

    @property
    def yarn_attn_factor(self) -> float:
        """YaRN magnitude scaling factor."""
        return self.p.yarn_attn_factor

    @yarn_attn_factor.setter
    def yarn_attn_factor(self, value: float):
        self.p.yarn_attn_factor = value

    @property
    def yarn_beta_fast(self) -> float:
        """YaRN low correction dim."""
        return self.p.yarn_beta_fast

    @yarn_beta_fast.setter
    def yarn_beta_fast(self, value: float):
        self.p.yarn_beta_fast = value

    @property
    def yarn_beta_slow(self) -> float:
        """YaRN high correction dim."""
        return self.p.yarn_beta_slow

    @yarn_beta_slow.setter
    def yarn_beta_slow(self, value: float):
        self.p.yarn_beta_slow = value


    @property
    def yarn_orig_ctx(self) -> int:
        """YaRN original context length."""
        return self.p.yarn_orig_ctx

    @yarn_orig_ctx.setter
    def yarn_orig_ctx(self, value: int):
        self.p.yarn_orig_ctx = value

    @property
    def defrag_thold(self) -> float:
        """KV cache defragmentation threshold."""
        return self.p.defrag_thold

    @defrag_thold.setter
    def defrag_thold(self, value: float):
        self.p.defrag_thold = value

    # @property
    # def cpuparams(self) -> CpuParams:
    #     """cpuparams instance."""
    #     return CpuParams.from_instance(self.p.cpuparams)

    # @property
    # def cpuparams_batch(self) -> CpuParams:
    #     """cpuparams_batch instance."""
    #     return CpuParams.from_instance(self.p.cpuparams_batch)

    # @property
    # def draft_cpuparams(self) -> CpuParams:
    #     """draft_cpuparams instance."""
    #     return CpuParams.from_instance(self.p.draft_cpuparams)

    # @property
    # def draft_cpuparams_batch(self) -> CpuParams:
    #     """draft_cpuparams_batch instance."""
    #     return CpuParams.from_instance(self.p.draft_cpuparams_batch)

    # @property
    # def cb_eval(self) -> llama_cpp.ggml_backend_sched_eval_callback:
    #     """ggml backend sched eval callback."""
    #     return self.p.cb_eval

    # @cb_eval.setter
    # def cb_eval(self, value: llama_cpp.ggml_backend_sched_eval_callback):
    #     self.p.cb_eval = value

    # @property
    # def cb_eval_user_data(self):
    #     """cb eval user data."""
    #     return self.p.cb_eval_user_data

    # @cb_eval_user_data.setter
    # def cb_eval_user_data(self, value):
    #     self.p.cb_eval_user_data = value

    @property
    def numa(self) -> llama_cpp.ggml_numa_strategy:
        """KV cache defragmentation threshold."""
        return self.p.numa

    @numa.setter
    def numa(self, value: llama_cpp.ggml_numa_strategy):
        self.p.numa = value

    @property
    def split_mode(self) -> llama_cpp.llama_split_mode:
        """how to split the model across GPUs."""
        return self.p.split_mode

    @split_mode.setter
    def split_mode(self, value: llama_cpp.llama_split_mode):
        self.p.split_mode = value

    # @property
    # def rope_scaling_type(self) -> llama_cpp.llama_rope_scaling_type:
    #     """rope scaling type."""
    #     return self.p.rope_scaling_type

    # @rope_scaling_type.setter
    # def rope_scaling_type(self, value: llama_cpp.rope_scaling_type):
    #     self.p.rope_scaling_type = value

    # @property
    # def pooling_type(self) -> llama_cpp.llama_pooling_type:
    #     """pooling type for embeddings."""
    #     return self.p.pooling_type

    # @pooling_type.setter
    # def pooling_type(self, value: llama_cpp.llama_pooling_type):
    #     self.p.pooling_type = value

    # @property
    # def attention_type(self) -> llama_cpp.llama_attention_type:
    #     """attention type for embeddings."""
    #     return self.p.attention_type

    # @attention_type.setter
    # def attention_type(self, value: llama_cpp.llama_attention_type):
    #     self.p.attention_type = value

    @property
    def sparams(self) -> llama_cpp.common_sampler_params:
        """gpt sampler params."""
        return self.p.sparams

    @sparams.setter
    def sparams(self, value: llama_cpp.common_sampler_params):
        self.p.sparams = value

    @property
    def model(self) -> str:
        """model path"""
        return self.p.model.decode()

    @model.setter
    def model(self, value: str):
        self.p.model = value.encode('utf8')

    @property
    def model_draft(self) -> str:
        """draft model for speculative decoding"""
        return self.p.model_draft.decode()

    @model_draft.setter
    def model_draft(self, value: str):
        self.p.model_draft = value.encode('utf8')

    @property
    def model_alias(self) -> str:
        """model alias"""
        return self.p.model_alias.decode()

    @model_alias.setter
    def model_alias(self, value: str):
        self.p.model_alias = value.encode('utf8')

    @property
    def model_url(self) -> str:
        """model url to download """
        return self.p.model_url.decode()

    @model_url.setter
    def model_url(self, value: str):
        self.p.model_url = value.encode('utf8')

    @property
    def hf_token(self) -> str:
        """hf token"""
        return self.p.hf_token.decode()

    @hf_token.setter
    def hf_token(self, value: str):
        self.p.hf_token = value.encode('utf8')

    @property
    def hf_repo(self) -> str:
        """hf repo"""
        return self.p.hf_repo.decode()

    @hf_repo.setter
    def hf_repo(self, value: str):
        self.p.hf_repo = value.encode('utf8')

    @property
    def hf_file(self) -> str:
        """hf file"""
        return self.p.hf_file.decode()

    @hf_file.setter
    def hf_file(self, value: str):
        self.p.hf_file = value.encode('utf8')

    @property
    def prompt(self) -> str:
        """the prompt text"""
        return self.p.prompt.decode()

    @prompt.setter
    def prompt(self, value: str):
        self.p.prompt = value.encode('utf8')

    @property
    def prompt_file(self) -> str:
        """store the external prompt file name"""
        return self.p.prompt_file.decode()

    @prompt_file.setter
    def prompt_file(self, value: str):
        self.p.prompt_file = value.encode('utf8')

    @property
    def path_prompt_cache(self) -> str:
        """path to file for saving/loading prompt eval state"""
        return self.p.path_prompt_cache.decode()

    @path_prompt_cache.setter
    def path_prompt_cache(self, value: str):
        self.p.path_prompt_cache = value.encode('utf8')

    @property
    def input_prefix(self) -> str:
        """string to prefix user inputs with"""
        return self.p.input_prefix.decode()

    @input_prefix.setter
    def input_prefix(self, value: str):
        self.p.input_prefix = value.encode('utf8')

    @property
    def input_suffix(self) -> str:
        """string to suffix user inputs with"""
        return self.p.input_suffix.decode()

    @input_suffix.setter
    def input_suffix(self, value: str):
        self.p.input_suffix = value.encode('utf8')

    @property
    def logdir(self) -> str:
        """directory in which to save YAML log files"""
        return self.p.logdir.decode()

    @logdir.setter
    def logdir(self, value: str):
        self.p.logdir = value.encode('utf8')

    @property
    def lookup_cache_static(self) -> str:
        """path of static ngram cache file for lookup decoding"""
        return self.p.lookup_cache_static.decode()

    @lookup_cache_static.setter
    def lookup_cache_static(self, value: str):
        self.p.lookup_cache_static = value.encode('utf8')

    @property
    def lookup_cache_dynamic(self) -> str:
        """path of dynamic ngram cache file for lookup decoding"""
        return self.p.lookup_cache_dynamic.decode()

    @lookup_cache_dynamic.setter
    def lookup_cache_dynamic(self, value: str):
        self.p.lookup_cache_dynamic = value.encode('utf8')

    @property
    def logits_file(self) -> str:
        """file for saving *all* logits"""
        return self.p.logits_file.decode()

    @logits_file.setter
    def logits_file(self, value: str):
        self.p.logits_file = value.encode('utf8')

    @property
    def rpc_servers(self) -> str:
        """comma separated list of RPC servers"""
        return self.p.rpc_servers.decode()

    @rpc_servers.setter
    def rpc_servers(self, value: str):
        self.p.rpc_servers = value.encode('utf8')

    @property
    def in_files(self) -> list[str]:
        """all input files."""
        result = []
        for i in range(self.p.in_files.size()):
            result.append(self.p.in_files[i].decode())
        return result

    @in_files.setter
    def in_files(self, files: list[str]):
        self.p.in_files.clear()
        for i in files:
            self.p.in_files.push_back(i.encode('utf8'))

    @property
    def antiprompt(self) -> list[str]:
        """strings upon which more user input is prompted (a.k.a. reverse prompts)."""
        result = []
        for i in range(self.p.antiprompt.size()):
            result.append(self.p.antiprompt[i].decode())
        return result

    @antiprompt.setter
    def antiprompt(self, values: list[str]):
        self.p.antiprompt.clear()
        for i in values:
            self.p.antiprompt.push_back(i.encode('utf8'))

    # std::vector<llama_model_kv_override> kv_overrides;

    @property
    def lora_init_without_apply(self) -> bool:
        """only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)."""
        return self.p.lora_init_without_apply

    @lora_init_without_apply.setter
    def lora_init_without_apply(self, value: bool):
        self.p.lora_init_without_apply = value

    # std::vector<llama_lora_adapter_info> lora_adapters; // lora adapter path with user defined scale

    # std::vector<llama_control_vector_load_info> control_vectors; // control vector with user defined scale


    @property
    def verbosity(self) -> int:
        """verbosity"""
        return self.p.verbosity

    @verbosity.setter
    def verbosity(self, value: int):
        self.p.verbosity = value

    @property
    def control_vector_layer_start(self) -> int:
        """layer range for control vector"""
        return self.p.control_vector_layer_start

    @control_vector_layer_start.setter
    def control_vector_layer_start(self, value: int):
        self.p.control_vector_layer_start = value

    @property
    def control_vector_layer_end(self) -> int:
        """layer range for control vector"""
        return self.p.control_vector_layer_end

    @control_vector_layer_end.setter
    def control_vector_layer_end(self, value: int):
        self.p.control_vector_layer_end = value

    @property
    def ppl_stride(self) -> int:
        """stride for perplexity calculations. If left at 0, the pre-existing approach will be used."""
        return self.p.ppl_stride

    @ppl_stride.setter
    def ppl_stride(self, value: int):
        self.p.ppl_stride = value

    @property
    def ppl_output_type(self) -> int:
        """0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line 

        (which is more convenient to use for plotting)
        """
        return self.p.ppl_output_type

    @ppl_output_type.setter
    def ppl_output_type(self, value: int):
        self.p.ppl_output_type = value

    @property
    def hellaswag(self) -> bool:
        """compute HellaSwag score over random tasks from datafile supplied in prompt"""
        return self.p.hellaswag

    @hellaswag.setter
    def hellaswag(self, value: bool):
        self.p.hellaswag = value

    @property
    def hellaswag_tasks(self) -> int:
        """number of tasks to use when computing the HellaSwag score"""
        return self.p.hellaswag_tasks

    @hellaswag_tasks.setter
    def hellaswag_tasks(self, value: int):
        self.p.hellaswag_tasks = value

    @property
    def winogrande(self) -> bool:
        """compute Winogrande score over random tasks from datafile supplied in prompt"""
        return self.p.winogrande

    @winogrande.setter
    def winogrande(self, value: bool):
        self.p.winogrande = value

    @property
    def winogrande_tasks(self) -> int:
        """number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed"""
        return self.p.winogrande_tasks

    @winogrande_tasks.setter
    def winogrande_tasks(self, value: int):
        self.p.winogrande_tasks = value

    @property
    def multiple_choice(self) -> bool:
        """compute TruthfulQA score over random tasks from datafile supplied in prompt"""
        return self.p.multiple_choice

    @multiple_choice.setter
    def multiple_choice(self, value: bool):
        self.p.multiple_choice = value

    @property
    def multiple_choice_tasks(self) -> int:
        """number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed"""
        return self.p.multiple_choice_tasks

    @multiple_choice_tasks.setter
    def multiple_choice_tasks(self, value: int):
        self.p.multiple_choice_tasks = value

    @property
    def kl_divergence(self) -> bool:
        """compute KL divergence"""
        return self.p.kl_divergence

    @kl_divergence.setter
    def kl_divergence(self, value: bool):
        self.p.kl_divergence = value

    @property
    def usage(self) -> bool:
        """print usage"""
        return self.p.usage

    @usage.setter
    def usage(self, value: bool):
        self.p.usage = value

    @property
    def use_color(self) -> bool:
        """use color to distinguish generations and inputs"""
        return self.p.use_color

    @use_color.setter
    def use_color(self, value: bool):
        self.p.use_color = value

    @property
    def special(self) -> bool:
        """enable special token output"""
        return self.p.special

    @special.setter
    def special(self, value: bool):
        self.p.special = value

    @property
    def interactive(self) -> bool:
        """interactive mode"""
        return self.p.interactive

    @interactive.setter
    def interactive(self, value: bool):
        self.p.interactive = value

    @property
    def interactive_first(self) -> bool:
        """wait for user input immediately"""
        return self.p.interactive_first

    @interactive_first.setter
    def interactive_first(self, value: bool):
        self.p.interactive_first = value

    @property
    def conversation(self) -> bool:
        """conversation mode (does not print special tokens and suffix/prefix)"""
        return self.p.conversation

    @conversation.setter
    def conversation(self, value: bool):
        self.p.conversation = value

    @property
    def prompt_cache_all(self) -> bool:
        """save user input and generations to prompt cache"""
        return self.p.prompt_cache_all

    @prompt_cache_all.setter
    def prompt_cache_all(self, value: bool):
        self.p.prompt_cache_all = value

    @property
    def prompt_cache_ro(self) -> bool:
        """open the prompt cache read-only and do not update it"""
        return self.p.prompt_cache_ro

    @prompt_cache_ro.setter
    def prompt_cache_ro(self, value: bool):
        self.p.prompt_cache_ro = value

    @property
    def use_color(self) -> bool:
        """use color to distinguish generations and inputs"""
        return self.p.use_color

    @use_color.setter
    def use_color(self, value: bool):
        self.p.use_color = value

    @property
    def special(self) -> bool:
        """enable special token output"""
        return self.p.special

    @special.setter
    def special(self, value: bool):
        self.p.special = value

    @property
    def interactive(self) -> bool:
        """interactive mode"""
        return self.p.interactive

    @interactive.setter
    def interactive(self, value: bool):
        self.p.interactive = value

    @property
    def interactive_first(self) -> bool:
        """wait for user input immediately"""
        return self.p.interactive_first

    @interactive_first.setter
    def interactive_first(self, value: bool):
        self.p.interactive_first = value

    @property
    def conversation(self) -> bool:
        """conversation mode (does not print special tokens and suffix/prefix)"""
        return self.p.conversation

    @conversation.setter
    def conversation(self, value: bool):
        self.p.conversation = value

    @property
    def prompt_cache_all(self) -> bool:
        """save user input and generations to prompt cache"""
        return self.p.prompt_cache_all

    @prompt_cache_all.setter
    def prompt_cache_all(self, value: bool):
        self.p.prompt_cache_all = value

    @property
    def prompt_cache_ro(self) -> bool:
        """ open the prompt cache read-only and do not update it"""
        return self.p.prompt_cache_ro

    @prompt_cache_ro.setter
    def prompt_cache_ro(self, value: bool):
        self.p.prompt_cache_ro = value

    @property
    def escape(self) -> bool:
        """escape special characters"""
        return self.p.escape

    @escape.setter
    def escape(self, value: bool):
        self.p.escape = value

    @property
    def multiline_input(self) -> bool:
        """reverse the usage of "\""""
        return self.p.multiline_input

    @multiline_input.setter
    def multiline_input(self, value: bool):
        self.p.multiline_input = value

    @property
    def simple_io(self) -> bool:
        """improves compatibility with subprocesses and limited consoles"""
        return self.p.simple_io

    @simple_io.setter
    def simple_io(self, value: bool):
        self.p.simple_io = value

    @property
    def cont_batching(self) -> bool:
        """insert new sequences for decoding on-the-fly"""
        return self.p.cont_batching

    @cont_batching.setter
    def cont_batching(self, value: bool):
        self.p.cont_batching = value

    @property
    def flash_attn(self) -> bool:
        """flash attention"""
        return self.p.flash_attn

    @flash_attn.setter
    def flash_attn(self, value: bool):
        self.p.flash_attn = value

    @property
    def no_perf(self) -> bool:
        """disable performance metrics"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value

    @property
    def ctx_shift(self) -> bool:
        """context shift on inifinite text generation"""
        return self.p.ctx_shift

    @ctx_shift.setter
    def ctx_shift(self, value: bool):
        self.p.ctx_shift = value

    @property
    def input_prefix_bos(self) -> bool:
        """prefix BOS to user inputs, preceding input_prefix"""
        return self.p.input_prefix_bos

    @input_prefix_bos.setter
    def input_prefix_bos(self, value: bool):
        self.p.input_prefix_bos = value

    @property
    def logits_all(self) -> bool:
        """return logits for all tokens in the batch"""
        return self.p.logits_all

    @logits_all.setter
    def logits_all(self, value: bool):
        self.p.logits_all = value

    @property
    def use_mmap(self) -> bool:
        """use mmap for faster loads"""
        return self.p.use_mmap

    @use_mmap.setter
    def use_mmap(self, value: bool):
        self.p.use_mmap = value

    @property
    def use_mlock(self) -> bool:
        """use mlock to keep model in memory"""
        return self.p.use_mlock

    @use_mlock.setter
    def use_mlock(self, value: bool):
        self.p.use_mlock = value

    @property
    def verbose_prompt(self) -> bool:
        """print prompt tokens before generation"""
        return self.p.verbose_prompt

    @verbose_prompt.setter
    def verbose_prompt(self, value: bool):
        self.p.verbose_prompt = value

    @property
    def display_prompt(self) -> bool:
        """print prompt before generation"""
        return self.p.display_prompt

    @display_prompt.setter
    def display_prompt(self, value: bool):
        self.p.display_prompt = value

    @property
    def dump_kv_cache(self) -> bool:
        """dump the KV cache contents for debugging purposes"""
        return self.p.dump_kv_cache

    @dump_kv_cache.setter
    def dump_kv_cache(self, value: bool):
        self.p.dump_kv_cache = value

    @property
    def no_kv_offload(self) -> bool:
        """disable KV offloading"""
        return self.p.no_kv_offload

    @no_kv_offload.setter
    def no_kv_offload(self, value: bool):
        self.p.no_kv_offload = value

    @property
    def warmup(self) -> bool:
        """warmup run"""
        return self.p.warmup

    @warmup.setter
    def warmup(self, value: bool):
        self.p.warmup = value

    @property
    def check_tensors(self) -> bool:
        """validate tensor data"""
        return self.p.check_tensors

    @check_tensors.setter
    def check_tensors(self, value: bool):
        self.p.check_tensors = value

    @property
    def mmproj(self) -> str:
        """path to multimodal projector"""
        return self.p.mmproj.decode()

    @mmproj.setter
    def mmproj(self, value: str):
        self.p.mmproj = value.encode('utf8')

    @property
    def image(self) -> list[str]:
        """paths to image file(s)"""
        result = []
        for i in range(self.p.image.size()):
            result.append(self.p.image[i].decode())
        return result

    @image.setter
    def image(self, files: list[str]):
        self.p.image.clear()
        for i in files:
            self.p.image.push_back(i.encode('utf8'))

    @property
    def embedding(self) -> bool:
        """get only sentence embedding"""
        return self.p.embedding

    @embedding.setter
    def embedding(self, value: bool):
        self.p.embedding = value

    @property
    def embd_normalize(self) -> int:
        """normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)"""
        return self.p.embd_normalize

    @embd_normalize.setter
    def embd_normalize(self, value: int):
        self.p.embd_normalize = value

    @property
    def embd_out(self) -> str:
        """empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix"""
        return self.p.embd_out.decode()

    @embd_out.setter
    def embd_out(self, value: str):
        self.p.embd_out = value.encode('utf8')

    @property
    def embd_sep(self) -> str:
        """separator of embendings"""
        return self.p.embd_sep.decode()

    @embd_sep.setter
    def embd_sep(self, value: str):
        self.p.embd_sep = value.encode('utf8')

    @property
    def reranking(self) -> bool:
        """enable reranking support on server"""
        return self.p.reranking

    @reranking.setter
    def reranking(self, value: bool):
        self.p.reranking = value

    @property
    def hostname(self) -> str:
        """server hostname"""
        return self.p.hostname.decode()

    @hostname.setter
    def hostname(self, value: str):
        self.p.hostname = value.encode('utf8')

    @property
    def public_path(self) -> str:
        """server public_path"""
        return self.p.public_path.decode()

    @public_path.setter
    def public_path(self, value: str):
        self.p.public_path = value.encode('utf8')

    @property
    def chat_template(self) -> str:
        """chat template"""
        return self.p.chat_template.decode()

    @chat_template.setter
    def chat_template(self, value: str):
        self.p.chat_template = value.encode('utf8')

    # @property
    # def system_prompt(self) -> str:
    #     """system prompt"""
    #     return self.p.system_prompt.decode()

    # @system_prompt.setter
    # def system_prompt(self, value: str):
    #     self.p.system_prompt = value.encode('utf8')

    @property
    def enable_chat_template(self) -> bool:
        """enable chat template"""
        return self.p.enable_chat_template

    @enable_chat_template.setter
    def enable_chat_template(self, value: bool):
        self.p.enable_chat_template = value

    @property
    def api_keys(self) -> list[str]:
        """list of api keys"""
        result = []
        for i in range(self.p.api_keys.size()):
            result.append(self.p.api_keys[i].decode())
        return result

    @api_keys.setter
    def api_keys(self, files: list[str]):
        self.p.api_keys.clear()
        for i in files:
            self.p.api_keys.push_back(i.encode('utf8'))

    @property
    def ssl_file_key(self) -> str:
        """ssl file key"""
        return self.p.ssl_file_key.decode()

    @ssl_file_key.setter
    def ssl_file_key(self, value: str):
        self.p.ssl_file_key = value.encode('utf8')

    @property
    def ssl_file_cert(self) -> str:
        """ssl file cert"""
        return self.p.ssl_file_cert.decode()

    @ssl_file_cert.setter
    def ssl_file_cert(self, value: str):
        self.p.ssl_file_cert = value.encode('utf8')

    @property
    def webui(self) -> bool:
        """enable webui"""
        return self.p.webui

    @webui.setter
    def webui(self, value: bool):
        self.p.webui = value

    @property
    def endpoint_slots(self) -> bool:
        """endpoint slots"""
        return self.p.endpoint_slots

    @endpoint_slots.setter
    def endpoint_slots(self, value: bool):
        self.p.endpoint_slots = value

    @property
    def endpoint_props(self) -> bool:
        """endpoint props"""
        return self.p.endpoint_props

    @endpoint_props.setter
    def endpoint_props(self, value: bool):
        self.p.endpoint_props = value

    @property
    def endpoint_metrics(self) -> bool:
        """endpoint metrics"""
        return self.p.endpoint_metrics

    @endpoint_metrics.setter
    def endpoint_metrics(self, value: bool):
        self.p.endpoint_metrics = value

    @property
    def log_json(self) -> bool:
        """log json"""
        return self.p.log_json

    @log_json.setter
    def log_json(self, value: bool):
        self.p.log_json = value

    @property
    def slot_save_path(self) -> str:
        """slot save path"""
        return self.p.slot_save_path.decode()

    @slot_save_path.setter
    def slot_save_path(self, value: str):
        self.p.slot_save_path = value.encode('utf8')

    @property
    def slot_prompt_similarity(self) -> float:
        """slot prompt similarity."""
        return self.p.slot_prompt_similarity

    @slot_prompt_similarity.setter
    def slot_prompt_similarity(self, value: float):
        self.p.slot_prompt_similarity = value

    # @property
    # def is_pp_shared(self) -> bool:
    #     """batched-bench params"""
    #     return self.p.is_pp_shared

    # @is_pp_shared.setter
    # def is_pp_shared(self, value: bool):
    #     self.p.is_pp_shared = value


    # std::vector<int32_t> n_pp;
    # std::vector<int32_t> n_tg;
    # std::vector<int32_t> n_pl;

    # // retrieval params
    # std::vector<std::string> context_files; // context files to embed

    # int32_t chunk_size = 64; // chunk size for context embedding

    # std::string chunk_separator = "\n"; // chunk separator for context embedding

    # // passkey params
    # int32_t n_junk = 250; // number of times to repeat the junk text
    # int32_t i_pos  = -1;  // position of the passkey in the junk text

    # // imatrix params
    # std::string out_file = "imatrix.dat"; // save the resulting imatrix to this file

    # int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    # int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    # int32_t i_chunk     =  0; // start processing from this chunk

    # bool process_output = false; // collect data for the output tensor
    # bool compute_ppl    = true;  // whether to compute perplexity

    # // cvector-generator params
    # int n_pca_batch = 100;
    # int n_pca_iterations = 1000;
    # dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    # std::string cvector_outfile       = "control_vector.gguf";
    # std::string cvector_positive_file = "examples/cvector-generator/positive.txt";
    # std::string cvector_negative_file = "examples/cvector-generator/negative.txt";

    # bool spm_infill = false; // suffix/prefix/middle pattern for infill

    # std::string lora_outfile = "ggml-lora-merged-f16.gguf";

    # // batched-bench params
    # bool batched_bench_output_jsonl = false;


cdef class ModelParams:
    cdef llama_cpp.llama_model_params p

    def __init__(self):
        self.p = llama_cpp.llama_model_default_params()

    @staticmethod
    cdef ModelParams from_instance(llama_cpp.llama_model_params params):
        cdef ModelParams wrapper = ModelParams.__new__(ModelParams)
        wrapper.p = params
        return wrapper

    @property
    def n_gpu_layers(self) -> int:
        """Number of layers to store in VRAM."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def split_mode(self) -> int:
        """How to split the model across multiple GPUs."""
        return self.p.split_mode

    @split_mode.setter
    def split_mode(self, value: int):
        self.p.split_mode = value

    @property
    def main_gpu(self) -> int:
        """main_gpu interpretation depends on split_mode:

        LLAMA_SPLIT_NONE: the GPU that is used for the entire model
        LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
        LLAMA_SPLIT_LAYER: ignored
        """
        return self.p.main_gpu

    @main_gpu.setter
    def main_gpu(self, value: int):
        self.p.main_gpu = value

    @property
    def vocab_only(self) -> bool:
        """Load only the vocabulary, no weights"""
        return self.p.vocab_only

    @vocab_only.setter
    def vocab_only(self, value: bool):
        self.p.vocab_only = value

    @property
    def use_mmap(self) -> bool:
        """Use mmap if possible"""
        return self.p.use_mmap

    @use_mmap.setter
    def use_mmap(self, value: bool):
        self.p.use_mmap = value

    @property
    def use_mlock(self) -> bool:
        """Force system to keep model in RAM"""
        return self.p.use_mlock

    @use_mlock.setter
    def use_mlock(self, value: bool):
        self.p.use_mlock = value

    @property
    def check_tensors(self) -> bool:
        """Validate model tensor data"""
        return self.p.check_tensors

    @check_tensors.setter
    def check_tensors(self, value: bool):
        self.p.check_tensors = value


cdef class ModelQuantizeParams:
    cdef llama_cpp.llama_model_quantize_params p

    def __init__(self):
        self.p = llama_cpp.llama_model_quantize_default_params()

    @staticmethod
    cdef ModelQuantizeParams from_instance(llama_cpp.llama_model_quantize_params params):
        cdef ModelQuantizeParams wrapper = ModelQuantizeParams.__new__(ModelQuantizeParams)
        wrapper.p = params
        return wrapper

    @property
    def nthread(self) -> int:
        """number of threads to use for quantizing.

        if <=0 will use std::thread::hardware_concurrency().
        """
        return self.p.nthread

    @nthread.setter
    def nthread(self, value: int):
        self.p.nthread = value

    @property
    def ftype(self) -> int:
        """quantize to this llama_ftype"""
        return self.p.ftype

    @ftype.setter
    def ftype(self, value: int):
        self.p.ftype = value

    @property
    def output_tensor_type(self) -> int:
        """output tensor type"""
        return self.p.output_tensor_type

    @output_tensor_type.setter
    def output_tensor_type(self, value: int):
        self.p.output_tensor_type = value

    @property
    def token_embedding_type(self) -> int:
        """itoken embeddings tensor type"""
        return self.p.token_embedding_type

    @token_embedding_type.setter
    def token_embedding_type(self, value: int):
        self.p.token_embedding_type = value

    @property
    def allow_requantize(self) -> bool:
        """allow quantizing non-f32/f16 tensors"""
        return self.p.allow_requantize

    @allow_requantize.setter
    def allow_requantize(self, value: bool):
        self.p.allow_requantize = value

    @property
    def quantize_output_tensor(self) -> bool:
        """quantize output.weight"""
        return self.p.quantize_output_tensor

    @quantize_output_tensor.setter
    def quantize_output_tensor(self, value: bool):
        self.p.quantize_output_tensor = value

    @property
    def only_copy(self) -> bool:
        """only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored"""
        return self.p.only_copy

    @only_copy.setter
    def only_copy(self, value: bool):
        self.p.only_copy = value

    @property
    def pure(self) -> bool:
        """quantize all tensors to the default type"""
        return self.p.pure

    @pure.setter
    def pure(self, value: bool):
        self.p.pure = value

    @property
    def keep_split(self) -> bool:
        """quantize to the same number of shards"""
        return self.p.keep_split

    @keep_split.setter
    def keep_split(self, value: bool):
        self.p.keep_split = value


cdef class LlamaModel:
    """cython wrapper for llama_cpp.cpp llama_model."""
    cdef llama_cpp.llama_model * ptr
    cdef public ModelParams params
    cdef public str path_model
    cdef public bint verbose
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, *, path_model: str, params: Optional[ModelParams] = None, verbose: bool = True):
        self.path_model = path_model
        self.params = params if params else ModelParams()
        self.verbose = verbose

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        # with suppress_stdout_stderr(disable=verbose):
        self.ptr = llama_cpp.llama_load_model_from_file(
            self.path_model.encode("utf-8"), 
            self.params.p
        )

        if self.ptr is NULL:
            raise ValueError(f"Failed to load model from file: {path_model}")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_free_model(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaModel from_ptr(llama_cpp.llama_model *ptr, bint owner=False):
        cdef LlamaModel wrapper = LlamaModel.__new__(LlamaModel)
        wrapper.ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    def vocab_type(self) -> llama_vocab_type:
        return llama_cpp.get_llama_vocab_type(self.ptr)

    def rope_type(self) -> llama_rope_type:
        return llama_cpp.get_llama_rope_type(self.ptr)

    def n_vocab(self) -> int:
        return llama_cpp.llama_n_vocab(self.ptr)

    def n_ctx_train(self) -> int:
        return llama_cpp.llama_n_ctx_train(self.ptr)

    def n_embd(self) -> int:
        return llama_cpp.llama_n_embd(self.ptr)

    def n_layer(self) -> int:
        return llama_cpp.llama_n_layer(self.ptr)

    def n_head(self) -> int:
        return llama_cpp.llama_n_head(self.ptr)

    def rope_freq_scale_train(self) -> float:
        """Get the model's RoPE frequency scaling factor"""
        return llama_cpp.llama_rope_freq_scale_train(self.ptr)

    def desc(self) -> str:
        """Get a string describing the model type"""
        cdef char buf[1024]
        llama_cpp.llama_model_desc(self.ptr, buf, 1024)
        return buf.decode("utf-8")

    def size(self) -> int:
        """Returns the total size of all the tensors in the model in bytes"""
        return llama_cpp.llama_model_size(self.ptr)

    def n_params(self) -> int:
        """Returns the total number of parameters in the model"""
        return llama_cpp.llama_model_n_params(self.ptr)

    def get_tensor(self, name: str) -> GGMLTensor:
        """Get a llama model tensor"""
        cdef llama_cpp.ggml_tensor * tensor = llama_cpp.llama_get_model_tensor(
            self.ptr, name.encode("utf-8"))
        return GGMLTensor.from_ptr(tensor)

    # lora

    def lora_adapter_init(self, str path_lora) -> LoraAdapter:
        """Load a LoRA adapter from file

        The loaded adapter will be associated to the given model, and will be free when the model is deleted
        """
        cdef llama_cpp.llama_lora_adapter * ptr = llama_cpp.llama_lora_adapter_init(
            self.ptr, path_lora.encode())
        cdef LoraAdapter adapter = LoraAdapter.from_ptr(ptr)
        return adapter

    # metadata

    def meta_val_str(self, str key) -> str:
        """Get metadata value as a string by key name"""
        cdef char buf[128]
        cdef int res = llama_cpp.llama_model_meta_val_str(self.ptr, key.encode(), buf, 128)
        if res == -1:
            raise ValueError(F"could not get metadata value from {key}")
        cdef str value = buf.decode('UTF-8')
        return value

    def meta_count(self):
        """Get the number of metadata key/value pairs"""
        return llama_cpp.llama_model_meta_count(self.ptr)

    def meta_key_by_index(self, int index) -> str:
        """Get metadata key name by index"""
        cdef char buf[128]
        cdef int res = llama_cpp.llama_model_meta_key_by_index(self.ptr, index, buf, 128)
        cdef str key = buf.decode('UTF-8')
        return key

    def meta_val_str_by_index(self, int index) -> str:
        """Get metadata key name by index"""
        cdef char buf[128]
        cdef int res = llama_cpp.llama_model_meta_val_str_by_index(self.ptr, index, buf, 128)
        cdef str value = buf.decode('UTF-8')
        return value

    # encode / decode

    def has_encoder(self) -> bool:
        """Returns true if the model contains an encoder that requires llama_encode() call"""
        return llama_cpp.llama_model_has_encoder(self.ptr)

    def has_decoder(self) -> bool:
        """Returns true if the model contains a decoder that requires llama_decode() callD"""
        return llama_cpp.llama_model_has_decoder(self.ptr)

    def decoder_start_token(self) -> int:
        """For encoder-decoder models, this function returns id of the token that must be provided
        to the decoder to start generating output sequence. For other models, it returns -1.
        """
        return llama_cpp.llama_model_decoder_start_token(self.ptr)

    def is_recurrent(self) -> bool:
        """Returns true if the model is recurrent (like Mamba, RWKV, etc.)"""
        return llama_cpp.llama_model_is_recurrent(self.ptr)


    # Vocab

    def token_get_text(self, llama_cpp.llama_token token) -> str:
        return llama_cpp.llama_token_get_text(self.ptr, token).decode("utf-8")

    def token_get_score(self, llama_cpp.llama_token token) -> float:
        return llama_cpp.llama_token_get_score(self.ptr, token)

    def token_get_attr(self, llama_cpp.llama_token token) -> llama_token_attr:
        return llama_cpp.llama_token_get_attr(self.ptr, token)

    def token_is_eog(self, llama_cpp.llama_token token) -> bool:
        """Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"""
        return llama_cpp.llama_token_is_eog(self.ptr, token)

    def token_is_control(self, llama_cpp.llama_token token) -> bool:
        """Identify if Token Id is a control token or a render-able token"""
        return llama_cpp.llama_token_is_control(self.ptr, token)

    # Special tokens

    def token_bos(self) -> int:
        return llama_cpp.llama_token_bos(self.ptr)

    def token_eos(self) -> int:
        return llama_cpp.llama_token_eos(self.ptr)

    def token_cls(self) -> int:
        return llama_cpp.llama_token_cls(self.ptr)

    def token_sep(self) -> int:
        return llama_cpp.llama_token_sep(self.ptr)

    def token_nl(self) -> int:
        return llama_cpp.llama_token_nl(self.ptr)

    def token_fim_prefix(self) -> int:
        return llama_cpp.llama_token_fim_pre(self.ptr)

    def token_fim_middle(self) -> int:
        return llama_cpp.llama_token_fim_suf(self.ptr)

    def token_fim_suffix(self) -> int:
        return llama_cpp.llama_token_fim_mid(self.ptr)

    def token_fim_pad(self) -> int:
        return llama_cpp.llama_token_fim_pad(self.ptr)

    def token_fim_rep(self) -> int:
        return llama_cpp.llama_token_fim_rep(self.ptr)

    def token_fim_sep(self) -> int:
        return llama_cpp.llama_token_fim_sep(self.ptr)

    # def token_eot(self) -> int:
    #     return llama_cpp.llama_token_eot(self.ptr)

    def add_bos_token(self) -> bool:
        return llama_cpp.llama_add_bos_token(self.ptr)

    def add_eos_token(self) -> bool:
        return llama_cpp.llama_add_eos_token(self.ptr)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        cdef int n_ctx = self.n_ctx_train()
        cdef vector[llama_cpp.llama_token] tokens
        tokens.reserve(n_ctx)
        n_tokens = llama_cpp.llama_tokenize(
            self.ptr, text, len(text), tokens.data(), n_ctx, add_bos, special
        )
        if n_tokens < 0:
            n_tokens = abs(n_tokens)
            # tokens = (llama_cpp.llama_token * n_tokens)()
            n_tokens = llama_cpp.llama_tokenize(
                self.ptr, text, len(text), tokens.data(), n_tokens, add_bos, special
            )
            if n_tokens < 0:
                raise RuntimeError()
                # raise RuntimeError(
                #     f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                # )

        return tokens[:n_tokens]

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        cdef char buf[32]
        llama_cpp.llama_token_to_piece(self.ptr, token, buf, 32, 0, special)
        return buf.decode()
        # return bytes(buf)

    def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
        assert self.ptr is not NULL
        output = b""
        size = 32
        cdef char buffer[32]
        for token in tokens:
            n = llama_cpp.llama_token_to_piece(
                self.ptr, int(token), buffer, size, 0, special
            )
            assert n <= size
            output += bytes(buffer[:n])
        # NOTE: Llama1 models automatically added a space at the start of the prompt
        # this line removes a leading space if the first token is a beginning of sentence token
        return (
            output[1:]
            if len(tokens) > 0 and tokens[0] == self.token_bos() and output[0:1] == b" "
            else output
        )

    # chat template


    # def chat_apply_template(self, str tmpl, str chat, size_t n_msg, bint add_ass) -> str:
    #     """Apply chat template. Inspired by hf apply_chat_template() on python.
        
    #     Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    #     NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    #     @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    #     @param chat Pointer to a list of multiple llama_chat_message
    #     @param n_msg Number of llama_chat_message in this chat
    #     @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    #     @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    #     @param length The size of the allocated buffer
    #     @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    #     """
    #     cdef char * buf = NULL
    #     cdef int length = 0
    #     cdef int bytes = llama_cpp.llama_chat_apply_template(
    #         self.ptr, tmpl.encode(), chat.encode(), n_msg, add_ass, buf, length)
    #     cdef str result = buf.decode()
    #     free(buf)
    #     return result



    # Extra

    def metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        buffer_size = 1024
        cdef int nbytes
        cdef char * buffer = <char*>calloc(buffer_size, sizeof(char))
        assert self.ptr is not NULL
        # iterate over model keys
        for i in range(llama_cpp.llama_model_meta_count(self.ptr)):
            nbytes = llama_cpp.llama_model_meta_key_by_index(
                self.ptr, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama_cpp.llama_model_meta_key_by_index(
                    self.ptr, i, buffer, buffer_size
                )
            key = buffer.decode("utf-8")
            nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                self.ptr, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                    self.ptr, i, buffer, buffer_size
                )
            value = buffer.decode("utf-8")
            metadata[key] = value
        free(buffer)
        return metadata

    @staticmethod
    def default_params() -> ModelParams:
        """Get the default llama_model_params."""
        # return llama_cpp.llama_model_default_params()
        return ModelParams()


cdef class ContextParams:
    cdef llama_cpp.llama_context_params p

    def __init__(self):
        self.p = llama_cpp.llama_context_default_params()

    @staticmethod
    cdef ContextParams from_common_params(CommonParams params):
        cdef ContextParams wrapper = ContextParams.__new__(ContextParams)
        wrapper.p = llama_cpp.common_context_params_to_llama(params.p)
        return wrapper

    # @property
    # def seed(self) -> int:
    #     """RNG seed, -1 for random."""
    #     return self.p.seed

    # @seed.setter
    # def seed(self, value: int):
    #     self.p.seed = value

    @property
    def n_ctx(self) -> int:
        """text context, 0 = from model."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_batch(self) -> int:
        """logical maximum batch size that can be submitted to llama_decode."""
        return self.p.n_batch

    @n_batch.setter
    def n_batch(self, value: int):
        self.p.n_batch = value

    @property
    def n_ubatch(self) -> int:
        """physical maximum batch size."""
        return self.p.n_ubatch

    @n_ubatch.setter
    def n_ubatch(self, value: int):
        self.p.n_ubatch = value

    @property
    def n_seq_max(self) -> int:
        """max number of sequences (i.e. distinct states for recurrent models)."""
        return self.p.n_seq_max

    @n_seq_max.setter
    def n_seq_max(self, value: int):
        self.p.n_seq_max = value

    @property
    def n_threads(self) -> int:
        """number of threads to use for generation."""
        return self.p.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.p.n_threads = value

    @property
    def n_threads_batch(self) -> int:
        """number of threads to use for batch processing"""
        return self.p.n_threads_batch

    @n_threads_batch.setter
    def n_threads_batch(self, value: int):
        self.p.n_threads_batch = value

    @property
    def rope_scaling_type(self) -> llama_cpp.llama_rope_scaling_type:
        """number of threads to use for batch processing"""
        return <llama_cpp.llama_rope_scaling_type>self.p.rope_scaling_type

    @rope_scaling_type.setter
    def rope_scaling_type(self, llama_cpp.llama_rope_scaling_type value):
        self.p.rope_scaling_type = value


cdef class LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context."""
    cdef llama_cpp.llama_context * ptr
    cdef public LlamaModel model
    cdef public ContextParams params
    cdef public bint verbose
    cdef public int n_tokens
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True
        self.n_tokens = 0

    def __init__(
        self,
        *,
        model: LlamaModel,
        params: Optional[ContextParams] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.params = params if params else ContextParams()
        self.verbose = verbose

        # self.ptr = None

        assert self.model.ptr is not NULL

        self.ptr = llama_cpp.llama_new_context_with_model(self.model.ptr, self.params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to create llama_context")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaContext from_ptr(llama_cpp.llama_context *ptr, bint owner=False):
        cdef LlamaContext wrapper = LlamaContext.__new__(LlamaContext)
        wrapper.ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    def close(self):
        self.__dealloc__()

    def n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx(self.ptr)

    def n_batch(self) -> int:
        return llama_cpp.llama_n_batch(self.ptr)

    def n_ubatch(self) -> int:
        return llama_cpp.llama_n_ubatch(self.ptr)

    def n_seq_max(self) -> int:
        return llama_cpp.llama_n_seq_max(self.ptr)

    def pooling_type(self) -> int:
        return llama_cpp.get_llama_pooling_type(self.ptr)

    # Lora
    # -------------------------------------------------------------------------

    def lora_adapter_set(self, LoraAdapter adapter, float scale):
        """Add a loaded LoRA adapter to given context
        
        This will not modify model's weight
        """
        cdef int32_t res = llama_cpp.llama_lora_adapter_set(
            self.ptr, adapter.ptr, scale)
        if res == -1:
            raise ValueError(f"cannot load lora adapter to context")

    def lora_adapter_remove(self, LoraAdapter adapter):
        """Remove a specific LoRA adapter from given context

        Return -1 if the adapter is not present in the context
        """
        cdef int32_t res = llama_cpp.llama_lora_adapter_remove(
            self.ptr, adapter.ptr)
        if res == -1:
            raise ValueError(f"cannot remove, lora the adapter is not present in the context")

    def lora_adapter_clear(self):
        """Remove all LoRA adapters from given context"""
        llama_cpp.llama_lora_adapter_clear(self.ptr)


    def control_vector_apply(self, data: list[float], n_embd: int, il_start: int, il_end: int) -> int:
        """Apply a loaded control vector to a llama_context, or if data is NULL, clear
        the currently loaded vector.
        
        `n_embd` should be the size of a single layer's control, and data should point
        to an n_embd x n_layers buffer starting from layer 1.

        `il_start` and `il_end` are the layer range the vector should apply to (both inclusive)
        
        See llama_control_vector_load in common to load a control vector.
        """
        cdef vector[float] vec
        for i in data:
            vec.push_back(i)

        return llama_cpp.llama_control_vector_apply(
            self.ptr,
            vec.data(),
            vec.size(),
            n_embd,
            il_start,
            il_end
        )

    # KV cache
    # -------------------------------------------------------------------------

    def kv_cache_clear(self):
        """Clear the KV cache - both cell info is erased and KV data is zeroed"""
        llama_cpp.llama_kv_cache_clear(self.ptr)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        """Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
        
        Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
        seq_id < 0 : match any sequence
        p0 < 0     : [0,  p1]
        p1 < 0     : [p0, inf)
        """
        llama_cpp.llama_kv_cache_seq_rm(self.ptr, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        """Copy all tokens that belong to the specified sequence to another sequence
        
        Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
        p0 < 0 : [0,  p1]
        p1 < 0 : [p0, inf)
        """
        llama_cpp.llama_kv_cache_seq_cp(self.ptr, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        """Removes all tokens that do not belong to the specified sequence"""
        llama_cpp.llama_kv_cache_seq_keep(self.ptr, seq_id)

    def kv_cache_seq_shift(self, seq_id: int, p0: int, p1: int, shift: int):
        """Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
        
        If the KV cache is RoPEd, the KV data is updated accordingly:
          - lazily on next llama_decode()
          - explicitly with llama_kv_cache_update()
        p0 < 0 : [0,  p1]
        p1 < 0 : [p0, inf)
        """
        llama_cpp.llama_kv_cache_seq_add(self.ptr, seq_id, p0, p1, shift)

    def kv_cache_seq_div(self, seq_id: int, p0: int, p1: int, d: int):
        """Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
        
        If the KV cache is RoPEd, the KV data is updated accordingly:
          - lazily on next llama_decode()
          - explicitly with llama_kv_cache_update()
        p0 < 0 : [0,  p1]
        p1 < 0 : [p0, inf)
        """
        llama_cpp.llama_kv_cache_seq_div(self.ptr, seq_id, p0, p1, d)

    def kv_cache_seq_pos_max(self, seq_id: int) -> int:
        """Returns the largest position present in the KV cache for the specified sequence"""
        return llama_cpp.llama_kv_cache_seq_pos_max(self.ptr, seq_id)


    def kv_cache_defrag(self):
        """Defragment the KV cache
        
        This will be applied:
        - lazily on next llama_decode()
        - explicitly with llama_kv_cache_update()
        """
        llama_cpp.llama_kv_cache_defrag(self.ptr)

    
    def kv_cache_update(self):
        """Apply the KV cache updates (such as K-shifts, defragmentation, etc.)"""
        llama_cpp.llama_kv_cache_update(self.ptr)

    # State / sessions
    # -------------------------------------------------------------------------

    def get_state_size(self) -> int:
        """Returns the *actual* size in bytes of the state

        (logits, embedding and kv_cache)
        Only use when saving the state, not when restoring it, otherwise the size may be too small.
        """
        return llama_cpp.llama_state_get_size(self.ptr)

    def get_state_data(self) -> list[int]:
        """Copies the state to the specified destination address.
        
        Destination needs to have allocated enough memory.
        Returns the number of bytes copied
        """
        cdef uint8_t * dst = NULL
        cdef size_t size = 0
        cdef vector[uint8_t] result
        cdef size_t copied = llama_cpp.llama_state_get_data(self.ptr, dst, size)
        for i in range(size):
            result.push_back(dst[i])
        return result

    def set_state_data(self, data: list[int]) -> int:
        """Set the state reading from the specified address

        Returns the number of bytes read
        """
        cdef vector[uint8_t] result = data
        cdef size_t read = llama_cpp.llama_state_set_data(self.ptr, result.data(), result.size())
        return read

    def load_state_file(self, path_session: str, max_n_tokens: int = 256) -> list[int]:
        """Load session file"""
        cdef llama_cpp.llama_token * tokens_out = NULL
        cdef size_t * n_token_count_out = NULL
        cdef bint loaded = llama_cpp.llama_state_load_file(
            self.ptr, 
            path_session.encode(), 
            tokens_out,
            max_n_tokens,
            n_token_count_out)
        cdef vector[int] result
        if loaded:
            for i in range(n_token_count_out[0]):
                result.push_back(tokens_out[i])
        return result

    def save_state_file(self, path_session: str, tokens: list[int]) -> bool:
        """Save session file"""
        cdef vector[llama_cpp.llama_token] vec_tokens
        for token in tokens:
            vec_tokens.push_back(<llama_cpp.llama_token>token)
        return llama_cpp.llama_state_save_file(
            self.ptr,
            path_session.encode(),
            vec_tokens.data(),
            vec_tokens.size())

    def get_state_seq_size(self, int seq_id) -> int:
        """Get the exact size needed to copy the KV cache of a single sequence"""
        return llama_cpp.llama_state_seq_get_size(self.ptr, seq_id)

    def get_state_seq_data(self, int seq_id) -> list[int]:
        """Copy the KV cache of a single sequence into the specified buffer"""
        cdef uint8_t * dst = NULL
        cdef size_t size = 0
        cdef size_t copied = llama_cpp.llama_state_seq_get_data(
            self.ptr, dst, size, seq_id)
        cdef vector[uint8_t] result
        if copied > 0:
            for i in range(size):
                result.push_back(dst[i])
        return result

    def set_state_seq_data(self, src: list[int], dest_seq_id: int):
        """Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence

        Returns:
         - Positive: Ok
         - Zero: Failed to load
        """
        cdef vector[int] vec
        cdef size_t res = 0 
        for i in src:
            vec.push_back(i)
        res = llama_cpp.llama_state_seq_set_data(
            self.ptr, src.data(), src.size(), dest_seq_id)
        if res == 0:
            raise ValueError("Failed to load sequence data")

    def save_state_seq_file(self, filepath: str, seq_id: int, tokens: list[int]):
        """Save state sequence data to a file"""
        cdef vector[int] vec
        cdef size_t res = 0
        for i in tokens:
            vec.push_back(i)
        res = llama_cpp.llama_state_seq_save_file(
            self.ptr,
            filepath.encode(),
            seq_id,
            <const llama_cpp.llama_token *>vec.data(),
            vec.size())
        if res == 0:
            raise ValueError(f"Failed to save seq data {filepath}")

    def load_state_seq_file(self, filepath: str, dest_seq_id: int, max_n_tokens: int = 256):
        """Load state sequence data from a file"""
        cdef llama_cpp.llama_token * tokens_out = NULL
        cdef size_t * n_token_count_out = NULL
        cdef size_t loaded = llama_cpp.llama_state_seq_load_file(
            self.ptr,
            filepath.encode(), 
            dest_seq_id,
            tokens_out,
            max_n_tokens,
            n_token_count_out)
        cdef vector[int] result
        if loaded:
            for i in range(n_token_count_out[0]):
                result.push_back(tokens_out[i])
        return result

    # Decoding
    # -------------------------------------------------------------------------

    def encode(self, LlamaBatch batch):
        """Processes a batch of tokens with the encoder part of the encoder-decoder model.

        Stores the encoder output internally for later use by the decoder cross-attention layers.
          0 - success
        < 0 - error
        """
        cdef int32_t res = llama_cpp.llama_encode(self.ptr, batch.p)
        if res < 0:
            raise RuntimeError("error encoding batch")

    def decode(self, LlamaBatch batch):
        """Positive return values does not mean a fatal error, but rather a warning.
          
          0 - success
          1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
        < 0 - error
        """
        cdef int32_t res = llama_cpp.llama_decode(
            self.ptr,
            batch.p,
        )
        self.n_tokens = batch.n_tokens
        if res == 1:
            raise ValueError("could not find a KV slot for the batch (try reducing the size of the batch or increase the context)")
        if res < 0:
            raise RuntimeError(f"llama_decode failed")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        """Set the number of threads used for decoding
        
        n_threads is the number of threads used for generation (single token)
        n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
        """
        llama_cpp.llama_set_n_threads(self.ptr, n_threads, n_threads_batch)

    def n_threads(self):
        """Get the number of threads used for generation of a single token."""
        return llama_cpp.llama_n_threads(self.ptr)

    def n_threads_batch(self):
        """Get the number of threads used for prompt and batch processing (multiple token)."""
        return llama_cpp.llama_n_threads_batch(self.ptr)

    def set_embeddings_mode(self, embeddings: bool):
        """Set whether the model is in embeddings mode or not
    
        If true, embeddings will be returned but logits will not
        """
        llama_cpp.llama_set_embeddings(self.ptr, embeddings)

    def set_causal_attn(self, causal_attn: bool):
        """Set whether to use causal attention or not

        If set to true, the model will only attend to the past tokens
        """
        llama_cpp.llama_set_causal_attn(self.ptr, causal_attn)

    # def set_abort_callback(self, abort_callback):
    #     """Set abort callback"""
    #     llama_cpp.llama_set_abort_callback(self.ptr, ggml_abort_callback abort_callback, void * abort_callback_data)

    def synchronize(self):
        """Wait until all computations are finished

        This is automatically done when using one of the functions below to obtain the computation results
        and is not necessary to call it explicitly in most cases
        """
        llama_cpp.llama_synchronize(self.ptr)

    # def n_outputs(self) -> int:
    #     return llama_cpp.llama_n_outputs(self.ptr)

    def get_logits(self) -> list[float]:
        """Token logits obtained from the last call to llama_decode()

        The logits for which llama_batch.logits[i] != 0 are stored contiguously
        in the order they have appeared in the batch.

        Rows: number of tokens for which llama_batch.logits[i] != 0
        Cols: n_vocab
        """
        cdef int n_vocab = self.model.n_vocab()
        cdef float * logits = llama_cpp.llama_get_logits(self.ptr)
        cdef vector[float] vec
        for i in range(n_vocab):
            vec.push_back(logits[i])
        return vec

    # def get_logits_ith(self, int i):
    #     """Logits for the ith token. For positive indices, 

    #     Equivalent to:
    #     llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    #     Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    #     returns NULL for invalid ids.
    #     """
    #     cdef float * logits = llama_get_logits_ith( llama_context * ctx, int32_t i)

    # def get_embeddings(self):
    #     """Get all output token embeddings.

    #     when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    #     the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    #     in the order they have appeared in the batch.
    #     shape: [n_outputs*n_embd]
    #     Otherwise, returns NULL.
    #     """
    #     cdef float * embds = llama_cpp.llama_get_embeddings(self.ptr)


    # def get_embeddings_ith(self, int i):
    #     """Get the embeddings for the ith token. For positive indices, Equivalent to:
        
    #     llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    #     Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    #     returns NULL for invalid ids.
    #     """
    #     cdef float * embds = llama_cpp.llama_get_embeddings_ith(self.ptr, i)


    # def get_embeddings_seq(self, int seq_id):
    #     """Get the embeddings for a sequence id

    #     Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    #     when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
    #     otherwise: float[n_embd] (1-dimensional)
    #     """
    #     cdef float * embds = llama_get_embeddings_seq(self.ptr, seq_id)


    # Sampling functions
    # -------------------------------------------------------------------------

    # def set_rng_seed(self, seed: int):
    #     # TODO: Fix
    #     llama_cpp.llama_set_rng_seed(self.ptr, seed)

    # def sample_repetition_penalties(
    #     self,
    #     candidates: "_LlamaTokenDataArray",
    #     last_tokens_data: "llama_cpp.Array[llama_cpp.llama_token]",
    #     penalty_last_n: int,
    #     penalty_repeat: float,
    #     penalty_freq: float,
    #     penalty_present: float,
    # ):
    #     llama_cpp.llama_sample_repetition_penalties(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #         last_tokens_data,
    #         penalty_last_n,
    #         penalty_repeat,
    #         penalty_freq,
    #         penalty_present,
    #     )

    # def sample_softmax(self, candidates: "_LlamaTokenDataArray"):
    #     llama_cpp.llama_sample_softmax(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # def sample_top_k(self, candidates: "_LlamaTokenDataArray", k: int, min_keep: int):
    #     llama_cpp.llama_sample_top_k(
    #         self.ctx, llama_cpp.byref(candidates.candidates), k, min_keep
    #     )

    # def sample_top_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
    #     llama_cpp.llama_sample_top_p(
    #         self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_min_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
    #     llama_cpp.llama_sample_min_p(
    #         self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_tail_free(
    #     self, candidates: "_LlamaTokenDataArray", z: float, min_keep: int
    # ):
    #     llama_cpp.llama_sample_tail_free(
    #         self.ctx, llama_cpp.byref(candidates.candidates), z, min_keep
    #     )

    # def sample_typical(
    #     self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int
    # ):
    #     llama_cpp.llama_sample_typical(
    #         self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_temp(self, candidates: "_LlamaTokenDataArray", temp: float):
    #     llama_cpp.llama_sample_temp(
    #         self.ctx, llama_cpp.byref(candidates.candidates), temp
    #     )

    # def sample_grammar(self, candidates: "_LlamaTokenDataArray", grammar: LlamaGrammar):
    #     llama_cpp.llama_sample_grammar(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #         grammar.grammar,
    #     )

    # def sample_token_mirostat(
    #     self,
    #     candidates: "_LlamaTokenDataArray",
    #     tau: float,
    #     eta: float,
    #     m: int,
    #     mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    # ) -> int:
    #     return llama_cpp.llama_sample_token_mirostat(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #         tau,
    #         eta,
    #         m,
    #         mu,
    #     )

    # def sample_token_mirostat_v2(
    #     self,
    #     candidates: "_LlamaTokenDataArray",
    #     tau: float,
    #     eta: float,
    #     mu: llama_cpp.CtypesPointerOrRef[ctypes.c_float],
    # ) -> int:
    #     return llama_cpp.llama_sample_token_mirostat_v2(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #         tau,
    #         eta,
    #         mu,
    #     )

    # def sample_token_greedy(self, candidates: "_LlamaTokenDataArray") -> int:
    #     return llama_cpp.llama_sample_token_greedy(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # def sample_token(self, candidates: "_LlamaTokenDataArray") -> int:
    #     return llama_cpp.llama_sample_token(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # # Grammar
    # def grammar_accept_token(self, grammar: LlamaGrammar, token: int):
    #     llama_cpp.llama_grammar_accept_token(grammar.grammar, self.ctx, token)

    # def reset_timings(self):
    #     llama_cpp.llama_perf_context_reset(self.ctx)

    # def print_timings(self):
    #     llama_cpp.llama_perf_context_print(self.ctx)

    # Performace

    # cdef llama_perf_context_data llama_perf_context(const llama_context * ctx)
    # cdef void llama_perf_context_print(const llama_context * ctx)
    # cdef void llama_perf_context_reset(      llama_context * ctx)



    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return LlamaContext()


cdef class LlamaBatch:
    """Input data for llama_decode
    
    A llama_batch object can contain input about one or many sequences
    The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens

    - token  : the token ids of the input (used when embd is NULL)
    - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    - pos    : the positions of the respective token in the sequence
               (if set to NULL, the token position will be tracked automatically by llama_decode)
    - seq_id : the sequence to which the respective token belongs
               (if set to NULL, the sequence ID will be assumed to be 0)
    - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
               (if set to NULL, only the logits for last token will be returned)
    """
    cdef llama_cpp.llama_batch p
    cdef int _n_tokens
    cdef public int embd
    cdef public int n_seq_max
    cdef public bint verbose
    cdef bint owner

    def __init__(self, *, n_tokens: int, embd: int, n_seq_max: int, verbose: bool = True):
        self._n_tokens = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose
        self.owner = True

        self.p = llama_cpp.llama_batch_init(
            self._n_tokens, self.embd, self.n_seq_max
        )

    def __dealloc__(self):
        if self.owner is True:
            llama_cpp.llama_batch_free(self.p)

    def close(self):
        self.__dealloc__()

    @property
    def n_tokens(self) -> int:
        return self.p.n_tokens

    def reset(self):
        self.p.n_tokens = 0

    def add(self, llama_cpp.llama_token id, llama_cpp.llama_pos pos, list[int] seq_ids, bint logits):
        cdef vector[llama_cpp.llama_seq_id] _seq_ids
        for i in seq_ids:
            _seq_ids.push_back(i)
        llama_cpp.common_batch_add(self.p, id, pos, _seq_ids, logits)

    def clear(self):
        llama_cpp.common_batch_clear(self.p)

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
        n_tokens = len(batch)
        self.p.n_tokens = n_tokens
        for i in range(n_tokens):
            self.p.token[i] = batch[i]
            self.p.pos[i] = n_past + i
            self.p.seq_id[i][0] = 0
            self.p.n_seq_id[i] = 1
            self.p.logits[i] = logits_all
        self.p.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        n_tokens = len(batch)
        n_tokens0 = self.p.n_tokens
        self.p.n_tokens += n_tokens
        for i in range(n_tokens):
            j = n_tokens0 + i
            self.p.token[j] = batch[i]
            self.p.pos[j] = i
            self.p.seq_id[j][0] = seq_id
            self.p.n_seq_id[j] = 1
            self.p.logits[j] = logits_all
        self.p.logits[n_tokens - 1] = True

    def set_last_logits_to_true(self):
        self.p.logits[self.p.n_tokens - 1] = True




    # typedef struct llama_token_data {
    #     llama_token id; // token id
    #     float logit;    // log-odds of the token
    #     float p;        // probability of the token
    # } llama_token_data;

    # typedef struct llama_token_data_array {
    #     # TODO: consider SoA
    #     llama_token_data * data;
    #     size_t size;
    #     int64_t selected; # this is the index in the data array (i.e. not the token id)
    #     bool sorted;
    # } llama_token_data_array;

# see: 
#   https://groups.google.com/g/cython-users/c/TbLbXdi0_h4/m/vf9iCcFtGHkJ
#   https://groups.google.com/g/cython-users/c/ytfJDZ_1DAI/m/xUHAm9uynacJ
#   https://groups.google.com/g/cython-users/c/hGMPOI0BNpk/m/0iy1Yi9tCAAJ

# FIXME: convert to buffer protocol or memoryview
cdef class LlamaTokenDataArray:
    """Intermediate Cython wrapper for a llama.cpp llama_batch."""
    cdef llama_cpp.llama_token_data_array * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    # def __dealloc__(self):
    #     if self.candidates is not NULL and self.owner is True:
    #         llama_cpp.llama_batch_free(self.batch[0])
    #         self.batch = NULL
    
    # def __init__(self, *, n_vocab: int):
    #     self.n_vocab = n_vocab
    #     self.candidates_data = np.recarray(
    #         (self.n_vocab,),
    #         dtype=np.dtype(
    #             [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
    #         ),
    #     )
    #     self.candidates = llama_cpp.llama_token_data_array(
    #         data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
    #         size=self.n_vocab,
    #         sorted=False,
    #     )
    #     self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc)  # type: ignore
    #     self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)

    # def copy_logits(self, logits: npt.NDArray[np.single]):
    #     self.candidates_data.id[:] = self.default_candidates_data_id
    #     self.candidates_data.logit[:] = logits
    #     self.candidates_data.p[:] = self.default_candidates_data_p
    #     self.candidates.sorted = False
    #     self.candidates.size = self.n_vocab






def llama_backend_init():
    """Initialize the llama + ggml backend

    If numa is true, use NUMA optimizations
    Call once at the start of the program
    """
    llama_cpp.llama_backend_init()

def llama_numa_init(llama_cpp.ggml_numa_strategy numa):
    llama_cpp.llama_numa_init(numa)

def common_model_params_to_llama(params: CommonParams) -> ModelParams:
    cdef llama_cpp.llama_model_params model_params = llama_cpp.common_model_params_to_llama(params.p)
    return ModelParams.from_instance(model_params)

def common_context_params_to_llama(params: CommonParams) -> ContextParams:
    return ContextParams.from_common_params(params)

def llama_sampler_chain_default_params() -> SamplerChainParams:
    return SamplerChainParams()

def llama_model_quantize_default_params() -> ModelQuantizeParams:
    return ModelQuantizeParams()

def common_tokenize(LlamaContext ctx, str text, bint add_special, bint parse_special = False):
    return llama_cpp.common_tokenize(<const llama_cpp.llama_context *>ctx.ptr, <string>text.encode(), add_special, parse_special)

def common_tokenize_from_model(LlamaModel model, str text, bint add_special, bint parse_special = False):
    return llama_cpp.common_tokenize(<const llama_cpp.llama_model *>model.ptr, <string>text.encode(), add_special, parse_special)

def llama_n_ctx(LlamaContext ctx) -> int:
    return llama_cpp.llama_n_ctx(ctx.ptr)

def common_token_to_piece(LlamaContext ctx, int token, bint special = True) -> str:
    return llama_cpp.common_token_to_piece(ctx.ptr, token, special).decode()

def common_batch_add(LlamaBatch batch, llama_cpp.llama_token id, llama_cpp.llama_pos pos, list[int] seq_ids, bint logits):
    return llama_cpp.common_batch_add(batch.p, id, pos, seq_ids, logits)

# def llama_decode(LlamaContext ctx, LlamaBatch batch) -> int:
#     return llama_cpp.llama_decode(ctx.ptr, batch.p)

# def llama_new_context_with_model(LlamaModel model, ContextParams params) -> LlamaContext:
#     cdef llama_cpp.llama_context * ctx = llama_cpp.llama_new_context_with_model(model.ptr, params.p)
#     return LlamaContext.from_ptr(ctx)

# def llama_load_model_from_file(str path_model, ModelParams params) -> LlamaModel:
#     cdef llama_cpp.llama_model * model = llama_cpp.llama_load_model_from_file(path_model.encode(), params.p)
#     return LlamaModel.from_ptr(model)

def ggml_time_us() -> int:
    return llama_cpp.ggml_time_us()

def llama_time_us() -> int:
    return llama_cpp.llama_time_us()

def llama_max_devices() -> int:
    return llama_cpp.llama_max_devices()

def llama_supports_mmap() -> bool:
    return llama_cpp.llama_supports_mmap()

def llama_supports_mlock() -> bool:
    return llama_cpp.llama_supports_mlock()

def llama_supports_gpu_offload() -> bool:
    return llama_cpp.llama_supports_gpu_offload()

def llama_supports_rpc() -> bool:
    return llama_cpp.llama_supports_rpc()



def llama_sampler_sample(LlamaSampler smplr, LlamaContext ctx, int idx) -> int:
    return llama_cpp.llama_sampler_sample(smplr.ptr, ctx.ptr, idx)

def llama_sampler_accept(LlamaSampler smplr, llama_cpp.llama_token id):
    llama_cpp.llama_sampler_accept(smplr.ptr, id)

def llama_token_is_eog(LlamaModel model, llama_cpp.llama_token token) -> bool:
    return llama_cpp.llama_token_is_eog(model.ptr, token)

def common_batch_clear(LlamaBatch batch):
    llama_cpp.common_batch_clear(batch.p)

def llama_backend_free():
    """Call once at the end of the program - currently only used for MPI"""
    llama_cpp.llama_backend_free()
