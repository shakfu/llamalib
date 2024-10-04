# distutils: language = c++

from libc.stdlib cimport malloc, calloc, realloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport llama_cpp

import os
from typing import Optional, Sequence


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
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, params: SamplerChainParams):
        self.ptr = llama_cpp.llama_sampler_chain_init(params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to init Sampler")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_sampler_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaSampler init_greedy():
        cdef LlamaSampler wrapper = LlamaSampler.__new__(LlamaSampler)
        wrapper.ptr = llama_cpp.llama_sampler_init_greedy()
        return wrapper

    def chain_add(self, smplr: LlamaSampler):
        smplr.owner = False
        llama_cpp.llama_sampler_chain_add(self.ptr, smplr.ptr)

    def chain_add_greedy(self):
        self.chain_add(LlamaSampler.init_greedy())


# cdef class CpuParams:
#     cdef llama_cpp.cpu_params p

#     @staticmethod
#     cdef CpuParams from_instance(llama_cpp.cpu_params params):
#         cdef CpuParams wrapper = CpuParams.__new__(CpuParams)
#         wrapper.p = params
#         return wrapper

#     @property
#     def n_threads(self) -> int:
#         """number of threads."""
#         return self.p.n_threads

#     @n_threads.setter
#     def n_threads(self, value: int):
#         self.p.n_threads = value

#     # @property
#     # def cpumask(self) -> list[bool]:
#     #     """CPU affinity mask."""
#     #     return self.p.cpumask

#     # @cpumask.setter
#     # def cpumask(self, value: list[bool]):
#     #     self.p.cpumask = value

#     @property
#     def mask_valid(self) -> bool:
#         """Default: any CPU."""
#         return self.p.mask_valid

#     @mask_valid.setter
#     def mask_valid(self, value: bool):
#         self.p.mask_valid = value

#     @property
#     def priority(self) -> llama_cpp.ggml_sched_priority:
#         """Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)."""
#         return self.p.priority

#     @priority.setter
#     def priority(self, value: llama_cpp.ggml_sched_priority):
#         self.p.priority = value

#     # @property
#     # def llama_cpp.ggml_sched_strict_cpu strict_cpu(self):
#     #     """Use strict CPU placement."""
#     #     return self.p.strict_cpu

#     # @strict_cpu.setter
#     # def strict_cpu(self, llama_cpp.ggml_sched_strict_cpu value):
#     #     self.p.strict_cpu = value

#     # @property
#     # def poll(self) -> llama_cpp.ggml_sched_poll:
#     #     """Use strict CPU placement"""
#     #     return self.p.poll

#     # @poll.setter
#     # def poll(self, value: llama_cpp.ggml_sched_poll):
#     #     self.p.poll = value



cdef class GptParams: # WIP!
    cdef llama_cpp.gpt_params p
    
    @property
    def n_threads(self) -> int:
        """number of threads."""
        return self.p.cpuparams.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.p.cpuparams.n_threads = value

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
    def sparams(self) -> llama_cpp.gpt_sampler_params:
        """gpt sampler params."""
        return self.p.sparams

    @sparams.setter
    def sparams(self, value: llama_cpp.gpt_sampler_params):
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
    def in_files(self) -> [str]:
        """all input files."""
        result = []
        for i in range(self.p.in_files.size()):
            result.append(self.p.in_files[i].decode())
        return result

    @in_files.setter
    def in_files(self, files: [str]):
        self.p.in_files.clear()
        for i in files:
            self.p.in_files.push_back(i.encode('utf8'))

    @property
    def antiprompt(self) -> [str]:
        """strings upon which more user input is prompted (a.k.a. reverse prompts)."""
        result = []
        for i in range(self.p.antiprompt.size()):
            result.append(self.p.antiprompt[i].decode())
        return result

    @antiprompt.setter
    def antiprompt(self, values: [str]):
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
    def image(self) -> [str]:
        """paths to image file(s)"""
        result = []
        for i in range(self.p.image.size()):
            result.append(self.p.image[i].decode())
        return result

    @image.setter
    def image(self, files: [str]):
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

    @property
    def system_prompt(self) -> str:
        """system prompt"""
        return self.p.system_prompt.decode()

    @system_prompt.setter
    def system_prompt(self, value: str):
        self.p.system_prompt = value.encode('utf8')

    @property
    def enable_chat_template(self) -> bool:
        """enable chat template"""
        return self.p.enable_chat_template

    @enable_chat_template.setter
    def enable_chat_template(self, value: bool):
        self.p.enable_chat_template = value

    @property
    def api_keys(self) -> [str]:
        """list of api keys"""
        result = []
        for i in range(self.p.api_keys.size()):
            result.append(self.p.api_keys[i].decode())
        return result

    @api_keys.setter
    def api_keys(self, files: [str]):
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
    def endpoint_slots(self) -> bool:
        """endpoint slots"""
        return self.p.endpoint_slots

    @endpoint_slots.setter
    def endpoint_slots(self, value: bool):
        self.p.endpoint_slots = value

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

    # FIXME: name collision
    # def vocab_type(self) -> llama_cpp.llama_vocab_type:
    #     # assert self.model is not None
    #     return llama_cpp.get_llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_n_vocab(self.ptr)

    def n_ctx_train(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_n_ctx_train(self.ptr)

    def n_embd(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_n_embd(self.ptr)

    def rope_freq_scale_train(self) -> float:
        assert self.ptr is not NULL
        return llama_cpp.llama_rope_freq_scale_train(self.ptr)

    def desc(self) -> str:
        cdef char buf[1024]
        assert self.ptr is not NULL
        llama_cpp.llama_model_desc(self.ptr, buf, 1024)
        return buf.decode("utf-8")

    def size(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_model_size(self.ptr)

    def n_params(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_model_n_params(self.ptr)

    def get_tensor(self, name: str) -> GGMLTensor:
        assert self.ptr is not NULL
        cdef llama_cpp.ggml_tensor * tensor = llama_cpp.llama_get_model_tensor(self.ptr, name.encode("utf-8"))
        return GGMLTensor.from_ptr(tensor)

    # def get_tensor(self, name: str) -> ctypes.c_void_p:
    #     return llama_cpp.llama_get_model_tensor(self.ptr, name.encode("utf-8"))

    # def apply_lora_from_file(
    #     self,
    #     lora_path: str,
    #     scale: float,
    #     path_base_model: Optional[str],
    #     n_threads: int,
    # ):
    #     return llama_cpp.llama_model_apply_lora_from_file(
    #         self.ptr,
    #         lora_path.encode("utf-8"),
    #         scale,
    #         (
    #             path_base_model.encode("utf-8")
    #             if path_base_model is not None
    #             else ctypes.c_char_p(0)
    #         ),
    #         n_threads,
    #     )

    # def apply_lora_from_file(
    #     self,
    #     lora_path: str,
    #     scale: float,
    #     path_base_model: Optional[str],
    #     n_threads: int,
    # ):
    #     assert self.ptr is not None
    #     return llama_cpp.llama_model_apply_lora_from_file(
    #         self.ptr,
    #         lora_path.encode("utf-8"),
    #         scale,
    #         (
    #             path_base_model.encode("utf-8")
    #             if path_base_model is not None
    #             else ctypes.c_char_p(0)
    #         ),
    #         n_threads,
    #     )

    # Vocab

    def token_get_text(self, token: int) -> str:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_get_text(self.ptr, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_get_score(self.ptr, token)

    def token_get_attr(self, token: int) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_get_attr(self.ptr, token)

    # Special tokens

    def token_bos(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_bos(self.ptr)

    def token_eos(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_eos(self.ptr)

    def token_cls(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_cls(self.ptr)

    def token_sep(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_sep(self.ptr)

    def token_nl(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_nl(self.ptr)

    def token_prefix(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_prefix(self.ptr)

    def token_middle(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_middle(self.ptr)

    def token_suffix(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_suffix(self.ptr)

    def token_eot(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_token_eot(self.ptr)

    def add_bos_token(self) -> bool:
        assert self.ptr is not NULL
        return llama_cpp.llama_add_bos_token(self.ptr)

    def add_eos_token(self) -> bool:
        assert self.ptr is not NULL
        return llama_cpp.llama_add_eos_token(self.ptr)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        assert self.ptr is not NULL
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
    cdef ContextParams from_gpt_params(GptParams params):
        cdef ContextParams wrapper = ContextParams.__new__(ContextParams)
        wrapper.p = llama_cpp.llama_context_params_from_gpt_params(params.p)
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
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(
        self,
        *,
        model: LlamaModel,
        # params: llama_cpp.llama_context_params,
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

    def close(self):
        self.__dealloc__()

    def n_ctx(self) -> int:
        assert self.ptr is not NULL
        return llama_cpp.llama_n_ctx(self.ptr)

    # FIXME: name collision
    # def pooling_type(self) -> int:
    #     assert self.ptr is not NULL
    #     return llama_cpp.llama_pooling_type(self.ptr)

    def kv_cache_clear(self):
        assert self.ptr is not NULL
        llama_cpp.llama_kv_cache_clear(self.ptr)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        assert self.ptr is not NULL
        llama_cpp.llama_kv_cache_seq_rm(self.ptr, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        assert self.ptr is not NULL
        llama_cpp.llama_kv_cache_seq_cp(self.ptr, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        assert self.ptr is not NULL
        llama_cpp.llama_kv_cache_seq_keep(self.ptr, seq_id)

    def kv_cache_seq_shift(self, seq_id: int, p0: int, p1: int, shift: int):
        assert self.ptr is not NULL
        llama_cpp.llama_kv_cache_seq_add(self.ptr, seq_id, p0, p1, shift)

    # def get_state_size(self) -> int:
    #     assert self.ptr is not NULL
    #     return llama_cpp.llama_get_state_size(self.ptr)

    # # TODO: copy_state_data

    # # TODO: set_state_data

    # # TODO: llama_load_session_file

    # # TODO: llama_save_session_file

    # def decode(self, batch: "LlamaBatch"):
    #     assert self.ptr is not None
    #     assert batch.ptr is not None
    #     return_code = llama_cpp.llama_decode(
    #         self.ptr,
    #         batch.ptr,
    #     )
    #     if return_code != 0:
    #         raise RuntimeError(f"llama_decode returned {return_code}")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        assert self.ptr is not NULL
        llama_cpp.llama_set_n_threads(self.ptr, n_threads, n_threads_batch)

    # def get_logits(self):
    #     assert self.ptr is not NULL
    #     return llama_cpp.llama_get_logits(self.ptr)

    # def get_logits_ith(self, i: int):
    #     assert self.ptr is not NULL
    #     return llama_cpp.llama_get_logits_ith(self.ptr, i)

    # def get_embeddings(self):
    #     assert self.ptr is not NULL
    #     return llama_cpp.llama_get_embeddings(self.ptr)

    # Sampling functions

    # def sample_repetition_penalties(
    #     self,
    #     candidates: "_LlamaTokenDataArray",
    #     last_tokens_data: "llama_cpp.Array[llama_cpp.llama_token]",
    #     penalty_last_n: int,
    #     penalty_repeat: float,
    #     penalty_freq: float,
    #     penalty_present: float,
    # ):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_repetition_penalties(
    #         self.ptr,
    #         llama_cpp.byref(candidates.candidates),
    #         last_tokens_data,
    #         penalty_last_n,
    #         penalty_repeat,
    #         penalty_freq,
    #         penalty_present,
    #     )

    # def sample_softmax(self, candidates: "_LlamaTokenDataArray"):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_softmax(
    #         self.ptr,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # def sample_top_k(self, candidates: "_LlamaTokenDataArray", k: int, min_keep: int):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_top_k(
    #         self.ptr, llama_cpp.byref(candidates.candidates), k, min_keep
    #     )

    # def sample_top_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_top_p(
    #         self.ptr, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_min_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_min_p(
    #         self.ptr, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_tail_free(
    #     self, candidates: "_LlamaTokenDataArray", z: float, min_keep: int
    # ):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_tail_free(
    #         self.ptr, llama_cpp.byref(candidates.candidates), z, min_keep
    #     )

    # def sample_typical(
    #     self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int
    # ):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_typical(
    #         self.ptr, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_temp(self, candidates: "_LlamaTokenDataArray", temp: float):
    #     assert self.ptr is not None
    #     llama_cpp.llama_sample_temp(
    #         self.ptr, llama_cpp.byref(candidates.candidates), temp
    #     )

    # def sample_grammar(self, candidates: "_LlamaTokenDataArray", grammar: LlamaGrammar):
    #     assert self.ptr is not None
    #     assert grammar.grammar is not None
    #     llama_cpp.llama_sample_grammar(
    #         self.ptr,
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
    #     assert self.ptr is not None
    #     return llama_cpp.llama_sample_token_mirostat(
    #         self.ptr,
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
    #     assert self.ptr is not None
    #     return llama_cpp.llama_sample_token_mirostat_v2(
    #         self.ptr,
    #         llama_cpp.byref(candidates.candidates),
    #         tau,
    #         eta,
    #         mu,
    #     )

    # def sample_token_greedy(self, candidates: "_LlamaTokenDataArray") -> int:
    #     assert self.ptr is not None
    #     return llama_cpp.llama_sample_token_greedy(
    #         self.ptr,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # def sample_token(self, candidates: "_LlamaTokenDataArray") -> int:
    #     assert self.ptr is not None
    #     return llama_cpp.llama_sample_token(
    #         self.ptr,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # Grammar
    # def grammar_accept_token(self, grammar: LlamaGrammar, token: int):
    #     assert self.ptr is not None
    #     assert grammar.grammar is not None
    #     llama_cpp.llama_grammar_accept_token(grammar.grammar, self.ptr, token)


    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return LlamaContext()


cdef class LlamaBatch:
    """Intermediate Python wrapper for a llama.cpp llama_batch."""
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
        # assert self.p is not NULL
        return self.p.n_tokens

    def reset(self):
        # assert self.p is not NULL
        self.p.n_tokens = 0

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
        # assert self.p is not NULL
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
        # assert self.p is not NULL
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


# FIXME: convert to buffer protocol or memoryview
# class LlamaTokenDataArray:
#     """Intermediate Python wrapper for a llama.cpp llama_batch."""
#     cdef llama_cpp.llama_token_data_array * candidates
#     cdef public int n_vocab
#     cdef public bint verbose
#     cdef bint owner

#     def __cinit__(self):
#         self.candidates = NULL
#         self.owner = True

#     def __init__(self, *, n_vocab: int):
#         self.n_vocab = n_vocab
#         self.candidates_data = np.recarray(
#             (self.n_vocab,),
#             dtype=np.dtype(
#                 [("id", np.intc), ("logit", np.single), ("p", np.single)], align=True
#             ),
#         )
#         self.candidates = llama_cpp.llama_token_data_array(
#             data=self.candidates_data.ctypes.data_as(llama_cpp.llama_token_data_p),
#             size=self.n_vocab,
#             sorted=False,
#         )
#         self.default_candidates_data_id = np.arange(self.n_vocab, dtype=np.intc)  # type: ignore
#         self.default_candidates_data_p = np.zeros(self.n_vocab, dtype=np.single)

#     def copy_logits(self, logits: npt.NDArray[np.single]):
#         self.candidates_data.id[:] = self.default_candidates_data_id
#         self.candidates_data.logit[:] = logits
#         self.candidates_data.p[:] = self.default_candidates_data_p
#         self.candidates.sorted = False
#         self.candidates.size = self.n_vocab

#     def __dealloc__(self):
#         if self.candidates is not NULL and self.owner is True:
#             llama_cpp.llama_batch_free(self.batch[0])
#             self.batch = NULL


def llama_backend_init():
    llama_cpp.llama_backend_init()

def llama_numa_init(llama_cpp.ggml_numa_strategy numa):
    llama_cpp.llama_numa_init(numa)

def llama_model_params_from_gpt_params(params: GptParams) -> ModelParams:
    cdef llama_cpp.llama_model_params model_params = llama_cpp.llama_model_params_from_gpt_params(params.p)
    return ModelParams.from_instance(model_params)

def llama_context_params_from_gpt_params(params: GptParams) -> ContextParams:
    return ContextParams.from_gpt_params(params)

def llama_sampler_chain_default_params() -> SamplerChainParams:
    return SamplerChainParams()

def llama_tokenize(LlamaContext ctx, str text, bint add_special, bint parse_special = False):
    return llama_cpp.llama_tokenize(ctx.ptr, text.encode(), add_special, parse_special)

# def llama_tokenize(LlamaModel model, str text, bint add_special, bint parse_special = False):
#     return llama_cpp.llama_tokenize(model.ptr, text.encode(), add_special, parse_special)

def llama_n_ctx(LlamaContext ctx) -> int:
    return llama_cpp.llama_n_ctx(ctx.ptr)

def llama_token_to_piece(LlamaContext ctx, int token, bint special = True) -> str:
    return llama_cpp.llama_token_to_piece2(ctx.ptr, token, special).decode()

def llama_batch_add(LlamaBatch batch, llama_cpp.llama_token id, llama_cpp.llama_pos pos, list[int] seq_ids, bint logits):
    return llama_cpp.llama_batch_add(batch.p, id, pos, seq_ids, logits)

def llama_decode(LlamaContext ctx, LlamaBatch batch) -> int:
    return llama_cpp.llama_decode(ctx.ptr, batch.p)

def ggml_time_us() -> int:
    return llama_cpp.ggml_time_us()

def llama_sampler_sample(LlamaSampler smplr, LlamaContext ctx, int idx) -> int:
    return llama_cpp.llama_sampler_sample(smplr.ptr, ctx.ptr, idx)

def llama_sampler_accept(LlamaSampler smplr, llama_cpp.llama_token id):
    llama_cpp.llama_sampler_accept(smplr.ptr, id)

def llama_token_is_eog(LlamaModel model, llama_cpp.llama_token token) -> bool:
    return llama_cpp.llama_token_is_eog(model.ptr, token)

def llama_batch_clear(LlamaBatch batch):
    llama_cpp.llama_batch_clear(batch.p)

def llama_backend_free():
    llama_cpp.llama_backend_free()
