# distutils: language = c++

from libc.stdlib cimport malloc, calloc, realloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string

cimport llama_cpp

import os
from typing import Optional, Sequence


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


cdef class ModelParams:
    cdef llama_cpp.llama_model_params p

    def __init__(self):
        self.p = llama_cpp.llama_model_default_params()

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
    cdef llama_cpp.llama_model * model
    cdef public ModelParams params
    cdef public str path_model
    cdef public bint verbose
    cdef bint owner

    def __cinit__(self):
        self.model = NULL
        self.owner = True

    def __init__(self, *, path_model: str, params: Optional[ModelParams] = None, verbose: bool = True):
        self.path_model = path_model
        self.params = params if params else ModelParams()
        self.verbose = verbose

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        # with suppress_stdout_stderr(disable=verbose):
        self.model = llama_cpp.llama_load_model_from_file(
            self.path_model.encode("utf-8"), 
            self.params.p
        )

        if self.model is NULL:
            raise ValueError(f"Failed to load model from file: {path_model}")

    def __dealloc__(self):
        if self.model is not NULL and self.owner is True:
            llama_cpp.llama_free_model(self.model)
            self.model = NULL

    # FIXME: name collision
    # def vocab_type(self) -> llama_cpp.llama_vocab_type:
    #     # assert self.model is not None
    #     return llama_cpp.get_llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_n_vocab(self.model)

    def n_ctx_train(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_n_ctx_train(self.model)

    def n_embd(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_n_embd(self.model)

    def rope_freq_scale_train(self) -> float:
        assert self.model is not NULL
        return llama_cpp.llama_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        cdef char buf[1024]
        assert self.model is not NULL
        llama_cpp.llama_model_desc(self.model, buf, 1024)
        return buf.decode("utf-8")

    def size(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_model_size(self.model)

    def n_params(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_model_n_params(self.model)

    def get_tensor(self, name: str) -> GGMLTensor:
        assert self.model is not NULL
        cdef llama_cpp.ggml_tensor * tensor = llama_cpp.llama_get_model_tensor(self.model, name.encode("utf-8"))
        return GGMLTensor.from_ptr(tensor)

    # def get_tensor(self, name: str) -> ctypes.c_void_p:
    #     return llama_cpp.llama_get_model_tensor(self.model, name.encode("utf-8"))

    # def apply_lora_from_file(
    #     self,
    #     lora_path: str,
    #     scale: float,
    #     path_base_model: Optional[str],
    #     n_threads: int,
    # ):
    #     return llama_cpp.llama_model_apply_lora_from_file(
    #         self.model,
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
    #     assert self.model is not None
    #     return llama_cpp.llama_model_apply_lora_from_file(
    #         self.model,
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
        assert self.model is not NULL
        return llama_cpp.llama_token_get_text(self.model, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        assert self.model is not NULL
        return llama_cpp.llama_token_get_score(self.model, token)

    def token_get_attr(self, token: int) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_get_attr(self.model, token)

    # Special tokens

    def token_bos(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_bos(self.model)

    def token_eos(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_eos(self.model)

    def token_cls(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_cls(self.model)

    def token_sep(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_sep(self.model)

    def token_nl(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_nl(self.model)

    def token_prefix(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_prefix(self.model)

    def token_middle(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_middle(self.model)

    def token_suffix(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_suffix(self.model)

    def token_eot(self) -> int:
        assert self.model is not NULL
        return llama_cpp.llama_token_eot(self.model)

    def add_bos_token(self) -> bool:
        assert self.model is not NULL
        return llama_cpp.llama_add_bos_token(self.model)

    def add_eos_token(self) -> bool:
        assert self.model is not NULL
        return llama_cpp.llama_add_eos_token(self.model)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        assert self.model is not NULL
        cdef int n_ctx = self.n_ctx_train()
        cdef vector[llama_cpp.llama_token] tokens
        tokens.reserve(n_ctx)
        n_tokens = llama_cpp.llama_tokenize(
            self.model, text, len(text), tokens.data(), n_ctx, add_bos, special
        )
        if n_tokens < 0:
            n_tokens = abs(n_tokens)
            # tokens = (llama_cpp.llama_token * n_tokens)()
            n_tokens = llama_cpp.llama_tokenize(
                self.model, text, len(text), tokens.data(), n_tokens, add_bos, special
            )
            if n_tokens < 0:
                raise RuntimeError()
                # raise RuntimeError(
                #     f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                # )

        return tokens[:n_tokens]

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        cdef char buf[32]
        llama_cpp.llama_token_to_piece(self.model, token, buf, 32, 0, special)
        return buf.decode()
        # return bytes(buf)

    def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
        assert self.model is not NULL
        output = b""
        size = 32
        cdef char buffer[32]
        for token in tokens:
            n = llama_cpp.llama_token_to_piece(
                self.model, int(token), buffer, size, 0, special
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
        assert self.model is not NULL
        # iterate over model keys
        for i in range(llama_cpp.llama_model_meta_count(self.model)):
            nbytes = llama_cpp.llama_model_meta_key_by_index(
                self.model, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama_cpp.llama_model_meta_key_by_index(
                    self.model, i, buffer, buffer_size
                )
            key = buffer.decode("utf-8")
            nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                self.model, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                    self.model, i, buffer, buffer_size
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
    cdef llama_cpp.llama_context * ctx
    cdef public LlamaModel model
    cdef public ContextParams params
    cdef public bint verbose
    cdef bint owner

    def __cinit__(self):
        self.ctx = NULL
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

        # self.ctx = None

        assert self.model.model is not NULL

        self.ctx = llama_cpp.llama_new_context_with_model(self.model.model, self.params.p)

        if self.ctx is NULL:
            raise ValueError("Failed to create llama_context")

    def __dealloc__(self):
        if self.ctx is not NULL and self.owner is True:
            llama_cpp.llama_free(self.ctx)
            self.ctx = NULL

    def close(self):
        self.__dealloc__()

    def n_ctx(self) -> int:
        assert self.ctx is not NULL
        return llama_cpp.llama_n_ctx(self.ctx)

    # FIXME: name collision
    # def pooling_type(self) -> int:
    #     assert self.ctx is not NULL
    #     return llama_cpp.llama_pooling_type(self.ctx)

    def kv_cache_clear(self):
        assert self.ctx is not NULL
        llama_cpp.llama_kv_cache_clear(self.ctx)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        assert self.ctx is not NULL
        llama_cpp.llama_kv_cache_seq_rm(self.ctx, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        assert self.ctx is not NULL
        llama_cpp.llama_kv_cache_seq_cp(self.ctx, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        assert self.ctx is not NULL
        llama_cpp.llama_kv_cache_seq_keep(self.ctx, seq_id)

    def kv_cache_seq_shift(self, seq_id: int, p0: int, p1: int, shift: int):
        assert self.ctx is not NULL
        llama_cpp.llama_kv_cache_seq_add(self.ctx, seq_id, p0, p1, shift)

    # def get_state_size(self) -> int:
    #     assert self.ctx is not NULL
    #     return llama_cpp.llama_get_state_size(self.ctx)

    # # TODO: copy_state_data

    # # TODO: set_state_data

    # # TODO: llama_load_session_file

    # # TODO: llama_save_session_file

    # def decode(self, batch: "_LlamaBatch"):
    #     assert self.ctx is not None
    #     assert batch.batch is not None
    #     return_code = llama_cpp.llama_decode(
    #         self.ctx,
    #         batch.batch,
    #     )
    #     if return_code != 0:
    #         raise RuntimeError(f"llama_decode returned {return_code}")

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        assert self.ctx is not NULL
        llama_cpp.llama_set_n_threads(self.ctx, n_threads, n_threads_batch)

    # def get_logits(self):
    #     assert self.ctx is not NULL
    #     return llama_cpp.llama_get_logits(self.ctx)

    # def get_logits_ith(self, i: int):
    #     assert self.ctx is not NULL
    #     return llama_cpp.llama_get_logits_ith(self.ctx, i)

    # def get_embeddings(self):
    #     assert self.ctx is not NULL
    #     return llama_cpp.llama_get_embeddings(self.ctx)

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
    #     assert self.ctx is not None
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
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_softmax(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # def sample_top_k(self, candidates: "_LlamaTokenDataArray", k: int, min_keep: int):
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_top_k(
    #         self.ctx, llama_cpp.byref(candidates.candidates), k, min_keep
    #     )

    # def sample_top_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_top_p(
    #         self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_min_p(self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int):
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_min_p(
    #         self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_tail_free(
    #     self, candidates: "_LlamaTokenDataArray", z: float, min_keep: int
    # ):
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_tail_free(
    #         self.ctx, llama_cpp.byref(candidates.candidates), z, min_keep
    #     )

    # def sample_typical(
    #     self, candidates: "_LlamaTokenDataArray", p: float, min_keep: int
    # ):
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_typical(
    #         self.ctx, llama_cpp.byref(candidates.candidates), p, min_keep
    #     )

    # def sample_temp(self, candidates: "_LlamaTokenDataArray", temp: float):
    #     assert self.ctx is not None
    #     llama_cpp.llama_sample_temp(
    #         self.ctx, llama_cpp.byref(candidates.candidates), temp
    #     )

    # def sample_grammar(self, candidates: "_LlamaTokenDataArray", grammar: LlamaGrammar):
    #     assert self.ctx is not None
    #     assert grammar.grammar is not None
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
    #     assert self.ctx is not None
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
    #     assert self.ctx is not None
    #     return llama_cpp.llama_sample_token_mirostat_v2(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #         tau,
    #         eta,
    #         mu,
    #     )

    # def sample_token_greedy(self, candidates: "_LlamaTokenDataArray") -> int:
    #     assert self.ctx is not None
    #     return llama_cpp.llama_sample_token_greedy(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # def sample_token(self, candidates: "_LlamaTokenDataArray") -> int:
    #     assert self.ctx is not None
    #     return llama_cpp.llama_sample_token(
    #         self.ctx,
    #         llama_cpp.byref(candidates.candidates),
    #     )

    # Grammar
    # def grammar_accept_token(self, grammar: LlamaGrammar, token: int):
    #     assert self.ctx is not None
    #     assert grammar.grammar is not None
    #     llama_cpp.llama_grammar_accept_token(grammar.grammar, self.ctx, token)


    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return LlamaContext()


cdef class LlamaBatch:
    """Intermediate Python wrapper for a llama.cpp llama_batch."""
    cdef llama_cpp.llama_batch * batch
    cdef int _n_tokens
    cdef public int embd
    cdef public int n_seq_max
    cdef public bint verbose
    cdef bint owner

    def __cinit__(self):
        self.batch = NULL
        self.owner = True

    def __init__(self, *, n_tokens: int, embd: int, n_seq_max: int, verbose: bool = True):
        self._n_tokens = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose

        self.batch[0] = llama_cpp.llama_batch_init(
            self._n_tokens, self.embd, self.n_seq_max
        )

    def __dealloc__(self):
        if self.batch is not NULL and self.owner is True:
            llama_cpp.llama_batch_free(self.batch[0])
            self.batch = NULL

    def close(self):
        self.__dealloc__()

    def n_tokens(self) -> int:
        assert self.batch is not NULL
        return self.batch.n_tokens

    def reset(self):
        assert self.batch is not NULL
        self.batch.n_tokens = 0

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
        assert self.batch is not NULL
        n_tokens = len(batch)
        self.batch.n_tokens = n_tokens
        for i in range(n_tokens):
            self.batch.token[i] = batch[i]
            self.batch.pos[i] = n_past + i
            self.batch.seq_id[i][0] = 0
            self.batch.n_seq_id[i] = 1
            self.batch.logits[i] = logits_all
        self.batch.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        assert self.batch is not NULL
        n_tokens = len(batch)
        n_tokens0 = self.batch.n_tokens
        self.batch.n_tokens += n_tokens
        for i in range(n_tokens):
            j = n_tokens0 + i
            self.batch.token[j] = batch[i]
            self.batch.pos[j] = i
            self.batch.seq_id[j][0] = seq_id
            self.batch.n_seq_id[j] = 1
            self.batch.logits[j] = logits_all
        self.batch.logits[n_tokens - 1] = True



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


