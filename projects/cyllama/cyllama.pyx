# distutils: language = c++

from libc.stdlib cimport malloc, free

cimport llama

import os
from typing import Optional


cdef class ModelParams:
    cdef llama.llama_model_params p

    def __init__(self):
        self.p = llama.llama_model_default_params()

    @property
    def n_gpu_layers(self) -> int:
       return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def split_mode(self) -> int:
        return self.p.split_mode

    @split_mode.setter
    def split_mode(self, value: int):
        self.p.split_mode = value

    @property
    def main_gpu(self) -> int:
        return self.p.main_gpu

    @main_gpu.setter
    def main_gpu(self, value: int):
        self.p.main_gpu = value

    @property
    def vocab_only(self) -> bool:
        return self.p.vocab_only

    @vocab_only.setter
    def vocab_only(self, value: bool):
        self.p.vocab_only = value

    @property
    def use_mmap(self) -> bool:
        return self.p.use_mmap

    @use_mmap.setter
    def use_mmap(self, value: bool):
        self.p.use_mmap = value

    @property
    def use_mlock(self) -> bool:
        return self.p.use_mlock

    @use_mlock.setter
    def use_mlock(self, value: bool):
        self.p.use_mlock = value

    @property
    def check_tensors(self) -> bool:
        return self.p.check_tensors

    @check_tensors.setter
    def check_tensors(self, value: bool):
        self.p.check_tensors = value


cdef class LlamaModel:
    """cython wrapper for llama.cpp llama_model."""
    cdef llama.llama_model * model
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
        self.model = llama.llama_load_model_from_file(
            self.path_model.encode("utf-8"), 
            self.params.p
        )

        if self.model is NULL:
            raise ValueError(f"Failed to load model from file: {path_model}")

    def __dealloc__(self):
        if self.model is not NULL and self.owner is True:
            llama.llama_free_model(self.model)
            self.model = NULL

    # FIXME: name collision
    # def vocab_type(self) -> llama.llama_vocab_type:
    #     # assert self.model is not None
    #     return llama.get_llama_vocab_type(self.model)

    def n_vocab(self) -> int:
        return llama.llama_n_vocab(self.model)

    def n_ctx_train(self) -> int:
        return llama.llama_n_ctx_train(self.model)

    def n_embd(self) -> int:
        return llama.llama_n_embd(self.model)

    def rope_freq_scale_train(self) -> float:
        return llama.llama_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        cdef char buf[1024]
        llama.llama_model_desc(self.model, buf, 1024)
        return buf.decode("utf-8")

    def size(self) -> int:
        return llama.llama_model_size(self.model)

    def n_params(self) -> int:
        return llama.llama_model_n_params(self.model)

    # def get_tensor(self, name: str) -> ctypes.c_void_p:
    #     return llama_cpp.llama_get_model_tensor(self.model, name.encode("utf-8"))

#     def apply_lora_from_file(
#         self,
#         lora_path: str,
#         scale: float,
#         path_base_model: Optional[str],
#         n_threads: int,
#     ):
#         assert self.model is not None
#         return llama_cpp.llama_model_apply_lora_from_file(
#             self.model,
#             lora_path.encode("utf-8"),
#             scale,
#             (
#                 path_base_model.encode("utf-8")
#                 if path_base_model is not None
#                 else ctypes.c_char_p(0)
#             ),
#             n_threads,
#         )

    # Vocab

    def token_get_text(self, token: int) -> str:
        return llama.llama_token_get_text(self.model, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        return llama.llama_token_get_score(self.model, token)

    def token_get_attr(self, token: int) -> int:
        return llama.llama_token_get_attr(self.model, token)

    # Special tokens

    def token_bos(self) -> int:
        return llama.llama_token_bos(self.model)

    def token_eos(self) -> int:
        return llama.llama_token_eos(self.model)

    def token_cls(self) -> int:
        return llama.llama_token_cls(self.model)

    def token_sep(self) -> int:
        return llama.llama_token_sep(self.model)

    def token_nl(self) -> int:
        return llama.llama_token_nl(self.model)

    def token_prefix(self) -> int:
        return llama.llama_token_prefix(self.model)

    def token_middle(self) -> int:
        return llama.llama_token_middle(self.model)

    def token_suffix(self) -> int:
        return llama.llama_token_suffix(self.model)

    def token_eot(self) -> int:
        return llama.llama_token_eot(self.model)

    def add_bos_token(self) -> bool:
        return llama.llama_add_bos_token(self.model)

    def add_eos_token(self) -> bool:
        return llama.llama_add_eos_token(self.model)

    # Tokenization

#     def tokenize(self, text: bytes, add_bos: bool, special: bool):
#         n_ctx = self.n_ctx_train()
#         tokens = (llama_cpp.llama_token * n_ctx)()
#         n_tokens = llama_cpp.llama_tokenize(
#             self.model, text, len(text), tokens, n_ctx, add_bos, special
#         )
#         if n_tokens < 0:
#             n_tokens = abs(n_tokens)
#             tokens = (llama_cpp.llama_token * n_tokens)()
#             n_tokens = llama_cpp.llama_tokenize(
#                 self.model, text, len(text), tokens, n_tokens, add_bos, special
#             )
#             if n_tokens < 0:
#                 raise RuntimeError(
#                     f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
#                 )
#         return list(tokens[:n_tokens])

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        cdef char buf[32]
        llama.llama_token_to_piece(self.model, token, buf, 32, 0, special)
        return buf.decode()
        # return bytes(buf)

#     def detokenize(self, tokens: List[int], special: bool = False) -> bytes:
#         assert self.model is not None
#         output = b""
#         size = 32
#         buffer = (ctypes.c_char * size)()
#         for token in tokens:
#             n = llama_cpp.llama_token_to_piece(
#                 self.model, llama_cpp.llama_token(token), buffer, size, 0, special
#             )
#             assert n <= size
#             output += bytes(buffer[:n])
#         # NOTE: Llama1 models automatically added a space at the start of the prompt
#         # this line removes a leading space if the first token is a beginning of sentence token
#         return (
#             output[1:]
#             if len(tokens) > 0 and tokens[0] == self.token_bos() and output[0:1] == b" "
#             else output
#         )

    # Extra
#     def metadata(self) -> Dict[str, str]:
#         assert self.model is not None
#         metadata: Dict[str, str] = {}
#         buffer_size = 1024
#         buffer = ctypes.create_string_buffer(buffer_size)
#         # zero the buffer
#         buffer.value = b"\0" * buffer_size
#         # iterate over model keys
#         for i in range(llama_cpp.llama_model_meta_count(self.model)):
#             nbytes = llama_cpp.llama_model_meta_key_by_index(
#                 self.model, i, buffer, buffer_size
#             )
#             if nbytes > buffer_size:
#                 buffer_size = nbytes + 1
#                 buffer = ctypes.create_string_buffer(buffer_size)
#                 nbytes = llama_cpp.llama_model_meta_key_by_index(
#                     self.model, i, buffer, buffer_size
#                 )
#             key = buffer.value.decode("utf-8")
#             nbytes = llama_cpp.llama_model_meta_val_str_by_index(
#                 self.model, i, buffer, buffer_size
#             )
#             if nbytes > buffer_size:
#                 buffer_size = nbytes + 1
#                 buffer = ctypes.create_string_buffer(buffer_size)
#                 nbytes = llama_cpp.llama_model_meta_val_str_by_index(
#                     self.model, i, buffer, buffer_size
#                 )
#             value = buffer.value.decode("utf-8")
#             metadata[key] = value
#         return metadata

    @staticmethod
    def default_params() -> ModelParams:
        """Get the default llama_model_params."""
        # return llama_cpp.llama_model_default_params()
        return ModelParams()

