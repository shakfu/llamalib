# distutils: language = c++

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

cimport llama_cpp

import os
from typing import Optional

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
        return llama_cpp.llama_n_vocab(self.model)

    def n_ctx_train(self) -> int:
        return llama_cpp.llama_n_ctx_train(self.model)

    def n_embd(self) -> int:
        return llama_cpp.llama_n_embd(self.model)

    def rope_freq_scale_train(self) -> float:
        return llama_cpp.llama_rope_freq_scale_train(self.model)

    def desc(self) -> str:
        cdef char buf[1024]
        llama_cpp.llama_model_desc(self.model, buf, 1024)
        return buf.decode("utf-8")

    def size(self) -> int:
        return llama_cpp.llama_model_size(self.model)

    def n_params(self) -> int:
        return llama_cpp.llama_model_n_params(self.model)

    def get_tensor(self, name: str) -> GGMLTensor:
        cdef llama_cpp.ggml_tensor * tensor = llama_cpp.llama_get_model_tensor(self.model, name.encode("utf-8"))
        return GGMLTensor.from_ptr(tensor)

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
        return llama_cpp.llama_token_get_text(self.model, token).decode("utf-8")

    def token_get_score(self, token: int) -> float:
        return llama_cpp.llama_token_get_score(self.model, token)

    def token_get_attr(self, token: int) -> int:
        return llama_cpp.llama_token_get_attr(self.model, token)

    # Special tokens

    def token_bos(self) -> int:
        return llama_cpp.llama_token_bos(self.model)

    def token_eos(self) -> int:
        return llama_cpp.llama_token_eos(self.model)

    def token_cls(self) -> int:
        return llama_cpp.llama_token_cls(self.model)

    def token_sep(self) -> int:
        return llama_cpp.llama_token_sep(self.model)

    def token_nl(self) -> int:
        return llama_cpp.llama_token_nl(self.model)

    def token_prefix(self) -> int:
        return llama_cpp.llama_token_prefix(self.model)

    def token_middle(self) -> int:
        return llama_cpp.llama_token_middle(self.model)

    def token_suffix(self) -> int:
        return llama_cpp.llama_token_suffix(self.model)

    def token_eot(self) -> int:
        return llama_cpp.llama_token_eot(self.model)

    def add_bos_token(self) -> bool:
        return llama_cpp.llama_add_bos_token(self.model)

    def add_eos_token(self) -> bool:
        return llama_cpp.llama_add_eos_token(self.model)

    # Tokenization

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
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
                raise RuntimeError(
                    f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
                )

        return tokens[:n_tokens]

    def token_to_piece(self, token: int, special: bool = False) -> bytes:
        cdef char buf[32]
        llama_cpp.llama_token_to_piece(self.model, token, buf, 32, 0, special)
        return buf.decode()
        # return bytes(buf)

    def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
        # assert self.model is not None
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

