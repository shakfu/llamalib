# distutils: language = c++

from libc.stdlib cimport malloc, free

cimport llama

import os
from typing import Optional




cdef class ModelParams:
    cdef llama.llama_model_params * _ptr
    cdef bint ptr_owner

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            free(self._ptr)
            self._ptr = NULL

    # def __init__(self):
    #     self.ptr[0] = llama.llama_model_default_params()

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef ModelParams from_instance(llama.llama_model_params p):
        cdef ModelParams params = ModelParams.__new__(ModelParams)
        params._ptr[0] = p
        return params

    @staticmethod
    cdef ModelParams from_ptr(llama.llama_model_params *_ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ModelParams wrapper = ModelParams.__new__(ModelParams)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper


    @staticmethod
    cdef ModelParams create():
        cdef llama.llama_model_params *_ptr = <llama.llama_model_params *>malloc(sizeof(llama.llama_model_params))

        if _ptr is NULL:
            raise MemoryError
        _ptr[0] = llama.llama_model_default_params()
        # _ptr.a = 0
        # _ptr.b = 0
        return ModelParams.from_ptr(_ptr, owner=True)

    @property
    def n_gpu_layers(self) -> int:
       return self._ptr.n_gpu_layers

    @property
    def split_mode(self) -> int:
        return self._ptr.split_mode

    @property
    def main_gpu(self) -> int:
        return self._ptr.main_gpu

    @property
    def vocab_only(self) -> bool:
        return self._ptr.vocab_only

    @property
    def use_mmap(self) -> bool:
        return self._ptr.use_mmap

    @property
    def use_mlock(self) -> bool:
        return self._ptr.use_mlock

    @property
    def check_tensors(self) -> bool:
        return self._ptr.check_tensors


def get_params():
    return ModelParams.create()



