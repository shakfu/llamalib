# distutils: language = c++

cimport llama

cdef class ModelParams:
    cdef llama.llama_model_params p

    def __init__(self):
        self.p = llama.llama_model_default_params()

    @property
    def n_gpu_layers(self) -> int:
       return self.p.n_gpu_layers

    @property
    def split_mode(self) -> int:
        return self.p.split_mode

    @property
    def main_gpu(self) -> int:
        return self.p.main_gpu

    @property
    def vocab_only(self) -> bool:
        return self.p.vocab_only

    @property
    def use_mmap(self) -> bool:
        return self.p.use_mmap

    @property
    def use_mlock(self) -> bool:
        return self.p.use_mlock

    @property
    def check_tensors(self) -> bool:
        return self.p.check_tensors

