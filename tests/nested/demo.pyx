# distutils: language = c++

cdef extern from *:
    """
    typedef struct _cpu_params {
        int n_threads;
    } cpu_params;

    typedef struct _app_params {
        int param1;
        cpu_params cpuparams;
    } app_params;

    float add(float x, float y) {
        return x + y;
    }

    int add(int x, int y) {
        return x + y;
    }

    """

    ctypedef struct cpu_params:
        int n_threads

    ctypedef struct app_params:
        int param1
        cpu_params cpuparams

    cdef float add(float x, float y)
    cdef int add(int x, int y)


# --------------------------------------------------
# simplest flattened solution

cdef class AppParams1:
    cdef app_params params

    @staticmethod
    cdef AppParams1 from_instance(app_params p):
        cdef AppParams1 wrapper = AppParams1.__new__(AppParams1)
        wrapper.params = p
        return wrapper

    def __cinit__(self):
        cdef app_params _ap
        cdef cpu_params _cp
        _cp.n_threads = 4
        _ap.param1 = 20
        _ap.cpuparams = _cp
        self.params = _ap

    @property
    def param1(self) -> int:
        """param1 desc."""
        return self.params.param1

    @param1.setter
    def param1(self, value: int):
        self.params.param1 = value

    @property
    def cpu_n_threads(self) -> int:
        """number of threads."""
        return self.params.cpuparams.n_threads

    @cpu_n_threads.setter
    def cpu_n_threads(self, value: int):
        self.params.cpuparams.n_threads = value


# --------------------------------------------------
# alternative nested solution

cdef class CpuParams:
    cdef cpu_params *ptr

    @staticmethod
    cdef CpuParams from_ptr(cpu_params *p):
        cdef CpuParams wrapper = CpuParams.__new__(CpuParams)
        wrapper.ptr = p
        return wrapper

    @property
    def n_threads(self) -> int:
        """number of threads."""
        return self.ptr.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.ptr.n_threads = value

cdef class AppParams:
    cdef app_params params
    cdef public CpuParams cpuparams

    @staticmethod
    cdef AppParams from_instance(app_params p):
        cdef AppParams wrapper = AppParams.__new__(AppParams)
        wrapper.params = p
        wrapper.cpuparams = CpuParams.from_ptr(&wrapper.params.cpuparams)
        return wrapper

    def __cinit__(self):
        self.cpuparams = CpuParams.from_ptr(&self.params.cpuparams)

    @property
    def param1(self) -> int:
        """param1 desc."""
        return self.params.param1

    @param1.setter
    def param1(self, value: int):
        self.params.param1 = value

    def get_n_threads(self): # test that it works
        return self.params.cpuparams.n_threads


def plus_float(float x, float y) -> float:
    return add(<float>x, <float>y)

def plus_int(int x, int y) -> int:
    return add(<int>x, <int>y)


def factory():
    cdef app_params _ap
    cdef cpu_params _cp
    _cp.n_threads = 6
    _ap.param1 = 15
    _ap.cpuparams = _cp
    cdef AppParams p = AppParams.from_instance(_ap)
    return p


