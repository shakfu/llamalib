


# scratch
# -----------------------------------------------------------------------------
#

def process_buffer(float[:,:] input_view not None,
                   float[:,:] output_view=None):

    if output_view is None:
        # Creating a default view, e.g.
        output_view = np.empty_like(input_view)

    # process 'input_view' into 'output_view'
    return output_view

# np.array(cy.process_buffer(np.arange(100, dtype=np.float32).reshape(2,-1)))

ctypedef struct my_struct_t:
    int idx
    float x
    float y

cdef class CMyClass:
    cdef my_struct_t c_buffer[1000]

    def __init__(self):
        for i in range(100):
            self.c_buffer[i].idx = i
            self.c_buffer[i].x = i*1.1
            self.c_buffer[i].y = i*2.1

    def get_array(self):
        return <my_struct_t[:1000]>&self.c_buffer[0]

    def get_memoryview(self):
        return memoryview(<my_struct_t[:1000]>&self.c_buffer[0])

# OK
# arr1 = np.array(a.get_memoryview(), dtype=np.dtype([('idx', '<i4'), ('x', '<f4'), ('y', '<f4')]))
# arr1.view(np.recarray)
# np.recarray((100,), dtype=np.dtype([('idx', '<i4'), ('x', '<f4'), ('y', '<f4')]), buf=a.get_memoryview())

# not OK
# narr = np.recarray(30, dtype=np.dtype([("idx", np.intc), ("x", np.float32), ("p", np.float32)]))

