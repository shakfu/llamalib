#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


typedef struct _Foo {
    std::array<int, 10> array;
} Foo;

typedef struct _config {
    std::vector<int> nums;
    int n_probs;
    float * probs;
} config;



float square(float x) {
    return x * x;
}


class Matrix {
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data = new float[rows*cols];
    }
    float *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    float *m_data;
};


config config_init(int n_probs) {
    config cfg = {
        {8, 4, 5, 9},
        0, 
        nullptr
    };
    cfg.n_probs = n_probs;
    cfg.probs = (float *) malloc(sizeof(float) * n_probs);
    return cfg;
}


std::vector<int> get_vector(void)
{
    config cfg = config_init(10);
    return cfg.nums;
}


template <typename T>
py::array_t<T> to_array(T * carr, size_t carr_size)
{
    py::array_t<T> arr({static_cast<ssize_t>(carr_size)});
    auto view = arr.mutable_unchecked();
    for(size_t i = 0; i < arr.shape(0); ++i) {
        printf("view(%zu) = %f\n", i, carr[i]);
        view(i) = carr[i];
    }
    return arr;
}

template <typename T>
py::array_t<T> to_array2(T * carr, size_t carr_size)
{
    constexpr size_t elem_size = sizeof(T);
    // size_t carr_size = sizeof(carr) / elem_size;
    size_t shape[1]{carr_size,};
    size_t strides[1]{carr_size * elem_size,};
    auto arr = py::array_t<T>(shape, strides);
    // py::array_t<T, py::array::c_style | py::array::forcecast> arr({static_cast<ssize_t>(carr_size)});
    // py::array_t<T, py::array::c_style | py::array::forcecast> arr({shape});
    // py::array_t<T> arr({shape});
    auto view = arr.mutable_unchecked();
    for(size_t i = 0; i < arr.shape(0); ++i) {
        printf("view(%zu) = %f\n", i, carr[i]);
        view(i) = carr[i];
    }
    return arr;
}

py::array_t<float> get_array(void)
{

    config cfg = config_init(10);
    for (size_t i = 0; i < cfg.n_probs; ++i) {
        printf("cfg.prob[%zu] = %f\n", i, 0.2);
        cfg.probs[i] = 0.2;
    }
    return to_array<float>(cfg.probs, cfg.n_probs);
}

template <typename T> 
py::array_t<T> to_matrix(T ** vals, size_t n_rows, size_t n_cols)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> arr({n_rows, n_cols});
    auto ra = arr.mutable_unchecked();
    for (size_t i = 0; i < n_rows; i++) {
        for (size_t j = 0; j < n_cols; j++) {
            ra(i, j) = vals[i][j];
        };
    };
    return arr;
};


py::array_t<float> get_matrix2(void)
{
    // allocate
    int n_rows = 4;
    int n_cols = 3; 
    float ** arr2d = (float **)malloc(n_rows * sizeof(float *));    
    for (int i = 0; i < n_rows; i++) {
        arr2d[i] = (float *)malloc(n_cols * sizeof(float));
    }
    // populate
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            arr2d[i][j] = 0.5;
        }
    }
    // return as numpy matrix
    return to_matrix<float>(arr2d, n_rows, n_cols);
}

template <typename T> 
py::array_t<T> get_matrix(void)
{
    std::vector<std::vector<T>> vals = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15}
    };

    size_t N = vals.size();
    size_t M = vals[0].size();

    py::array_t<T, py::array::c_style | py::array::forcecast> arr({N, M});

    auto ra = arr.mutable_unchecked();

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            ra(i, j) = vals[i][j];
        };
    };

    return arr;
};



struct Person {
    int age;
    int n_grades;
    float* grades;
};

class WrappedPerson
{

public:
    // + maybe some constructor accepting struct Person
    WrappedPerson() {
        this->p.age = 0;
        this->p.n_grades = 0;
        this->p.grades = nullptr;
        this->owner = true;
    }
    WrappedPerson(int age, std::vector<float>& grades) {
        this->p.age = age;
        this->p.n_grades = grades.size(); 
        if (grades.size() == 0) {
            this->p.grades = nullptr;
        } else {
            this->p.grades = new float[this->p.n_grades];
            std::memcpy(this->p.grades, grades.data(), this->p.n_grades*sizeof(float));
        }
    }
    WrappedPerson(Person p) { this->p = p; }
    ~WrappedPerson() {
        if (this->owner) {
            delete[] this->p.grades;
        };
    }
    int get_age() { return this->p.age; }
    void set_age(int age) { this->p.age = age; }
    int get_n_grades() { return this->p.n_grades; }
    void set_n_grades(int n) { this->p.n_grades = n; }  
    std::vector<float> get_grades() {
        std::vector<float> result(this->p.grades, this->p.grades + this->p.n_grades);
        // std::vector<float> result;
        // result.reserve(this->p.n_grades);
        // for (int i = 0; i < this->p.n_grades; i++) {
        //     result.push_back(this->p.grades[i]);
        // }
        return result;
    }
    void set_grades(const std::vector<float>& grades) {
        this->p.n_grades = grades.size();
        if (grades.size() == 0) {
            this->p.grades = nullptr;
        } else {
            this->p.grades = new float[this->p.n_grades];
            std::memcpy(this->p.grades, grades.data(), this->p.n_grades*sizeof(float));
        }
    }
    void set_grade_by_id(float val, size_t idx) { this->p.grades[idx] = val; }

private:
    Person p;
    bool owner = false;
};


template <class T> class ptr_wrapper
{
    public:
        ptr_wrapper() : ptr(nullptr) {}
        ptr_wrapper(T* ptr) : ptr(ptr) {}
        ptr_wrapper(const ptr_wrapper& other) : ptr(other.ptr) {}
        T& operator* () const { return *ptr; }
        T* operator->() const { return  ptr; }
        T* get() const { return ptr; }
        void destroy() { delete ptr; }
        T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};

py::array wrap_array_ptr(float *v) {
  auto capsule = py::capsule(
      &v, [](void *v) { delete reinterpret_cast<std::vector<float> *>(v); });
  return py::array(static_cast<pybind11::ssize_t>(sizeof(v)), v, capsule);
}



PYBIND11_MODULE(scratch, m) {
    m.doc() = "scratch: pybind11 dummy wrapper"; // optional module docstring
    m.attr("__version__") = "0.0.1";


    PYBIND11_NUMPY_DTYPE(Foo, array);

    // -----------------------------------------------------------------------
    // scratch

    m.def("square", py::vectorize(square)); // vectorized for use w/ np arrays

    py::class_<Foo>(m, "Foo")
        .def(py::init<>())
        .def_readwrite("array", &Foo::array);
    
    m.def("get_vector", &get_vector);

    // m.def("get_matrix", &get_matrix); // doesn't work
    m.def("get_matrix", &get_matrix<float>);
    // m.def("get_matrix", &get_matrix<int>);

    // this also works
    // m.def("to_matrix", (py::array_t<float> (*)(void)) &to_matrix);
    // m.def("to_matrix", (py::array_t<int> (*)(void)) &to_matrix);

    m.def("get_matrix2", &get_matrix2);

    m.def("get_array", &get_array);

    py::class_<Matrix, std::shared_ptr<Matrix>>(m, "Matrix", py::buffer_protocol())
       .def(py::init<size_t, size_t>()) // can be used: np.array(m) or np.array(m, copy = False)
       .def_buffer([](Matrix &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { m.rows(), m.cols() },                 /* Buffer dimensions */
                { sizeof(float) * m.cols(),             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

    py::class_<WrappedPerson>(m, "WrappedPerson")
        .def(py::init<>())
        .def(py::init([](int age, std::vector<float> grades) -> std::unique_ptr<WrappedPerson> {
            return std::unique_ptr<WrappedPerson>(new WrappedPerson(age, grades));
        }))
        .def_property("age", &WrappedPerson::get_age, &WrappedPerson::set_age)
        .def_property_readonly("n_grades", &WrappedPerson::get_n_grades)
        // .def_property("grades", &WrappedPerson::get_grades, nullptr)
        .def_property("grades", &WrappedPerson::get_grades, &WrappedPerson::set_grades)
        .def("set_grade_by_id", &WrappedPerson::set_grade_by_id);

    py::class_<Person, std::shared_ptr<Person>>(m, "Person")
        .def( py::init( [](){ return new Person(); } ))
        .def( py::init( [](int age, std::vector<float> grades) { 
            Person* p = new Person();
            p->age = age;
            p->n_grades = grades.size(); 
            if (grades.size() == 0) {
                p->grades = nullptr;
            } else {
                p->grades = new float[p->n_grades];
                std::memcpy(p->grades, grades.data(), p->n_grades*sizeof(float));
            }
            return p;
        }))
        .def_readwrite("age", &Person::age)
        .def_readonly("n_grades", &Person::n_grades)
        .def_property_readonly("grades", [](Person& self) -> std::vector<float> {
            std::vector<float> result(self.grades, self.grades + self.n_grades);
            return result;
        });
}