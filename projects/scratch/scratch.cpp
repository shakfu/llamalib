#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


std::vector<int> demo(void)
{
    std::vector<int> v = {8, 4, 5, 9};
    return v;
}


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


py::array_t<float> to_matrix()
{
    std::vector<std::vector<float>> vals = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15}
    };

    size_t N = vals.size();
    size_t M = vals[0].size();

    py::array_t<float, py::array::c_style> arr({N, M});

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



PYBIND11_MODULE(scratch, m) {
    m.doc() = "scratch: pybind11 dummy wrapper"; // optional module docstring
    m.attr("__version__") = "0.0.1";

    // -----------------------------------------------------------------------
    // scratch
    
    m.def("demo", (std::vector<int> (*)()) &demo);

    m.def("to_matrix", &to_matrix);

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