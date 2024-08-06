# Opaque Pointers in Pybind11


- [Dealing with Opaque Pointers in Pybind11](https://stackoverflow.com/questions/50641461/dealing-with-opaque-pointers-in-pybind11)

```cpp
PYBIND_MODULE(mymodule, m) {
  py::class_<mystruct>(m, "mystruct");
  m.def("f1", &myfunction1);
  m.def("f2", &myfunction2);
}
```

> If you wish to avoid conflict with other pybind11 modules that might declare types on this third-party type, consider using `py::module_local()` refered to in the docs [here](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings) 


see also:

- [PyBind11 : share opaque pointers between independently-built C++ modules through Python](https://stackoverflow.com/questions/76272413/pybind11-share-opaque-pointers-between-independently-built-c-modules-through)

- [Mixing type conversions and opaque types with pybind11](https://stackoverflow.com/questions/58169847/mixing-type-conversions-and-opaque-types-with-pybind11) also [issue-1940](https://github.com/pybind/pybind11/issues/1940)

- [Does there exists alternative for opaque pointer compared with boost python?](https://github.com/pybind/pybind11/issues/1778)

-  