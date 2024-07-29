set(PROJECT_NAME pbllama)

set(PROJECT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pbllama.cpp)

# Detect the installed pybind11 package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${PB_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    ${PROJECT_NAME}
    ${PROJECT_SRC}
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_INCLUDES}
)

# target_compile_definitions(
#     ${PROJECT_NAME}
#     PUBLIC
# )

target_link_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_LIB_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    ${LLAMA_STATICLIB}
    ${GGML_STATICLIB}
    ${LLAMA_LINK_LIBS}
)