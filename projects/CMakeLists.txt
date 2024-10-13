find_package(Python 3.8 COMPONENTS Interpreter Development Development.Module REQUIRED)


if(DEBUG)
message(STATUS "Python_FOUND: ${Python_FOUND}")
message(STATUS "Python_Development_FOUND: ${Python_Development_FOUND}")
message(STATUS "Python_Development.Embed_FOUND: ${Python_Development.Embed_FOUND}")
message(STATUS "Python_Development.SABIModule_FOUND: ${Python_Development.SABIModule_FOUND}")
message(STATUS "Python_Interpreter_FOUND: ${Python_Interpreter_FOUND}")
message(STATUS "Python_INTERPRETER_ID: ${Python_INTERPRETER_ID}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")
message(STATUS "Python_LINK_OPTIONS: ${Python_LINK_OPTIONS}")
message(STATUS "Python_NumPy_FOUND: ${Python_NumPy_FOUND}")
message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "Python_NumPy_VERSION: ${Python_NumPy_VERSION}")
message(STATUS "Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message(STATUS "Python_RUNTIME_SABI_LIBRARY_DIRS: ${Python_RUNTIME_SABI_LIBRARY_DIRS}")
message(STATUS "Python_SABI_LIBRARIES: ${Python_SABI_LIBRARIES}")
message(STATUS "Python_SABI_LIBRARY_DIRS: ${Python_SABI_LIBRARY_DIRS}")
message(STATUS "Python_SITEARCH: ${Python_SITEARCH}")
message(STATUS "Python_SITELIB: ${Python_SITELIB}")
message(STATUS "Python_SOABI: ${Python_SOABI}")
message(STATUS "Python_SOSABI: ${Python_SOSABI}")
message(STATUS "Python_STDARCH: ${Python_STDARCH}")
message(STATUS "Python_STDLIB: ${Python_STDLIB}")
message(STATUS "Python_VERSION: ${Python_VERSION}")
message(STATUS "Python_VERSION_MAJOR: ${Python_VERSION_MAJOR}")
message(STATUS "Python_VERSION_MINOR: ${Python_VERSION_MINOR}")
message(STATUS "Python_VERSION_PATCH: ${Python_VERSION_PATCH}")
endif()

# common variables


set(LLAMACPP_DIR ${CMAKE_SOURCE_DIR}/thirdparty/llama.cpp)
set(PROJECT_LIB_DIRS ${LLAMACPP_DIR}/lib)
set(PROJECT_INCLUDES
	${LLAMACPP_DIR}/include
	${CMAKE_SOURCE_DIR}/projects/llamalib
)

set(LLAMA_STATICLIB
	${PROJECT_LIB_DIRS}/libllama${CMAKE_STATIC_LIBRARY_SUFFIX})

set(GGML_STATICLIB
	${PROJECT_LIB_DIRS}/libggml${CMAKE_STATIC_LIBRARY_SUFFIX})

set(COMMON_STATICLIB
	${PROJECT_LIB_DIRS}/libcommon${CMAKE_STATIC_LIBRARY_SUFFIX})

set(LLAMA_LINK_LIBS
    $<IF:$<BOOL:${LLAMA_SHAREDLIB}>,-lllama,${LLAMA_STATICLIB}>
    $<IF:$<BOOL:${LLAMA_SHAREDLIB}>,-lggml,${GGML_STATICLIB}>
    $<IF:$<BOOL:${LLAMA_SHAREDLIB}>,-lcommon,${COMMON_STATICLIB}>
    "$<$<PLATFORM_ID:Darwin>:-framework Accelerate>"
    "$<$<PLATFORM_ID:Darwin>:-framework Foundation>"
    "$<$<PLATFORM_ID:Darwin>:-framework Metal>"
    "$<$<PLATFORM_ID:Darwin>:-framework MetalKit>"
)

if(Python_FOUND)
	add_subdirectory(scratch)
	add_subdirectory(cyllama)
    add_subdirectory(nbllama)
    add_subdirectory(pbllama)
endif()
