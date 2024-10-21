import os
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

PLATFORM = platform.system()

WITH_DYLIB = os.getenv("WITH_DYLIB", False)

LLAMACPP_INCLUDE = "thirdparty/llama.cpp/include"
LLAMACPP_LIBS_DIR = "thirdparty/llama.cpp/lib"

INCLUDE_DIRS = [
    "projects/llamalib",
    LLAMACPP_INCLUDE,
]
LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]
EXTRA_OBJECTS = []
EXTRA_LINK_ARGS = []
LIBRARIES = ["pthread"]

if WITH_DYLIB:
    LIBRARIES.extend([
        'common',
        'ggml',
        'llama',
    ])
else:
    EXTRA_OBJECTS.extend([
        f'{LLAMACPP_LIBS_DIR}/libcommon.a', 
        f'{LLAMACPP_LIBS_DIR}/libllama.a', 
        f'{LLAMACPP_LIBS_DIR}/libggml.a',
    ])

CWD = os.getcwd()
INCLUDE_DIRS.append(os.path.join(CWD, 'include'))

if PLATFORM == 'Darwin':
    EXTRA_LINK_ARGS.append('-mmacosx-version-min=13.6')
    # add local rpath
    EXTRA_LINK_ARGS.append('-Wl,-rpath,'+LLAMACPP_LIBS_DIR)
    os.environ['LDFLAGS'] = ' '.join([
        '-framework Accelerate',
        '-framework Foundation',
        '-framework Metal',
        '-framework MetalKit',
    ])

if PLATFORM == 'Linux':
    EXTRA_LINK_ARGS.append('-fopenmp')

extensions = [
    Extension("cyllama", 
        [
            "projects/cyllama/cyllama.pyx", 
        ],
        # define_macros = [
        #     ("INTERP_DSP", 1),
        #     ("__MACOSX_CORE__", None)
        # ],
        include_dirs = INCLUDE_DIRS,
        libraries = LIBRARIES,
        library_dirs = LIBRARY_DIRS,
        extra_objects = EXTRA_OBJECTS,
        extra_compile_args = ['-std=c++14'],
        extra_link_args = EXTRA_LINK_ARGS,
    ),
]

setup(
    name='cyllama',
    ext_modules=cythonize(
        extensions,
        language_level="3str",
    ),
)
