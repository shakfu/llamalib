import os
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

PLATFORM = platform.system()

WITH_DYLIB = os.getenv("WITH_DYLIB", False)

INCLUDE_DIRS = [
    "projects/llamalib",
]
LIBRARY_DIRS = []
EXTRA_OBJECTS = []
EXTRA_LINK_ARGS = []
LIBRARIES = ["pthread"]

if WITH_DYLIB:
    LIBRARIES.extend([
        'common',
        'llama',
        'ggml',
    ])
else:
    EXTRA_OBJECTS.extend([
        'lib/libcommon.a', 
        'lib/libllama.a', 
        'lib/libggml.a',
    ])

CWD = os.getcwd()
LIB = os.path.join(CWD, 'lib')
LIBRARY_DIRS.append(LIB)
INCLUDE_DIRS.append(os.path.join(CWD, 'include'))

if PLATFORM == 'Darwin':
    EXTRA_LINK_ARGS.append('-mmacosx-version-min=13.6')
    # add local rpath
    EXTRA_LINK_ARGS.append('-Wl,-rpath,'+LIB)
    os.environ['LDFLAGS'] = ' '.join([
        '-framework Accelerate',
        '-framework Foundation',
        '-framework Metal',
        '-framework MetalKit',
    ])

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
