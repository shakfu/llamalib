import os
import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

WITH_DYLIB = os.getenv("WITH_DYLIB", False)

INCLUDE_DIRS = []
LIBRARY_DIRS = []
EXTRA_OBJECTS = []
EXTRA_LINK_ARGS = ['-mmacosx-version-min=13.6']
LIBRARIES = ["pthread"]

if WITH_DYLIB:
    LIBRARIES.extend(['llama', 'ggml'])
else:
    EXTRA_OBJECTS.extend(['lib/libfaust.a', 'lib/libggml.a'])

CWD = os.getcwd()
LIB = os.path.join(CWD, 'lib')
LIBRARY_DIRS.append(LIB)
INCLUDE_DIRS.append(os.path.join(CWD, 'include'))

# add local rpath
if platform.system() == 'Darwin':
    EXTRA_LINK_ARGS.append('-Wl,-rpath,'+LIB)

    "$<$<PLATFORM_ID:Darwin>:-framework Accelerate>"
    "$<$<PLATFORM_ID:Darwin>:-framework Foundation>"
    # "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
    "$<$<PLATFORM_ID:Darwin>:-framework Metal>"
    "$<$<PLATFORM_ID:Darwin>:-framework MetalKit>"

os.environ['LDFLAGS'] = ' '.join([
    '-framework Acceleraten',
    '-framework Foundation',
    # '-framework CoreFoundation',
    '-framework Metal',
    '-framework MetalKit',
])

extensions = [
    Extension("cyfaust", 
        [
            "projects/cyfaust/cyfaust.pyx", 
            "include/rtaudio/RtAudio.cpp",
            "include/rtaudio/rtaudio_c.cpp",
        ],
        define_macros = [
            ("INTERP_DSP", 1),
            ("__MACOSX_CORE__", None)
        ],
        include_dirs = INCLUDE_DIRS,
        libraries = LIBRARIES,
        library_dirs = LIBRARY_DIRS,
        extra_objects = EXTRA_OBJECTS,
        extra_compile_args = ['-std=c++11'],
        extra_link_args = EXTRA_LINK_ARGS,
    ),
]

setup(
    name='cyfaust',
    ext_modules=cythonize(
        extensions,
        language_level="3str",
    ),
)
