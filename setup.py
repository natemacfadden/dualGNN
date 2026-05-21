from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "dualgnn.reference_samplers._grow2d.grow2d",
        sources       = ["src/dualgnn/reference_samplers/_grow2d/grow2d.pyx"],
        include_dirs  = ["src/dualgnn/reference_samplers/_grow2d"],
        define_macros = [("GROW2D_IMPLEMENTATION", None)],
        extra_compile_args = ["-O3"],
        language      = "c",
    ),
    Extension(
        "dualgnn.reference_samplers._pushing.pushing",
        sources       = ["src/dualgnn/reference_samplers/_pushing/pushing.pyx"],
        include_dirs  = ["src/dualgnn/reference_samplers/_pushing"],
        define_macros = [("PUSHING_IMPLEMENTATION", None)],
        extra_compile_args = ["-O3"],
        language      = "c",
    ),
]

setup(
    ext_modules = cythonize(extensions, language_level=3),
)
