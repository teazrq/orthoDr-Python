"""Compile the standalone orthoDr kernels against shared native libraries."""
import importlib.util
import os
from pathlib import Path
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
sources = [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/cpp").glob("*.cpp"))]
sources += ["src/bindings.cpp"]
openmp = ["-fopenmp"] if sys.platform.startswith("linux") else []
include_dirs = ["src", "src/cpp"]
if os.environ.get("ORTHODR_ARMADILLO_INCLUDE"):
    include_dirs.append(os.environ["ORTHODR_ARMADILLO_INCLUDE"])


class BuildNative(build_ext):
    def build_extension(self, extension):
        if os.name != "nt":
            return super().build_extension(extension)
        spec = importlib.util.spec_from_file_location("orthodr_build_native", ROOT / "tools/build_native.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.build(Path(self.get_ext_fullpath(extension.name)), Path(self.build_temp), jobs=self.parallel or 3)


extension = Pybind11Extension(
    "orthodr._core", sources=sources, include_dirs=include_dirs, cxx_std=17,
    libraries=["lapack", "blas"] if os.name != "nt" else [],
    extra_compile_args=["-O2", "-DARMA_DONT_USE_WRAPPER", "-DARMA_WARN_LEVEL=1", *openmp] if os.name != "nt" else [],
    extra_link_args=openmp,
)
setup(ext_modules=[extension], cmdclass={"build_ext": BuildNative})
