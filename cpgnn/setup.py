from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

package = [Extension('random_choice', ["util/cython/random_choice.pyx"], extra_compile_args=["-std=c++11"])]

setup(ext_modules=cythonize(package, language="c++"))
