import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# INCLUDE paths should be specified by the user prior to building
GMP_INCLUDE = '/opt/homebrew/Cellar/gmp/6.3.0/include'
MPFR_INCLUDE = '/opt/homebrew/Cellar/mpfr/4.2.1/include'
BOOST_INCLUDE = '/opt/homebrew/Cellar/boost/1.84.0_1/include'
EIGEN_INCLUDE = '/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3'
GRAPHVIZ_INCLUDE = '/opt/homebrew/Cellar/graphviz/12.0.0/include'

# Infer LIB paths from INCLUDE paths (which should end with /include)
GMP_LIB = os.path.join(os.path.dirname(GMP_INCLUDE), 'lib')
MPFR_LIB = os.path.join(os.path.dirname(MPFR_INCLUDE), 'lib')
GRAPHVIZ_LIB = os.path.join(os.path.dirname(GRAPHVIZ_INCLUDE), 'lib')

ext_modules = [
    Pybind11Extension(
        'pygraph',
        ['src/pygraph.cpp'],
        include_dirs=[
            GMP_INCLUDE, MPFR_INCLUDE, BOOST_INCLUDE, EIGEN_INCLUDE,
            GRAPHVIZ_INCLUDE
        ],
        library_dirs=[GMP_LIB, MPFR_LIB, GRAPHVIZ_LIB],
        extra_compile_args=[
            '--std=c++14'
        ],
        extra_link_args=[
            '-lgmp',
            '-lmpfr',
            '-lgvc',
            '-lcgraph'
        ]
    )
]

setup(
    name='pygraph',
    version='0.9b',
    ext_modules=ext_modules
)
