from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

GMP_INCLUDE = '/usr/local/Cellar/gmp/6.2.1_1/include'
MPFR_INCLUDE = '/usr/local/Cellar/mpfr/4.1.0/include'
BOOST_INCLUDE = '/usr/local/Cellar/boost/1.76.0/include'
EIGEN_INCLUDE = '/usr/local/Cellar/eigen/3.4.0_1/include/eigen3'
GRAPHVIZ_INCLUDE = '/usr/local/Cellar/graphviz/2.40.1/include'

# Infer LIB paths from INCLUDE paths (which should end with /include)
GMP_LIB = GMP_INCLUDE[:-8] + '/lib'
MPFR_LIB = MPFR_INCLUDE[:-8] + '/lib'
GRAPHVIZ_LIB = GRAPHVIZ_INCLUDE[:-8] + '/lib'

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
