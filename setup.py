from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

BOOST_INCLUDE = '/usr/local/Cellar/boost/1.76.0/include'
EIGEN_INCLUDE = '/usr/local/Cellar/eigen/3.4.0_1/include/eigen3'
GRAPHVIZ_INCLUDE = '/usr/local/Cellar/graphviz/2.40.1/include'

# Infer GRAPHVIZ_LIB from GRAPHVIZ_INCLUDE (which should end with /include)
GRAPHVIZ_LIB = GRAPHVIZ_INCLUDE[:-8] + '/lib'

ext_modules = [
    Pybind11Extension(
        'pygraph',
        ['src/pygraph.cpp'],
        include_dirs=[BOOST_INCLUDE, EIGEN_INCLUDE, GRAPHVIZ_INCLUDE],
        library_dirs=[GRAPHVIZ_LIB],
        extra_compile_args=[
            '--std=c++11'
        ],
        extra_link_args=[
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
