from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        'pygraph',
        ['src/pygraph.cpp'],
        include_dirs=[
            '/usr/local/Cellar/boost/1.76.0/include',
            '/usr/local/Cellar/eigen/3.4.0_1/include/eigen3',
            '/usr/local/Cellar/graphviz/2.40.1/include'
        ],
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
