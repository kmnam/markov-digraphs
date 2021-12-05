from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'pygraph', ['pygraph.pyx'], 
        include_dirs=[
            '/usr/local/Cellar/eigen/3.4.0_1/include/eigen3',
            '/usr/local/Cellar/boost/1.76.0/include'
        ],
        language='c++',
        libraries=['mpfr'],
        extra_compile_args=['--std=c++11']
    )
]

setup(ext_modules=cythonize(extensions, language_level='3'))
