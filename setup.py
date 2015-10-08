import os
from setuptools import setup, Extension

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='pyfht',
    version='1.0.0',
    author='Adam Greig',
    author_email='adam@adamgreig.com',
    url='https://github.com/adamgreig/pyfht',
    py_modules=['pyfht'],
    ext_modules=[Extension('libfht', sources=['fht.c'])],
    license='MIT',
    description='Fast Hadamard Transform',
    install_requires=['numpy'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering'
    ],
)
