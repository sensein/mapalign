"""
Installation script for mapalign
"""

from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()
setup(
    name='mapalign',

    version='0.1.0',
    description='Mapalign',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/satra/mapalign',

    # Author details
    author='satra',

    # Choose your license
    license='Apache',
    
    packages=['mapalign'],
    py_modules=["align", "dist", "embed"],
    
    install_requires=['numpy', 'scipy']
)
