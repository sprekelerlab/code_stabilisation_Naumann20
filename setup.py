# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:59:51 2016

@author: laura
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('rate_network_cython', ['rate_network_cython.pyx'],)]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
  name='Rate network with pre inh',
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include()],
  cmdclass={'build_ext': build_ext},
)