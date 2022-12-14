# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright © 2016 Mattias Südow
# https://github.com/drsudow
#
# Copyright © Segdio Project Contributors
# https://github.com/geo-stack/segdio/graphs/contributors
#
# This code is licensed under the terms of the MIT License as published
# by the Open Source Initiative. For more details, see the MIT license at
# https://opensource.org/licenses/MIT/.
# -----------------------------------------------------------------------------

from setuptools import setup
from segdio import __version__, __project_url__
import csv

"""Installation script """

with open('requirements.txt', 'r') as csvfile:
    INSTALL_REQUIRES = list(csv.reader(csvfile))
INSTALL_REQUIRES = [item for sublist in INSTALL_REQUIRES for item in sublist]

setup(name='segdio',
      version=__version__,
      description='A Python3 reader for SEG-D binary data.',
      url=__project_url__,
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          "Operating System :: OS Independent",
          'Programming Language :: Python :: 3',
          ],
      packages=['segdio'],
      install_requires=INSTALL_REQUIRES,
      )
