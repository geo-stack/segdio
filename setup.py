from setuptools import setup
from segdio import __version__, __project_url__

"""Installation script """

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
      )
