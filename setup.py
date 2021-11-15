from setuptools import setup, find_packages

import numpy as np

setup(name='tssb',
      version='0.1',
      url='https://github.com/ermshaua/time-series-segmentation-benchmark',
      license='BSD 3-Clause License',
      author='Arik Ermshaus',
      description='This repository contains a time series segmentation benchmark.',
      packages=find_packages(exclude=['tests', 'examples']),
      install_requires=np.loadtxt(fname='requirements.txt', delimiter='\n', dtype=np.str).tolist(),
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      zip_safe=False)