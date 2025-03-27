from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(name='tssb',
      version='0.1',
      url='https://github.com/ermshaua/time-series-segmentation-benchmark',
      license='BSD 3-Clause License',
      author='Arik Ermshaus',
      description='This repository contains a time series segmentation benchmark.',
      packages=find_packages(exclude=['tests', 'examples']),
      package_data={'': ['LICENSE']},
      include_package_data=True,
      install_requires=requirements,
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      zip_safe=False)