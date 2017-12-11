#!/usr/bin/env python3
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(name='imdb sa',
      version='0.1.0',
      description='Sentimental Analysis on IMDb movie reviews using CNN/RNN',
      long_description=readme,
      author='Jiho Noh',
      author_email='jiho.noh@uky.edu',
      license=license,
      packages=find_packages(exclude=('data', 'scripts', 'var')),
      install_requires=reqs.strip().split('\n'),
      zip_safe=False)
