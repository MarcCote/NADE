#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='ConvNADE',
    version='1.0.0',
    author='Marc-Alexandre Côté',
    author_email='marc-alexandre.cote@usherbrooke.ca',
    url='https://github.com/MarcCote/NADE',
    packages=find_packages(),
    license='LICENSE',
    description='Neural Autoregressive Distribution Estimation.',
    long_description=open('README.md').read(),
    install_requires=['theano', 'smartlearner'],
    dependency_links=['https://github.com/SMART-Lab/smartlearner/archive/master.zip#egg=smartlearner-0.0.1']
)
