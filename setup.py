#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='ConvNADE',
    version='0.0.1',
    author='Marc-Alexandre Côté',
    author_email='marc-alexandre.cote@usherbrooke.ca',
    url='https://github.com/MarcCote/ConvNADE',
    packages=find_packages(),
    license='LICENSE',
    description='Convolutional extension for NADE.',
    long_description=open('README.md').read(),
    install_requires=['smartlearner'],
    dependency_links=['https://github.com/SMART-Lab/smartlearner/archive/master.zip#egg=smartlearner-0.0.1']
)
