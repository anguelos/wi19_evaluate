#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(
    name='wi19evaluate',
    version='0.0.1dev',
    packages=['wi19'],
    scripts=['bin/wi19evaluate','bin/wi19leaderboard'],
    license='GNU Lesser General Public License v3.0',
    long_description=open('README.md').read(),
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/wi19_evaluate',
    package_data={'scenethecizer': ["data/backgrounds/paper_texture.jpg","data/corpora/01_the_ugly_duckling.txt"]},
    install_requires=[
        'numpy','matplotlib','tqdm'
    ],
)