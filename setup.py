#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Cython.Build import cythonize

from setuptools import setup, Extension

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'Cython>=0.22',
    'numpy>=1.10',
    'scipy>=0.9'
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='dataprovider',
    version='0.1.6',
    description="Deep learning platform-independent volumetric data provider for training 3D convolutional network.",
    long_description=readme + '\n\n' + history,
    author="Seung Lab",
    author_email='tartavull@gmail.com',
    url='https://github.com/tartavull/dataprovider',
    packages=[
        'dataprovider',
    ],
    package_dir={'dataprovider':
                 'dataprovider'},
    entry_points={
        'console_scripts': [
            'dataprovider=dataprovider.cli:main'
        ]
    },
    include_package_data=True,
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.xml', '*.html', '*.js','*.c','*.pyx']
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='dataprovider',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    ext_modules=cythonize([Extension('*', ['dataprovider/warping/*.pyx'])])
)
