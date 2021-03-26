#! /usr/bin/env python
"""
setup.py for kinisi

@author: Andrew R. McCluskey (andrew.mccluskey@ess.eu)
"""

# System imports
import io
from os import path
from setuptools import setup, find_packages
from kinisi import __version__

PACKAGES = find_packages(exclude=['tests*'])

# versioning

ISRELEASED = False
VERSION = __version__


THIS_DIRECTORY = path.abspath(path.dirname(__file__))
with io.open(path.join(THIS_DIRECTORY, 'README.md')) as f:
    LONG_DESCRIPTION = f.read()

INFO = {
        'name': 'kinisi',
        'description': 'Uncertainty analysis and model comparison '
                       'for atomistic molecular dynamics.',
        'author': 'Andrew R. McCluskey and Benjamin J. Morgan',
        'author_email': 'andrew.mccluskey@ess.eu, b.j.morgan@bath.ac.uk',
        'packages': PACKAGES,
        'include_package_data': True,
        'setup_requires': ['numpy', 'pandas', 'statsmodels', 'scikit-learn',
                           'scipy', 'tqdm', 'uravu>=1.2.4'],
        'install_requires': ['numpy', 'pandas', 'statsmodels', 'scikit-learn',
                             'scipy', 'tqdm', 'uravu>=1.2.4'],
        'version': VERSION,
        'license': 'MIT',
        'long_description': LONG_DESCRIPTION,
        'long_description_content_type': 'text/markdown',
        'classifiers': ['Development Status :: 4 - Beta',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 3.7',
                        'Programming Language :: Python :: 3.8',
                        'Programming Language :: Python :: 3.9',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Scientific/Engineering :: Chemistry',
                        'Topic :: Scientific/Engineering :: Physics']
        }

####################################################################
# this is where setup starts
####################################################################


def setup_package():
    """
    Runs package setup
    """
    setup(**INFO)


if __name__ == '__main__':
    setup_package()
