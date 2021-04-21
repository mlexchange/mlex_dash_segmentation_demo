from setuptools import setup, find_packages
from os import path
import sys

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup(
        name="mlex_api",
        version="0.0.1",
        packages=find_packages(include=['mlex_api', 'mlex_api.*']),
        description="API for interfacing MLExchange with the broader world",
        author="Adam Green",
        install_requires=requirements,
        license="BSD (3-clause)",
        
        )
