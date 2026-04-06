"""Packaging settings."""

from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from pyMV2H_with_harmony import __version__

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


class RunTests(Command):
    """Run all tests."""
    description = 'run tests'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run all tests!"""
        errno = call(['py.test', '--cov=pyMV2H', '--cov-report=term-missing'])
        raise SystemExit(errno)


setup(
    name='pyMV2H',
    version=__version__,
    description='A python implementation of MV2H metric',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/lucasmpaim/pyMV2H',
    author='Lucas Mrowskovsky Paim',
    author_email='lucas.mrowskovsky@pucpr.edu.br',
    license='UNLICENSE',
    classifiers=[],
    keywords='cli',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=['docopt', 'pretty_midi', 'mido', 'tqdm'],
    extras_require={
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    entry_points={
        'console_scripts': [
            'pyMV2H=pyMV2H.cli:main',
        ],
    },
    cmdclass={'test': RunTests},
)
