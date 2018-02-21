from setuptools import setup, find_packages

requires = [
    'numpy',
    'scipy',
    ]

setup(name='pyre',
      version='5.0.3',
      description='PyRe (Python Reliability) is a python module for structural reliability analysis',
      long_description=open('README.rst').read(),
      license=open('LICENSE').read(),
      classifiers=[
        "Programming Language :: Python",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "License :: General Public License (GPL)",
        ],
      author='Juergen Hackl',
      author_email='hackl.j@gmx.at',
      url='http://github.com/hackl/pyre',
      keywords='structural reliability analysis',
      install_requires = requires,
      packages=find_packages(),
      )
