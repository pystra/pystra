from setuptools import setup, find_packages

requires = [
    'numpy',
    'scipy',
]

setup(name='pystra',
      version='0.0.1',
      description='PyStRA (Python Structural Reliability Analysis) is a python module for structural reliability analysis',
      long_description=open('README.rst', encoding='utf8').read(),
      license=open('LICENSE', encoding='utf8').read(),
      classifiers=[
        "Programming Language :: Python",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "License :: General Public License (GPL)",
        ],
      author='Juergen Hackl',
      author_email='hackl.j@gmx.at',
      url='http://github.com/pystra/pystra',
      keywords='structural reliability analysis',
      install_requires = requires,
      packages=find_packages(),
      )
