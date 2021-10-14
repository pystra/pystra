from setuptools import setup, find_packages

requires = [
    "numpy",
    "scipy",
]

setup(
    name="pystra",
    version="0.1.0",
    description="Pystra (Python Structural Reliability Analysis) is a python module for structural reliability analysis",
    long_description="Coming Soon",
    license="AGPL-3.0+",
    classifiers=[
        "Programming Language :: Python",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    ],
    author="Juergen Hackl",
    author_email="hackl.j@gmx.at",
    url="http://github.com/pystra/pystra",
    keywords="structural reliability analysis",
    install_requires=requires,
    packages=find_packages(),
)
