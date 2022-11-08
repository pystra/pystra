Instructions for Developers
===========================

This file presents instructions for ``Pystra`` developers.

.. _install_dev:

Create working repository with developer install
------------------------------------------------

1. Fork ``Pystra`` `GitHub repository <https://github.com/pystra/pystra/>`_

2. Clone repo ::

	git clone <forked-repo>


3. Create new `Pystra` developer environment ::
	
	conda create -n pystra-dev python=3.10


4. Activate developer environment ::

	conda activate pystra-dev

5. Change directory to pystra fork


6. Install developer version ::

	pip install -e .

7. Install depedencies ::

	conda install -c anaconda pytest
	conda install -c anaconda sphinx
	conda install -c conda-forge black
	pip install sphinx_autodoc_typehints
	pip install nbsphinx
	pip install pydata_sphinx_theme

8. Add ``Pystra`` as upstream ::

	git remote add upstream https://github.com/pystra/pystra.git

.. _pr:

Develop and create pull-request (PR)
------------------------------------

1. Create new branch ::

	git checkout -b <new-branch>

2. Pull updates from Pystra main ::
	
	git pull upstream main 

3. Develop package

4. [If applicable] Create unit tests for ``pytest`` and store test file in
``./tests/<test-file.py>``

5. [If applicable] Add new dependencies in ``./setup.cfg``

6. Build documentation

	* Change directory to ``./docs/``
	* ``make clean``
	* ``make html``
	* ``xdg-open build/html/index.html``

7. Update version number in the following files

	* ``./docs/source/index.rst``
	* ``./pystra/__init__.py``

8. Stage changes; commit; and push to remote fork

9. Go to GitHub and create PR for the branch
