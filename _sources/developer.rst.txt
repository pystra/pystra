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

	conda create -n pystra-dev python=3.12


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

4. [If applicable] Create unit tests for ``pytest``.
    
    * Store test file in ``./tests/<test-file.py>``.

5. [If applicable] Create new example notebook.
    
    * Store notebook in ``./docs/source/notebooks/<tutorial.ipynb>``.
    * Index notebook in ``./docs/source/tutorial.rst``

6. [If applicable] Add new dependencies in ``./pyproject.toml``

7. Build documentation

	* Change directory to ``./docs/``
	* ``make clean``
	* ``make html``
	* ``xdg-open build/html/index.html``

8. Update version number in ``./src/pystra/__init__.py`` (the docs
   version is derived automatically via ``conf.py``).

9. Stage changes; commit; and push to remote fork

10. Go to GitHub and create PR for the branch


.. _adding_distributions:

Adding a New Distribution
-------------------------

All distributions inherit from
:class:`~pystra.distributions.distribution.Distribution`.  Two
approaches are available:

- **Wrapping a SciPy distribution** (most common) — construct a
  ``scipy.stats`` frozen distribution object and pass it as
  ``dist_obj`` to ``super().__init__()``.  The base class then
  delegates ``pdf``, ``cdf``, ``ppf``, and the Nataf-space
  transformations automatically.
- **Hardcoded implementation** — override the transformation and
  Jacobian methods directly (see :class:`~pystra.distributions.zeroinflated.ZeroInflated`
  for an example).  This is useful for distributions
  that cannot be expressed as a single SciPy object.

To make the distribution work correctly with **sensitivity analysis**
there are a few additional conventions to follow.

Extra constructor arguments (``_ctor_kwargs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your distribution's constructor takes arguments beyond
``(name, mean, stdv)`` — for instance bounds, shift, or shape
parameters — store them as attributes *and* populate ``_ctor_kwargs``
**before** calling ``super().__init__()``::

    class MyDist(Distribution):
        def __init__(self, name, mean, stdv, shape, epsilon=0):
            self.shape = shape
            self.epsilon = epsilon
            self._ctor_kwargs = {"shape": shape, "epsilon": epsilon}

            # Build scipy distribution object ...
            self.dist_obj = ...

            super().__init__(name=name, dist_obj=self.dist_obj)

The base-class method
:meth:`~pystra.distributions.distribution.Distribution._make_copy`
uses ``_ctor_kwargs`` to reconstruct the distribution when parameters
are perturbed during sensitivity analysis.  Without it, reconstruction
fails or silently produces wrong results.

Declaring sensitivity parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, sensitivities are computed with respect to the mean and
standard deviation.  If your distribution has additional parameters of
physical interest (e.g. a shape parameter that controls tail
behaviour), override
:attr:`~pystra.distributions.distribution.Distribution.sensitivity_params`::

    @property
    def sensitivity_params(self):
        return {
            "mean": self.mean,
            "std": self.stdv,
            "shape": self.shape,
        }

**Important distinction:** parameters in ``_ctor_kwargs`` but *not* in
``sensitivity_params`` are held fixed during sensitivity analysis.
For example, the Beta distribution stores its bounds ``a`` and ``b``
in ``_ctor_kwargs`` (so ``_make_copy`` can reconstruct it) but does
not add them to ``sensitivity_params`` (bounds are treated as fixed
constants, not sensitivity parameters).

Analytical CDF derivatives (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base class computes :math:`\partial F_X/\partial\theta` numerically
via central differences.  For better accuracy and performance you can
override
:meth:`~pystra.distributions.distribution.Distribution.dF_dtheta`
with analytical expressions.  The Normal and Lognormal distributions
do this::

    def dF_dtheta(self, x):
        z = (x - self.mean) / self.stdv
        phi_z = self.std_normal.pdf(z)
        return {
            "mean": -phi_z / self.stdv,
            "std": -(x - self.mean) * phi_z / self.stdv**2,
        }

The returned dict must have the same keys as ``sensitivity_params``.

Verifying your distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Basic reliability analysis** — the distribution should work in a
   simple FORM problem.  Build a small model with your distribution,
   run FORM, and check that the reliability index is sensible::

       model = ra.StochasticModel()
       model.addVariable(MyDist("X", 100, 15, shape=0.2))
       model.addVariable(ra.Normal("Y", 50, 10))
       ls = ra.LimitState(lambda X, Y: X - Y)
       f = ra.Form(stochastic_model=model, limit_state=ls)
       f.run()
       f.showDetailedOutput()

2. **Round-trip reconstruction** — if you set ``_ctor_kwargs``,
   verify that ``_make_copy()`` with no overrides produces a
   distribution whose CDF matches the original::

       d = MyDist("X", 100, 15, shape=0.2)
       d2 = d._make_copy()
       assert abs(d.cdf(110) - d2.cdf(110)) < 1e-10

3. **Sensitivity analysis** — the closed-form method exercises a lot
   of the distribution plumbing (``dF_dtheta``, ``_dmoments_dtheta``,
   ``_make_copy``), so running both methods is a good integration
   check::

       sa = ra.SensitivityAnalysis(limit_state, model)
       fd = sa.run(numerical=True)    # finite-difference
       cf = sa.run(numerical=False)   # closed-form

   FD and CF sensitivities should agree (typically within 5 % for
   mean/std, possibly 10–15 % for shape parameters due to inherent
   FD instability).

See ``tests/test_sensitivity.py`` for concrete examples.
