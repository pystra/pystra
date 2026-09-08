Probability transformations and copulas
***************************************

Probability transformation
==========================


Classical FORM uses independent standard-normal coordinates. A probability
transformation maps the joint law of the physical variables into that space.
Generalized Nataf also permits spherical non-normal standard spaces, provided
the reliability calculation uses the corresponding probability law.

Transformation of dependent random variables using Nataf approach
-----------------------------------------------------------------

One method to handle this is using the Nataf joint distribution model, if the
marginal cdfs are known. [Baker2010]_ The correlated random variables
:math:`{\bf X} = ( X_1 , \dots , X_n )` with the correlation matrix :math:`\bf
R` can be transformed by

.. math::
   :label: eq:2_82

           y_i=\Phi^{-1}\left(F_{X_{i}}(x_i)\right) \qquad i = 1,\dots,n

into normally distributed random variables :math:`\bf Y` with zero means and
unit variance, but still correlated with :math:`{\bf R}_0` . Nataf’s
distribution for :math:`\bf X` is obtained by assuming that :math:`\bf Y` is
jointly normal. [Liu1986]_

The correlation coefficients for :math:`\bf X` and :math:`\bf Y` are related by

.. math::
   :label: eq:2_83

           \rho_{X_i,X_j} =
           \int\limits_{-\infty}^{\infty}\int\limits_{-\infty}^{\infty}
           \left(\frac{x_i-\mu_{X_i}}{\sigma_{X_i}}\right)
           \left(\frac{x_j-\mu_{X_j}}{\sigma_{X_j}}\right)
           \frac{1}{2\pi \sqrt{1-\rho_{Y_i,Y_j}^2}}
           \exp\left(-\frac{y_i^2-2\rho_{Y_i,Y_j}y_iy_j+y_j^2}{2(1-\rho_{Y_i,Y_j}^2)}\right) d y_i d y_j

Once this is done, the transformation from the correlated normal random
variables :math:`\bf Y` to uncorrelated normal random variables :math:`\bf Z`
is addressed. Hence, the transformation is

.. math::
   :label: eq:2_84

           {\bf z}={\bf L}_0^{-1}{\bf y} \quad \Leftrightarrow \quad {\bf y} =
           {\bf L}_0{\bf z}

where :math:`\mathbf{L}_0\mathbf{L}_0^T=\mathbf{R}_0` is the
Cholesky factorisation of the correlation matrix of :math:`\bf Y`. The Jacobian matrix, denoted by :math:`\bf J`,
for the transformation is given by

.. math::
   :label: eq:2_85

           {\bf J}_{ZX} = \frac{\partial {\bf z}}{\partial {\bf x}} = {\bf
           L}_0^{-1}\text{diag} \left(\frac{f_{X_i}(x_i)}{\varphi(y_i)}\right)

This approach is useful when the marginal distribution for the random
variables :math:`\bf X` is known and the knowledge about the variables
dependence is limited to correlation coefficients. [Baker2010]_
[DerKiureghian2006]_

Transformation of dependent random variables using Rosenblatt approach
----------------------------------------------------------------------

An alternative to the Nataf approach is to consider the joint pdf of
:math:`\bf X` as a product of conditional pdfs.

.. math::
   :label: eq:2_86

           f_{{\bf X}}({\bf x}) = f_{X_1}(x_1) f_{X_2|X_1}(x_2|x_1) \dots
           f_{X_n|X_1,\dots,X_{n-1}}(x_n|x_1,\dots,x_{n-1})

As a result of the sequential conditioning in the pdf, the conditional cdfs
are given for :math:`i \in [1,n]`

.. math::
   :label: eq:2_87

           F_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) =
           \int_{-\infty}^{x_i}
           f_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) d x_i

These conditional distributions for the random variables :math:`\bf X` can be
transformed into standard normal marginal distributions for the variables
:math:`\bf Z`, using the so called Rosenblatt transformation
[Rosenblatt1952]_, suggested by Hohenbichler and Rackwitz (1981).

.. math::
   :label: eq:2_88

           \begin{split}
           z_1 &= \Phi^{-1}\left( F_{X_1}(x_1) \right)\\
           z_2 &= \Phi^{-1}\left( F_{X_2|X_1}(x_2|x_1) \right)\\
           &\vdots\\
           z_n &= \Phi^{-1}\left(
           F_{X_n|X_1,\dots,X_{n-1}}(x_n|x_1,\dots,x_{n-1}) \right)
           \end{split}

The Jacobian of this transformation is a lower triangular matrix having the
elements [Baker2010]

.. math::
   :label: eq:2_89

           \left[{\bf J}_{ZX}\right]_{i,j} = \frac{\partial z_i}{\partial x_j} =
           \begin{cases}\displaystyle
           \frac{1}{\varphi(z_i)}\frac{\partial}{\partial x_j}
           F_{X_i|X_1,\dots,X_{i-1}}(x_i|x_1,\dots,x_{i-1}) & i \geq j\\
           0 & i < j
           \end{cases}

Here :math:`\varphi` is the standard-normal density, and the triangular
structure follows the chosen conditioning order.

In some cases the Rosenblatt transformation cannot be applied, because the
required conditional pdfs cannot be provided. In this case other
transformations may be useful, for example Nataf transformation.
[Faber2009]_


.. _theory_copulas:
.. _/theory.rst#theory-copulas:

Copulas as joint distribution specifications
--------------------------------------------

For continuous marginals, dependence is specified by a copula :math:`C`:

.. math::

   F_{\mathbf X}(\mathbf x)=C(F_1(x_1),\ldots,F_n(x_n)),\qquad
   f_{\mathbf X}(\mathbf x)=c(F_1(x_1),\ldots,F_n(x_n))\prod_i f_i(x_i).

The copula and marginals define the joint law before a transformation is
chosen. Physical Pearson correlations alone do not determine that law.
Pystra's legacy Pearson interface assumes a Gaussian copula and calibrates
its latent correlation using :eq:`eq:2_83`. An explicit ``GaussianCopula``
already specifies the latent correlation, so that calibration is bypassed.
Changing to Student-t or Frank changes the dependence law, rather than just
changing a numerical factorisation.

For Gaussian and Student-t copulas, the pairwise Kendall rank correlations
and latent unit-diagonal shape entries satisfy

.. math::

   \tau_{ij}=\frac{2}{\pi}\arcsin R_{ij}.

The resulting full shape matrix must be positive definite. Arbitrary
physical marginals can be used with the same copula. See
:doc:`/notebooks/ex_copulas` for sampling and density examples, and
:ref:`chap_copulas` for the supported specification interface.

Generalized Nataf for elliptical copulas
----------------------------------------

Following [LebrunDutfoy2009a]_, let :math:`H` be the univariate CDF of a
centred elliptical representative with unit-diagonal shape matrix
:math:`R=LL^T`. Generalized Nataf is

.. math::

   w_i=H^{-1}(F_i(x_i)),\qquad \mathbf v=L^{-1}\mathbf w,
   \qquad x_i=F_i^{-1}(H((L\mathbf v)_i)).

The transformed distribution is spherical. Its coordinates are independent
only in the Gaussian case within these Gaussian/Student-t families.
For Student-t, :math:`H=T_\nu` and :math:`\mathbf V` is spherical Student-t
with identity shape. When :math:`\nu>2`, its covariance is
:math:`\nu/(\nu-2)I`; for smaller degrees of freedom, covariance does not
exist. Identity shape therefore must not be interpreted as independence.
The Nataf Jacobian is

.. math::

   J_{VX}=L^{-1}\operatorname{diag}\left(\frac{f_i(x_i)}{h(w_i)}\right),

where :math:`h=H'`. Spherical symmetry makes every unit projection have CDF
:math:`H`. Hence a tangent failure half-space at signed geometric distance
:math:`\beta` has probability

.. math::

   P_{f,\mathrm{FORM}}=H(-\beta).

For Student-t, Pystra uses :math:`T_\nu(-\beta)`, while
``get_equivalent_beta()`` reports :math:`-\Phi^{-1}(P_f)`.
SORM, system FORM, simulation methods and the current Strong Maximum Test
require independent normal coordinates and reject spherical Student-t Nataf
space. A normal Rosenblatt mapping supports these analyses with the same
Student-t copula.

Rosenblatt conditioning and approximation invariance
----------------------------------------------------

A copula supplies the conditional CDFs required by Rosenblatt. For an order
:math:`\pi`, define :math:`p_i=F_i(x_i)` and

.. math::

   q_{\pi_k}=C_{\pi_k\mid\pi_1,\ldots,\pi_{k-1}}
       (p_{\pi_k}\mid p_{\pi_1},\ldots,p_{\pi_{k-1}}),\qquad
   u_{\pi_k}=\Phi^{-1}(q_{\pi_k}).

The :math:`q_i` are independent uniforms and the :math:`u_i` are independent
standard normals. Pystra keeps coordinate arrays in original marginal order;
``u[order[k]]`` denotes the k-th conditional innovation.

For a Student-t copula, reorder the latent vector and shape first, and write
:math:`\mathbf z=L^{-1}\mathbf w`. With one-based conditional index :math:`k`,

.. math::

   s_k^2=\frac{\nu+\sum_{j<k}z_j^2}{\nu+k-1},\qquad
   q_{\pi_k}=T_{\nu+k-1}(z_k/s_k).

The inverse constructs each :math:`z_k=s_kT_{\nu+k-1}^{-1}(q_{\pi_k})`
sequentially, then recovers the latent vector and physical marginals.
This additional conditional scaling removes the dependence that remains
after Student-t Nataf whitening.

For a Gaussian copula, Cholesky Nataf and Rosenblatt coincide for a fixed
order. Different orders are related by an orthogonal change of normal
coordinates; the optimum FORM distance and probability are invariant.
Numerical optimization can still converge to different local points.
For non-Gaussian copulas, transformations need not be related orthogonally:
exact probability remains invariant, but FORM's tangent approximation can
change with conditioning order [LebrunDutfoy2009b]_. All system components
must use the same mapping and order before their normal directions are
compared.

The bivariate Frank copula used in that paper is

.. math::

   C_\theta(p_1,p_2)=-\frac{1}{\theta}\log\left[
     1+\frac{(e^{-\theta p_1}-1)(e^{-\theta p_2}-1)}{e^{-\theta}-1}\right],

with the independence limit at :math:`\theta=0`. Its conditional CDF is
:math:`C_{2\mid1}=\partial C_\theta/\partial p_1`.
For exponential rates 1 and 3, :math:`\theta=10`, and failure event
:math:`8X_1+2X_2-1\leq0`, the tutorial reproduces FORM probabilities about
0.107 and 0.122 for the two orders, compared with direct integration of
about 0.1038. This difference is approximation error, not a different
underlying probability for each order.

**Use this method:** :doc:`/copulas` · :doc:`/notebooks/ex_copulas` · :doc:`/api/probability`

For coordinate conventions, see :doc:`notation`.
