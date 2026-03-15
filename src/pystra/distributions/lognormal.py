#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import lognorm
from .distribution import Distribution


class Lognormal(Distribution):
    """Lognormal distribution

    :Arguments:
      - name (str):         Name of the random variable
      - mean (float):       Mean or lamb
      - stdv (float):       Standard deviation or zeta\n
      - input_type (any):   Change meaning of mean and stdv\n
      - startpoint (float): Start point for seach\n

    Note: Could use scipy to do the heavy lifting. However, there is a small
    performance hit, so for this common dist use bespoke implementation
    for the PDF, CDF.
    """

    def __init__(self, name, mean, stdv, input_type=None, startpoint=None):
        if input_type is None:
            # infer parameters from the moments
            self._update_params(mean, stdv)
        else:
            # parameters directly passed in
            self.lamb = mean
            self.zeta = stdv

        # Could use scipy to do the heavy lifting. However, there is a small
        # performance hit, so for this common dist use bespoke implementation
        # for the PDF, CDF.
        # Careful: the scipy parametrization is tricky!
        self.dist_obj = lognorm(scale=np.exp(self.lamb), s=self.zeta)

        super().__init__(
            name=name,
            dist_obj=self.dist_obj,
            startpoint=startpoint,
        )

        self.dist_type = "Lognormal"

    def _update_params(self, mean, stdv):
        cov = stdv / mean
        self.zeta = (np.log(1 + cov**2)) ** 0.5
        self.lamb = np.log(mean) - 0.5 * self.zeta**2

    # Overriding base class implementations for speed

    def pdf(self, x):
        """
        Probability density function
        Note: asssumes x>0 for performance, scipy manages this appropriately
        """
        z = (np.log(x) - self.lamb) / self.zeta
        p = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * self.zeta * x)
        return p  # self.lognormal.pdf(x)

    def cdf(self, x):
        """
        Cumulative distribution function
        """
        z = (np.log(x) - self.lamb) / self.zeta
        p = 0.5 + math.erf(z / np.sqrt(2)) / 2
        return p  # self.lognormal.cdf(x)

    def u_to_x(self, u):
        """
        Transformation from u to x
        """
        x = np.exp(u * self.zeta + self.lamb)
        return x

    def x_to_u(self, x):
        """
        Transformation from x to u
        Note: asssumes x>0 for performance
        """
        u = (np.log(x) - self.lamb) / self.zeta
        return u

    def dF_dtheta(self, x):
        r"""Analytical derivatives of the Lognormal CDF w.r.t. μ and σ.

        The CDF is ``F(x) = Φ((ln x - λ) / ζ)`` where
        ``ζ = sqrt(ln(1 + (σ/μ)²))`` and ``λ = ln(μ) - ζ²/2``.

        The chain rule gives:

        .. math::
            \frac{\partial F}{\partial \theta}
            = \frac{\varphi(z)}{\zeta}
              \left(-\frac{\partial\lambda}{\partial\theta}
                    - z\,\frac{\partial\zeta}{\partial\theta}\right)

        where ``z = (ln x - λ) / ζ``.
        """
        cov = self.stdv / self.mean
        cov2 = cov**2
        z = (np.log(x) - self.lamb) / self.zeta
        phi_z = self.std_normal.pdf(z)

        # Derivatives of ζ and λ w.r.t. μ and σ
        # ζ² = ln(1 + cov²),  cov = σ/μ
        # ∂ζ/∂μ = (1/ζ) × (1/(1+cov²)) × (-cov²/μ) = -cov² / (μ ζ (1+cov²))
        # ∂ζ/∂σ = (1/ζ) × (1/(1+cov²)) × (cov/μ)   =  cov  / (μ ζ (1+cov²))
        dzeta_dmu = -cov2 / (self.mean * self.zeta * (1 + cov2))
        dzeta_dsig = cov / (self.mean * self.zeta * (1 + cov2))

        # λ = ln(μ) - ζ²/2
        # ∂λ/∂μ = 1/μ - ζ ∂ζ/∂μ
        # ∂λ/∂σ = -ζ ∂ζ/∂σ
        dlamb_dmu = 1.0 / self.mean - self.zeta * dzeta_dmu
        dlamb_dsig = -self.zeta * dzeta_dsig

        # ∂F/∂θ = (φ(z)/ζ) × (-∂λ/∂θ - z ∂ζ/∂θ)
        coeff = phi_z / self.zeta
        dF_dmu = coeff * (-dlamb_dmu - z * dzeta_dmu)
        dF_dsig = coeff * (-dlamb_dsig - z * dzeta_dsig)

        return {"mean": dF_dmu, "std": dF_dsig}

    def set_location(self, loc=0):
        """
        Updating the distribution location parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update pe.arams directly.
        """

        self._update_params(loc, self.stdv)
        self.mean = loc

    def set_scale(self, scale=1):
        """
        Updating the distribution scale parameter.
        For Lognormal, even though we have a SciPy object, it's not being used in the
        functions above for performance, so we need to update params directly.
        """
        self._update_params(self.mean, scale)
        self.stdv = scale
