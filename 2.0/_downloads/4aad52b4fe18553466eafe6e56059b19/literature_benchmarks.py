"""Shared tutorial models; these are examples, not PySTRA's public API.

Truss: Marelli & Sudret (2018), Structural Safety 75, Section 3.2.
Hat: the UQLab 2.2.0 reliability example, also named in the 2022 review.
The analytical expressions were checked against the local UQLab examples.
Copyright (c) 2018–2026 Stefano Marelli and Bruno Sudret (ETH Zurich).
Adapted under BSD-3-Clause; see THIRD_PARTY_NOTICES and
../literature-benchmarks.rst for provenance and intentional differences.

Original UQLab notice (retained here for standalone helper downloads):
Copyright (c) 2018-2026, Stefano Marelli and Bruno Sudret (ETH Zurich)

Redistribution and use of UQLab in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of Stefano Marelli, Bruno Sudret or ETH Zurich nor
the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import gumbel_r, norm, qmc

import pystra as ra

HORIZONTAL_WEIGHTS = np.array([36, 100, 140, 140, 100, 36])
DIAGONAL_WEIGHTS = np.sqrt(2) * np.array([2, 6, 10, 10, 6, 2])


def truss_model():
    """Ten independent variables, in SI units; parameters are mean and std."""
    model = ra.StochasticModel()
    for name, mean in (("E1", 2.1e11), ("E2", 2.1e11), ("A1", 2e-3), ("A2", 1e-3)):
        model.add_variable(ra.Lognormal(name, mean, 0.1 * mean))
    for i in range(1, 7):
        model.add_variable(ra.Gumbel(f"P{i}", 5e4, 7.5e3))
    return model


def truss_deflection(E1, E2, A1, A2, P1, P2, P3, P4, P5, P6):
    """Downward midspan displacement in metres, by the unit-load method.

    Inputs may be scalars or broadcastable arrays. Horizontal members use
    E1*A1; the twelve diagonals use E2*A2. Coefficients have units m.
    """
    loads = np.broadcast_arrays(P1, P2, P3, P4, P5, P6)
    horizontal = sum(weight * load for weight, load in zip(HORIZONTAL_WEIGHTS, loads))
    diagonal = sum(weight * load for weight, load in zip(DIAGONAL_WEIGHTS, loads))
    return horizontal / (E1 * A1) + diagonal / (E2 * A2)


def truss_limit_state(E1, E2, A1, A2, P1, P2, P3, P4, P5, P6):
    """Failure when downward displacement reaches 0.12 m."""
    return 0.12 - truss_deflection(E1, E2, A1, A2, P1, P2, P3, P4, P5, P6)


def truss_reference(power=16, n_replications=8, seed=2026):
    """Independent conditional integration with scrambled Sobol points.

    Return one failure-probability estimate per independent scramble. Integrate
    P3 analytically, sampling the other five Gumbel loads and the two lognormal
    stiffness products. This uses neither a surrogate nor PySTRA transforms.
    Replicate dispersion measures integration error, not surrogate uncertainty.
    """
    load_scale = 7.5e3 * np.sqrt(6) / np.pi
    load_location = 5e4 - np.euler_gamma * load_scale
    log_variance = np.log1p(0.1**2)
    log_stiffness = np.log([2.1e11 * 2e-3, 2.1e11 * 1e-3]) - log_variance
    retained = [0, 1, 3, 4, 5]
    estimates = []
    for child in np.random.SeedSequence(seed).spawn(n_replications):
        points = qmc.Sobol(
            7, scramble=True, seed=np.random.default_rng(child)
        ).random_base2(power)
        stiffness = np.exp(
            log_stiffness + np.sqrt(2 * log_variance) * norm.ppf(points[:, :2])
        )
        coefficients = (
            HORIZONTAL_WEIGHTS / stiffness[:, :1] + DIAGONAL_WEIGHTS / stiffness[:, 1:2]
        )
        loads = gumbel_r.ppf(points[:, 2:], loc=load_location, scale=load_scale)
        threshold = (
            0.12 - np.sum(coefficients[:, retained] * loads, axis=1)
        ) / coefficients[:, 2]
        estimates.append(
            np.mean(gumbel_r.sf(threshold, loc=load_location, scale=load_scale))
        )
    return np.array(estimates)


def hat_model():
    """UQLab hat example: independent normal variables with mean 0.25, std 1."""
    model = ra.StochasticModel()
    for name in ("x1", "x2"):
        model.add_variable(ra.Normal(name, 0.25, 1))
    return model


def hat_limit_state(x1, x2):
    """Cubic hat function with a=4, b=8, c=20; failure is g <= 0."""
    return 20 - (x1 - x2) ** 2 - 8 * (x1 + x2 - 4) ** 3


def hat_reference():
    """Integrate exactly in the sum coordinate, quadrature in the difference."""

    def integrand(difference):
        threshold = (4 + np.cbrt((20 - 2 * difference**2) / 8) - 0.5) / np.sqrt(2)
        return norm.pdf(difference) * norm.sf(threshold)

    # Omitted normal tails are bounded by 2*Phi(-10) < 1.6e-23.
    return quad(integrand, -10, 10, points=[-np.sqrt(10), np.sqrt(10)], epsabs=1e-12)[0]
