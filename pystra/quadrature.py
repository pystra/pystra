#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np


def quadratureRule(n, wfun=None, alpha=None, beta=None):
    """Quadrature rule"""
    #  bp = base points (abscissas)
    #  wf = weight factors
    #  n  = number of base points (abscissas) (integrates a (2n-1)th order

    if beta == None:
        beta = 0
    if alpha == None:
        alpha = 0
    if alpha <= -1 or beta <= -1:
        print("Error: alpha and beta must be greater than -1")
    if wfun == None:
        wfun = 1

    if wfun == 1:
        #  This routine computes Gauss base points and weight factors
        #  using the algorithm given by Davis and Rabinowitz in 'Methods
        #  of Numerical Integration', page 365, Academic Press, 1975.
        bp = np.zeros(n)
        wf = bp
        iter = 2
        m = np.fix((n + 1) * 2 ** (-1))
        e1 = n * (n + 1)
        mm = int(4 * m - 1)
        vec = np.arange(3, mm + 1, 4)
        t = (np.pi * (4 * n + 2) ** (-1)) * vec
        nn = 1 - (1 - 1 * n ** (-1)) * (8 * n * n) ** (-1)
        xo = nn * np.cos(t)
        for j in range(iter):
            pkm1 = 1
            pk = xo
            for k in range(2, n + 1):
                t1 = xo * pk
                pkp1 = t1 - pkm1 - (t1 - pkm1) * (k) ** (-1) + t1
                pkm1 = pk
                pk = pkp1
            den = 1 - xo * xo
            d1 = n * (pkm1 - xo * pk)
            dpn = d1 * den ** (-1)
            d2pn = (2 * xo * dpn - e1 * pk) * den ** (-1)
            d3pn = (4 * xo * d2pn + (2 - e1) * dpn) * den ** (-1)
            d4pn = (6 * xo * d3pn + (6 - e1) * d2pn) * den ** (-1)
            u = pk * dpn ** (-1)
            v = d2pn * dpn ** (-1)
            h = -u * (1 + (0.5 * u) * (v + u * (v * v - u * d3pn * (3 * dpn) ** (-1))))
            p = pk + h * (
                dpn + (0.5 * h) * (d2pn + (h * 3 ** (-1)) * (d3pn + 25 * h * d4pn))
            )
            dp = dpn + h * (d2pn + (0.5 * h) * (d3pn + h * d4pn * 3 ** (-1)))
            h = h - p * dp ** (-1)
            xo = xo + h
        bp = -xo - h
        fx = d1 - h * e1 * (
            pk
            + (h * 0.5)
            * (
                dpn
                + (h * 3 ** (-1)) * (d2pn + (h * 4 ** (-1)) * (d3pn + (0.2 * h) * d4pn))
            )
        )
        wf = 2 * (1 - bp**2) * (fx * fx) ** (-1)
        if ((m - 1) + (m - 1)) > (n - 1):
            bp[m - 1] = 0
        if (m + m) != n:
            m = m - 1
        jj = np.arange(m)
        jj = jj.astype(int)
        n1j = m - 1 - jj
        n1j = n1j.astype(int)
        bp = np.append(bp, -bp[n1j])
        wf = np.append(wf, wf[n1j])
        return bp, wf
