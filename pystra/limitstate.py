#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np
import types


def evaluateLimitState(
    x, stochastic_model, analysis_options, limit_state, diff_mode=None
):
    """Evaluate the limit state"""

    # names = stochastic_model.getNames()
    expression = limit_state.getExpression()

    nx = x.shape[1]
    nrv = x.shape[0]

    G = np.zeros((1, nx))
    grad_g = np.zeros((nrv, nx))

    if diff_mode == None:
        diff_mode = analysis_options.getDiffMode()
    else:
        diff_mode = "no"

    if analysis_options.getMultiProc() == 0:
        raise NotImplementedError("getMultiProc")
    else:
        block_size = analysis_options.getBlockSize()
        # No differentiation for MCS
        if diff_mode == "no":
            if nx > 1:
                k = 0
                while k < nx:
                    block_size = np.min([block_size, nx - k])
                    indx = list(range(k, k + block_size))
                    blockx = x[:, indx]

                    blockG, _ = computeLimitStateFunction(
                        blockx, stochastic_model, expression
                    )

                    G[:, indx] = blockG
                    # grad_g[indx] = blockdummy
                    k += block_size

                stochastic_model.addCallFunction(nx)

        elif diff_mode == "ddm":
            for k in range(nx):
                G[k], grad_g[:, k : k + 1] = computeLimitStateFunction(
                    x[:, k : k + 1], stochastic_model, expression, ddm=True
                )
            stochastic_model.addCallFunction(nx)

        elif diff_mode == "ffd":
            ffdpara = analysis_options.getffdpara()
            allx = np.zeros((nrv, nx * (1 + nrv)))
            allx[:] = x
            allh = np.zeros(nrv)

            marg = stochastic_model.getMarginalDistributions()

            x0 = x
            for j in range(nrv):
                x = x0
                allh[j] = marg[j].stdv / ffdpara
                x[j] = x[j] + allh[j] * np.ones(nx)
                indx = list(range(j + 1, 1 + (1 + j + (nx - 1) * (1 + nrv)), (1 + nrv)))
                allx[j, indx] = x[j]

            allG = np.zeros(nx * (1 + nrv))

            k = 0
            while k < (nx * (1 + nrv)):
                block_size = np.min([block_size, nx * (1 + nrv) - k])
                indx = list(range(k, k + block_size))
                blockx = allx[:, indx]

                blockG, _ = computeLimitStateFunction(
                    blockx, stochastic_model, expression
                )

                allG[indx] = blockG.squeeze()
                k += block_size

            indx = list(range(0, (1 + (nx - 1) * (1 + nrv)), (1 + nrv)))
            G = allG[indx]

            for j in range(nrv):
                indx = list(range(j + 1, 1 + (1 + j + (nx - 1) * (1 + nrv)), (1 + nrv)))
                grad_g[j, :] = (allG[indx] - G) / allh[j]

            stochastic_model.addCallFunction(nx * (1 + nrv))

    return G, grad_g


def computeLimitStateFunction(x, stochastic_model, expression, ddm=False):
    """Compute the limit state function"""
    _, nc = np.shape(x)
    variables = stochastic_model.getVariables()
    constants = stochastic_model.getConstants()

    inpdict = dict()
    for i, var in enumerate(variables):
        inpdict[var] = x[i]
    for c, val in constants.items():
        inpdict[c] = val * np.ones(nc)
    Gvals = expression(**inpdict)
    try:
        if ddm:
            G, gradient = Gvals
        else:
            if isinstance(Gvals, tuple):
                G = Gvals[0]
            else:
                G = Gvals
            gradient = 0
    except TypeError:
        raise TypeError("Limit state function return must match differentiation mode")

    return G, gradient
