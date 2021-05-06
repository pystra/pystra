#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy as np

from .form import Form
from .model import StochasticModel, AnalysisOptions, LimitState
from .limitstate import evaluateLimitState, computeLimitStateFunction
from .transformation import jacobian, x_to_u, u_to_x
from .distributions import Normal


class Sorm(object):
    """
    Second Order Reliability Method (SORM)

    Unlike FORM, this method approximates the failure surface in standard
    normal space using a quadratic surface. The basic approximation is given
    by [Breitung1984]_:

        .. math::
       :label: eq:2_91

              p_f \\approx p_{f2} = \Phi(-\\beta) \\Pi_{i=1}^{n-1}\\[ 1 + \\kappa_i \\beta \\]^{-0.5}

    The corresponding generalized reliability index
    :math:`\\beta_G \\= -\\Phi^{-1}\\( 1-p_{f2}\\)` is reported. The Breitung
    approximation is asymptotic and accurate for higher values of \\beta.
    Also reported is the Hohenbichler and Rackwitz modification to Brietung's
    formula, which is more accurate for lower values of \\beta.

    :Attributes:
      - analysis_option (AnalysisOption): Option for the structural analysis
      - limit_state (LimitState): Information about the limit state
      - stochastic_model (StochasticModel): Information about the model
      - form (Form): Form object, if a FORM analysis has already been completed
    """

    def __init__(self, analysis_options=None, limit_state=None,
                 stochastic_model=None, form=None):
        """
        Class constructor
        """

        # The stochastic model
        if stochastic_model is None:
            self.model = StochasticModel()
        else:
            self.model = stochastic_model

        # Options for the calculation
        if analysis_options is None:
            self.options = AnalysisOptions()
        else:
            self.options = analysis_options

        # The limit state function
        if limit_state is None:
            self.limitstate = LimitState()
        else:
            self.limitstate = limit_state

        # Has FORM already been run? If it exists it has, otherwise run it now
        if form is None:
            self.form = Form(self.options, self.limitstate, self.model)
        else:
            self.form = form

        self.betaHL = 0
        self.kappa = 0
        self.pf2_breitung = 0
        self.betag_breitung = 0
        self.pf2_breitung_m = 0
        self.betag_breitung_m = 0
        # Could add Tvedt too

    def run(self,fit_type='cf'):
        """
        Run SORM analysis using either:
            Curve-fitting: fit_type == 'cf'
            Point-fitting: fit_type == 'pf'
        """
        if fit_type != 'cf':
            raise ValueError("Point-Fitting not yet supported")
        else:
            self.run_curvefit()

    def run_curvefit(self):
        """
        Run SORM analysis using curve fitting
        """
        hess_G = self.computeHessian()
        R1 = self.orthonormal_matrix()
        A = R1 @ hess_G @ R1.T / np.linalg.norm(self.form.gradient)
        kappa,_ = np.linalg.eig(A[:-1,:-1])
        self.betaHL = self.form.beta
        self.kappa = np.sort(kappa)
        self.pf_breitung(self.betaHL,self.kappa)
        self.pf_breitung_m(self.betaHL,self.kappa)
        self.showResults()
        
        
    def run_fittingpoint(self):
        threshold = 10 ** -6
        stop_flag = 0
        
        # Useful parameters after Form Analysis
        beta = self.form.beta
        alpha = self.form.alpha
        R1 = self.orthonormal_matrix()
        nrv = self.model.getLenMarginalDistributions()
        
        # Determine the side of the axis the fitting point belongs to with respect to the matrix U_prime_1 construction
        i = num
        
        if num <= nrv - 1:    # Negative axis
            counter = i
            sign = -1
        else:                 # Positive axis
            counter = i - nrv + 1
            sign = 1
        end
        
        vect = np.zeros((nrv,1))            
        u_prime_i = U_prime_1[:,i]   # Define the coordinates of the fitting point in the rotated space
        a = u_prime_i[counter]
        b = u_prime_i[nrv]
        
        u = np.transpose(R1) * u_prime_i   # Rotation to the standard normal space
        x = u_to_x(u, self.model)
        J_u_x = jacobian(u, x, self.model)
        J_x_u = np.linalg.inv(J_u_x)
        G, grad = self.evaluateLSF(x,
        grad = R1 * np.transpose(grad*J_x_u)
        
        
        # Case where G is negative at the starting points
        if G < 0:
            a = sign * k * beta
            dadb = 0

            vect[counter] = dadb
            vect[nrv] = 1

        while stop_flag == 0:
            # for j = 1:4
            b = b - G / (np.transpose(grad_G) * vect)
            a = sign * k * beta

            # new point coordinates
            u_prime_i[counter] = a
            u_prime_i[nrv] = b

            u = np.transpose(R1) * u_prime_i
            x = u_to_x(u, self.model)
            J_u_x = jacobian(x, u, self.model)
            J_x_u = inv(J_u_x)

            G, grad = self.evaluateLSF(x,
            grad_G = R1 * np.transpose(grad*J_x_u)


        if abs(G) < threshold:
            stop_flag = 1
        end

        end
        u_prime_i()
        G_u = G

        # Case where G is positive at the starting points      
        else:    
            a = sign * ((k * beta) ** 2 - (b - beta) ** 2) ** 0.5
            dadb = sign * (-(b - beta) / ((k * beta) ** 2 - (b - beta) ** 2) ** 0.5)

            vect[counter] = dadb
            vect[nrv] = 1

        while stop_flag == 0:
            #  for j = 1:4

            b = b - G / (np.transpose(grad_G) * vect)
            a = sign * ((k * beta) ** 2 - (b - beta) ** 2) ** 0.5

            # New point coordinates
            u_prime_i[counter] = a
            u_prime_i[nrv] = b

            u = np.transpose(R1) * u_prime_i
            x = u_to_x(u, self.model)
            J_u_x = jacobian(x, u, self.model)
            J_x_u = inv(J_u_x)

            G, grad = self.evaluateLSF(x,
            grad_G = R1 * np.transpose(grad * J_x_u)

        if abs(G) < threshold:
            stop_flag = 1
        end

            dadb = sign * (-(b - beta) / ((k * beta) ** 2 - (b - beta) ** 2) ** 0.5)
            vect[counter] = dadb
        end
        
        u_prime_i()
        G_u = G
            
        
    def run_pointfit_mod(self):
        # Useful parameters after Form analysis
        beta = self.form.beta
        alpha = self.form.pf
        alpha = self.form.alpha
        itera = self.form.i
        
        # Design point
        dsptx = self.form.x
        dsptu = self.form.u
        
        # Rotation matrix obtained by Gram-Schmidt scheme
        R1 = self.orthonormal_matrix()
        nrv = self.model.getLenMarginalDistributions()
        
        # Determination of the coefficient k
        if abs(beta) < 1:
            k = 1 / abs(beta)
        elif abs(beta) >= 1 and abs(beta) <= 3:
            k = 1
        else:
            k = 3 / abs(beta)
        end
        
        # Initial trial points of ordinates +beta
        U_prime_1 = np.array([[-k*beta*np.eye(nrv-1),k*beta*np.eye(nrv-1)],[beta*np.ones(1,nrv-1),beta*np.ones(1,nrv-1)]])

        
        # Determination of the fitting points in the rotated space
        # Compute the fitting points on the negative side of axes and then on positive side of axes
        for i in range(2*nrv-1):
            num = i
            u_prime_i = self.fittingpoint.u_prime_i
            G_u = self.fittingpoint.G_u
            u_prime_final[:,i] = u_prime_i
            g_final[i] = G_u
        end
        
        # Compute the curvatures a_i_+/-
        for i in range(nrv-1):
             a_curvatures_minus[i] = 2 * (U_prime_final_negative(nrv, i) - beta) / (U_prime_final_negative(i, i)) ** 2
             a_curvatures_plus[i] = 2 * (U_prime_final_positive(nrv, i) - beta) / (U_prime_final_positive(i, i)) ** 2
        end
        a_curvatures_plus()
        a_curvatures_minus()
        
        kappa_plus_minus[1,:] = a_curvatures_plus
        kappa_plus_minus[2,:] = a_curvatures_minus
        
        
        # Along minus axis
        U_prime_minus =  np.zeros((nrv-1,5))
        minus_array = [*range(1,nrv,1)]
        minus_arrayt = np.transpose(minus_array)
        U_prime_minus[:,1] = round(minus_arrayt)
        
        for i in range(nrv-1):
            U_prime_minus[i,2] = U_prime_final[i][i]
        end
        
        U_prime_minus[:,3] = np.transpose(U_prime_final_negative[nrv])
        U_prime_minus[:,4] = np.transpose(g_final[1:nrv-1])
        U_prime_mminuys[:,5] = np.transpose(a_curvatures_minus)
        U_prime_min()
        
        # Along plus axis
        U_prime_plus = np.zeros((nrv-1,5))
        plus_array = [*range(1,nrv,1)]
        U_prime_plus[:,1] = np.transpose(plus_array)
        
        for i in range(nrv,2*(nrv-1)):
            U_prime_plus[i-nrv+1,2] = U_prime_final[i-nrv+1][i]
        end
        
        U_prime_plus[:,3] = np.transpose(U_prime_final_positive[nrv])
        U_prime_plus[:,4] = np.transpose(g_final[nrv:2*(nrv-1)])
        U_prime_plus[:,5] = np.transpose(a_curvatures_plus)
        U_prime_plus()
            

        # Breitung
        
    

    def pf_breitung(self,beta,kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using Breitung's (1984) formula. This formula is good for higher
        values of beta.

        Breitung, K. 1984 Asymptotic approximations for multinormal integrals.
        J. Eng. Mech. ASCE 110, 357–367.
        """
        is_invalid = np.any(kappa < -1/beta)
        if not is_invalid:
            self.pf2_breitung = Normal.cdf(-beta, 0, 1)*np.prod((1+beta*kappa)**(-0.5))
            self.betag_breitung = -Normal.inv_cdf(self.pf2_breitung)
        else:
            print("*** SORM Breitung error, excessive curavtures")
            self.pf2_breitung = 0
            self.betag_breitung = 0

    def pf_breitung_m(self,beta,kappa):
        """
        Calculates the probability of failure and generalized reliability
        index using Breitung's (1984) formula as modified by Hohenbichler and
        Rackwitz (1988). This formula is better for lower values of beta.

        Hohenbichler, M. & Rackwitz, R. 1988 Improvement of second-order
        reliability estimates by importance sampling.
        J. Eng. Mech. ASCE 14, 2195–2199.
        """
        k = Normal.pdf(beta,0,1)/Normal.cdf(-beta,0,1)
        is_invalid = np.any(kappa < -1/k)
        if not is_invalid:
            self.pf2_breitung_m = Normal.cdf(-beta, 0, 1)*np.prod((1+k*kappa)**(-0.5))
            self.betag_breitung_m = -Normal.inv_cdf(self.pf2_breitung_m)
        else:
            print("*** SORM Breitung Modified error")
            self.pf2_breitung_m = 0
            self.betag_breitung_m = 0

    def showResults(self):
        n_hyphen = 54
        print('')
        print("=" * n_hyphen)
        print('')
        print('RESULTS FROM RUNNING SECOND ORDER RELIABILITY METHOD')
        print('')
        print('Generalized reliability index: ', self.betag_breitung)
        print('Probability of failure:        ', self.pf2_breitung)
        print('')
        for i,k in enumerate(self.kappa):
            print(f"Curavture {i+1}: {k}")
        print("=" * n_hyphen)
        print('')

    def showDetailedOutput(self):
        """Get detailed output to console"""
        names = self.form.model.getNames()
        u_star = self.form.getDesignPoint()
        x_star = self.form.getDesignPoint(uspace=False)
        alpha = self.form.getAlpha()
        betaHL = self.form.beta[0]
        pfFORM = self.form.Pf

        n_hyphen = 54
        print('')
        print("=" * n_hyphen)
        print("FORM/SORM")
        print("=" * n_hyphen)
        print("{:15s} \t\t {:1.10e}".format("Pf FORM",pfFORM))
        print("{:15s} \t\t {:1.10e}".format("Pf SORM Breitung",self.pf2_breitung))
        print("{:15s} \t {:1.10e}".format("Pf SORM Breitung HR",self.pf2_breitung_m))
        print("{:15s} \t\t {:2.10f}".format("Beta_HL",betaHL))
        print("{:15s} \t\t {:2.10f}".format("Beta_G Breitung",self.betag_breitung))
        print("{:15s} \t\t {:2.10f}".format("Beta_G Breitung HR",self.betag_breitung_m))
        print("{:15s} \t\t {:d}".format('Model Evaluations',
                                      self.model.getCallFunction()))

        print("-" * n_hyphen)
        for i,k in enumerate(self.kappa):
            print(f"Curvature {i+1}: {k}")

        print("-" * n_hyphen)
        print("{:10s} \t {:>9s} \t {:>12s} \t {:>9s}".format("Variable",
                                                            'U_star',
                                                            'X_star',
                                                            'alpha'))
        for i, name in enumerate(names):
            print("{:10s} \t {: 5.6f} \t {:12.6f} \t {:+5.6f}".format(name,
                                                                   u_star[i],
                                                                   x_star[i],
                                                                   alpha[i]))
        print("=" * n_hyphen)
        print('')

    def computeHessian(self, diff_type=None):
        """
        Computes the matrix of second derivatives using forward finite
        difference, using the evaluation of the gradient already done
        for FORM, at the design point

        Could use numdifftools as external library instead

        """

        h = 1/self.options.ffdpara
        nrv = self.form.alpha.shape[1]
        hess_G = np.zeros((nrv,nrv))

        if diff_type is None:
            # Differentiation based on the gradients
            x0 = self.form.getDesignPoint(uspace=False)
            _, grad_g0 = self.evaluateLSF(x0[:, np.newaxis],
                                          calc_gradient=True)
            u0 = self.form.getDesignPoint()
            for i in range(nrv):
                u1 = np.copy(u0)
                u1[i] += h
                x1 = u_to_x(u1,self.model)
                _, grad_g1 = self.evaluateLSF(x1[:, np.newaxis],
                                              calc_gradient=True)
                hess_G[:,i] = ((grad_g1 - grad_g0)/h).reshape(nrv)

        else:
            # FERUM-implementation using a mix of central and foward diffs
            # It would be good if u_to_x could take an nvr x nx matrix and
            # return the corresponding x-matrix. This would make it easier to
            # add more numerical differentiation schemes.
            u0 = self.form.getDesignPoint()

            all_x_plus = np.zeros((nrv,nrv))
            all_x_minus = np.zeros((nrv,nrv))
            all_x_both = np.zeros((nrv,int(nrv*(nrv-1)/2)))

            for i in range(nrv):
                # Plus perturbation and transformation
                u_plus = np.copy(u0)
                u_plus[i] += h
                x_plus = u_to_x(u_plus, self.model)
                all_x_plus[:,i] = x_plus

                # Minus perturbation and transformation
                u_minus = np.copy(u0)
                u_minus[i] -= h
                x_minus = u_to_x(u_minus,self.model)
                all_x_minus[:,i] = x_minus

                for j in range(i):
                    # Mixed perturbation and transformation
                    u_both = np.copy(u_plus)
                    u_both[j] += h
                    x_both = u_to_x(u_both,self.model)
                    all_x_both[:,int((i-1)*(i)/2)+j] = x_both

            # Assemble all x-space vecs, solve for G, then separate
            all_x = np.concatenate((all_x_plus, all_x_minus, all_x_both), axis=1)
            all_G, _ = self.evaluateLSF(all_x, calc_gradient=False)
            all_G = all_G.squeeze()
            all_G_plus = all_G[:nrv]
            all_G_minus = all_G[nrv:2*nrv]
            all_G_both = all_G[2*nrv::]
            G = self.form.G

            # Now use finite differences to estimate hessian
            for i in range(nrv):
                # Second-order central difference
                hess_G[i,i] = (all_G_plus[i] - 2*G + all_G_minus[i])/h**2
                for j in range(i):
                    # Second order forward difference
                    hess_G[i,j] = (all_G_both[int((i-1)*(i)/2) + j] - all_G_plus[j]
                                   - all_G_plus[i] + G)/h**2
                    hess_G[j,i] = hess_G[i,j]

        return hess_G

    def evaluateLSF(self, x, calc_gradient=False, u_space=True):
        """
        For use in computing the Hessian without altering the FORM object.
        Considers the coord transform so the limit state function is evaluated
        in physical coordinates, but gradient returned in u-space.

        This code already in FORM, and a more integrated approach would put
        this in a base class for common use.

        """
        G, grad = 0, 0
        x0 = np.copy(x)  # avoid modifying argument in func calls below

        if calc_gradient:
            G, grad = evaluateLimitState(x0, self.model, self.options,
                                         self.limitstate)
            grad = np.transpose(grad)
            if u_space:
                u = x_to_u(x0, self.model)
                J_u_x = jacobian(u, x0, self.model)
                J_x_u = np.linalg.inv(J_u_x)
                grad = np.dot(grad, J_x_u)
        else:
            names = self.model.getNames()
            expression = self.limitstate.getExpression()
            G, _ = computeLimitStateFunction(x0, names, expression)

        return G, grad

    def orthonormal_matrix(self):
        """
        Computes the rotation matrix of the standard normal coordinate
        space where the design point is located at Beta along the last
        axis.
        """
        alpha = self.form.alpha.squeeze()
        nrv = len(alpha)
        # Assemble A per theory
        A = np.eye(nrv)
        A[-1, :] = alpha
        # Now rotate 270 degrees anticlockwise as MGS expects the reference
        # vector in the first column
        A = np.rot90(A, k=3)
        # Orthoganalize
        Q = self.gram_schmidt(A)
        # And undo this rotation with a final 90 dgree rotation as order of
        # the column vector entries is not relevant
        R1 = np.rot90(Q)
        return R1

    def gram_schmidt(self, A):
        """
        Creates an orthonormal matrix using the modified Gram-Schmidt process.
        Note that QR decomposition doesn't work for this application; while
        it does return an orthonormal matrix, the signs are different
        to the modified Gram Schmidt. The signs should be arbitrary, but the
        resulting rotation matrix does care cabout the signs of the Q, since
        it is based on the correct direction of the beta vector [Madsen1986]
        """

        A = np.asfarray(A)
        nvr = A.shape[0]
        Q = np.zeros(A.shape)
        for j in range(nvr):
            q = A[:, j]
            for i in range(j):
                rij = np.dot(q, Q[:, i])
                q = q - rij*Q[:, i]
            rjj = np.linalg.norm(q, ord=2)
            if np.isclose(rjj, 0.0):
                raise ValueError("Singular rotation matrix")
            else:
                Q[:, j] = q/rjj
        return Q
