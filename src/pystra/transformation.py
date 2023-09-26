# -*- coding: utf-8 -*-

import numpy as np


class Transformation:
    """
    Isoprobabilistic transformations
    """

    def __init__(self, transform_type=None):
        """
        Initialization of the Transformation class
        """
        self.transform_types = ["cholesky", "svd"]

        self.transform_type = transform_type

        if self.transform_type is None:
            self.transform_type = "cholesky"

        if self.transform_type not in self.transform_types:
            raise ValueError("Undefined transformation type")

        self.T = None
        self.inv_T = None

    def x_to_u(self, x, marg):
        """Transformation from x to u space"""
        nrv = len(marg)
        u = np.zeros(nrv)
        for i in range(nrv):
            u[i] = marg[i].x_to_u(x[i])

        u = np.dot(self.T, u)
        return u

    def u_to_x(self, u, marg):
        """Transformation from x to u space"""
        nrv = len(marg)
        z = np.dot(self.inv_T, u)

        x = np.zeros(nrv)
        for i in range(nrv):
            x[i] = marg[i].u_to_x(z[i])
        return x

    def jacobian(self, u, x, marg):
        """Jacobian for the transformation"""
        nrv = len(marg)
        u = np.dot(self.inv_T, u)
        J_u_x = np.zeros((nrv, nrv))

        for i in range(nrv):
            J_u_x[i][i] = marg[i].jacobian(u[i], x[i])

        J_u_x = np.dot(self.T, J_u_x)
        return J_u_x

    def compute(self, Ro):
        """
        Compute the Isoprobabilistic Transformation using the chosen method
        """
        if self.transform_type == self.transform_types[0]:
            self._computeCholesky(Ro)
        elif self.transform_type == self.transform_types[1]:
            self._computeSVD(Ro)
        else:
            raise ValueError("Transform type not set")

    def _computeCholesky(self, Ro):
        """Compute Cholesky Decomposition"""
        # Ro = self.model.getModifiedCorrelation()
        try:
            L = np.linalg.cholesky(Ro)
        except np.linalg.LinAlgError as e:
            print(f"Error: Cholesky decomposition: {e.message}")

        self.T = np.linalg.inv(L)
        self.inv_T = L

    def _computeSVD(self, Ro):
        """
        Singular Value Decomposition
        """
        try:
            U, D, V = np.linalg.svd(Ro)
        except np.linalg.LinAlgError as e:
            print(f"Error: singular value decomposition: {e.message}")

        sqrtD = np.sqrt(D) * np.eye(len(D))
        R = U @ sqrtD

        self.T = np.linalg.inv(R)
        self.inv_T = R
