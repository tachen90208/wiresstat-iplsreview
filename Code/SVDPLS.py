"""SVDPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.sparse.linalg import svds


class SVDPLS(BaseEstimator):
    """SVDPLS.

    Parameters
    ----------
    n_components : int or None, (default=10)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References: PLS for Big Data: A unified parallel algorithm for
        regularised group PLS

    """
    def CentralizedData(self,X):
        if (len(X.shape) == 1):
            Xc = X - np.mean(X)
        else:
            X_mean = np.mean(X,axis=0)
            Xc = np.zeros(X.shape)
            for i in np.arange(X.shape[0]):
                Xc[i,:] = X[i,:] - X_mean
        return Xc

    def __init__(self, n_components=10, copy=True):
        self.__name__ = ' (SVDPLS)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.W = None
        # self.theta_x = theta_x
        # self.theta_y = theta_y

    # def soft_x(self,v,S):
    #     wrk = S @ v
    #     u = np.maximum(np.abs(wrk)-self.theta_x, np.zeros((self.n_features,)))
    #     return u*np.sign(wrk)

    # def soft_y(self,u,S):
    #     wrk = S.T @ u
    #     v = np.maximum(np.abs(wrk)-self.theta_y, np.zeros((self.n_labels,)))
    #     return v*np.sign(wrk)

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n == 0:
            self.n_samples, self.n_features = X.shape
            self.n = self.n_samples

            if (len(Y.shape) > 1):
                self.n_labels = Y.shape[1]
            else:
                self.n_labels = 1
            # << Initialisation >>
            X = self.CentralizedData(X)
            Y = self.CentralizedData(Y)
            W_rotations = np.zeros((self.n_components, self.n_features))

        S = X.T @ Y
        for k in range(self.n_components):
            # << SVD >>
            u,_,vh = svds(S, k= 1)
            u = u.ravel()
            v = vh.ravel()
            # # << Sparsity step >>
            # if (self.theta_x > 0):
            #     u_old = np.zeros((self.n_features,))
            #     while ( norm(u-u_old)/norm(u) > 1e-6):
            #         wrk_u = self.soft_x(v,S)
            #         u_old = u
            #         u = wrk_u/norm(wrk_u)
            #         wrk_v = self.soft_y(u,S)
            #         v = wrk_v/norm(wrk_v)

            # << Adjusted weights step >>
            W_rotations[k,:] = u.ravel()
            # << Deflation step >>
            # ========================================
            # x_scores = X @ u
            # y_scores = Y @ v
            # X = X - np.outer(x_scores, u)
            # Y = Y - np.outer(y_scores, v)
            # S = X.T @ Y
            # ========================================
            delta = u.T @ S @ v
            S -= delta*np.outer(u,v)
            # ========================================
        self.W = W_rotations.T
        return self

    def PLSR_coef(self,X,Y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X = self.CentralizedData(X)
        Y = self.CentralizedData(Y)

        T  = X @ self.W  # score matrix of Xc
        Qt = T.T @ Y     # transpose of loading matrix of Yc
        return self.W @ inv(T.T @ T) @ Qt
