"""SIMPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from pytictoc import TicToc


class SIMPLS(BaseEstimator):
    """SIMPLS.

    Parameters
    ----------
    n_components : int or None, (default=10)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References:

    """
    def __init__(self, n_components=10, copy=True):
        self.__name__ = ' (SIMPLS)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.W = None
        self.tic_svd = TicToc()
        self.tic_upd = TicToc()
        self.tim_svd = 0
        self.tim_upd = 0

    def normalize(self, x):
        return normalize(x[:, np.newaxis], axis=0).ravel()

    def CentralizedData(self,X,Y):
        # y n*m, ave 1*m
        # X n*p, ave 1*p
        self.mu_x = np.mean(X,axis=0)
        Xc = np.zeros(X.shape)
        for i in np.arange(X.shape[0]):
            Xc[i,:] = X[i,:] - self.mu_x

        if (len(Y.shape) == 1):
            Yc = Y - np.mean(Y)
        else:
            Y_mean = np.mean(Y,axis=0)
            Yc = np.zeros(Y.shape)
            for i in np.arange(Y.shape[0]):
                Yc[i,:] = Y[i,:] - Y_mean
        return Xc,Yc

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        # Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy)
        n_samples, n_features = X.shape
        if self.n == 0:
            self.n_features = n_features
            self.x_rotations = np.zeros((n_features, self.n_components))

        W_rotations = np.zeros((self.n_components,n_features))
        V = np.zeros((self.n_components,n_features))

        # self.tic_upd.tic()
        Xc,Yc = self.CentralizedData(X,Y)
        S = Xc.T @ Yc
        # self.tim_upd += self.tic_upd.tocvalue()
        if (len(Y.shape) == 1):
            W_rotations[0,:] = S/norm(S)
            t = Xc @ W_rotations[0,:]
            t /= norm(t)
            p = Xc.T @ t
            V[0,:] = p/norm(p)
            S -= V[0,:]*np.dot(V[0,:],S)
            for k in range(1,self.n_components):
                W_rotations[k,:] = S/norm(S)
                t = Xc @ W_rotations[k,:]
                t /= norm(t)
                p = Xc.T @ t # p
                V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
                V[k,:] /= norm(V[k,:])
                S -= V[k,:]*np.dot(V[k,:],S)
        else:
            # self.tic_svd.tic()
            wrk,_,_ = svds(S, k= 1, return_singular_vectors="u")
            # self.tim_svd += self.tic_svd.tocvalue()

            W_rotations[0,:] = wrk.ravel()
            t = Xc @ W_rotations[0,:]
            t /= norm(t)
            p = Xc.T @ t
            V[0,:] = p/norm(p)
            S -= np.outer(V[0,:], (V[0,:].T @ S))
            for k in np.arange(1,self.n_components):
                # self.tic_svd.tic()
                wrk,_,_ = svds(S, k= 1, return_singular_vectors="u")
                # self.tim_svd += self.tic_svd.tocvalue()
                W_rotations[k,:] = wrk.ravel()
                t = Xc @ W_rotations[k,:]
                t /= norm(t)
                p = Xc.T @ t
                V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
                V[k,:] /= norm(V[k,:])
                S -= np.outer(V[k,:], (V[k,:].T @ S))
        self.W = W_rotations.T
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -= self.mu_x
        return np.dot(X, self.W)
    def PLSR_coef(self,X,Y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Y = check_array(y, copy=copy, dtype=FLOAT_DTYPES)

        X_mean = np.mean(X,axis=0)
        for i in np.arange(X.shape[0]):
            X[i,:] -= X_mean

        Yc = np.zeros(Y.shape)
        if (len(Y.shape) == 1):
            Yc = Y - np.mean(Y)
        else:
            Y_mean = np.mean(Y,axis=0)
            for i in np.arange(Y.shape[0]):
                Yc[i,:] = Y[i,:] - Y_mean

        T    = X @ self.W   # score matrix of Xc
        Qt   = T.T @ Yc  # transpose of loading matrix of yc
        return  self.W  @ inv( T.T @ T ) @ Qt
