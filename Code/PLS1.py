"""IPLS1."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

class PLS1(BaseEstimator):
    """PLS1(close form).

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
        self.__name__ = ' (IPLS1)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy

    def CentralizedData(self,X,y):
        # y n*m, ave 1*m
        # X n*p, ave 1*p
        X_mean = np.mean(X,axis=0)
        Xc = np.zeros(X.shape)
        for i in np.arange(X.shape[0]):
            Xc[i,:] = X[i,:] - X_mean

        yc = y - np.mean(y)
        return Xc,yc

    def _Arnoldi(self,A,v):
        W_rotations = np.zeros((self.n_components, self.n_features))
        W_rotations[0,:] = v/norm(v)
        for i in np.arange(1,self.n_components):
            W_rotations[i,:] = A @ W_rotations[i-1,:]
            W_rotations[i,:] = W_rotations[i,:] - \
                W_rotations[:i,:].T @ (W_rotations[:i,:] @ W_rotations[i,:])
            W_rotations[i,:] = W_rotations[i,:]/norm(W_rotations[i,:])
        return W_rotations.T

    def fit(self, X, y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        # Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n == 0:
            self.n, self.n_features = X.shape
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features,))

        Xc,yc = self.CentralizedData(X,y)
        self.C = Xc.T @ Xc/self.n
        self.S = Xc.T @ yc
        self.W = self._Arnoldi(self.C,self.S)
        return self

    def transform(self, X, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -=  np.mean(X,axis=0)
        return np.dot(X, self.W)

    def PLSR_coef(self,X,y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X_mean = np.mean(X,axis=0)
        for i in np.arange(X.shape[0]):
            X[i,:] -= X_mean

        yc = y-np.mean(y)

        T    = X @ self.W   # score matrix of Xc
        Qt   = T.T @ yc  # transpose of loading matrix of yc
        return  self.W  @ inv( T.T @ T ) @ Qt


class IPLS1(BaseEstimator):
    """IPLS1(close form).

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
        self.__name__ = ' (IPLS1)'
        self.n_components = n_components
        self.n = 0
        self.mu_x = 0
        self.mu_y = 0
        self.copy = copy

    def CentralizedData(self,X,y):
        # y n*m, ave 1*m
        # X n*p, ave 1*p
        X_mean = np.mean(X,axis=0)
        Xc = np.zeros(X.shape)
        for i in np.arange(X.shape[0]):
            Xc[i,:] = X[i,:] - X_mean

        yc = y - np.mean(y)
        return Xc,yc

    def _Comp_ScatterMats(self,X,y):
        N = X.shape[0]
        mu_x = np.mean(X,axis=0)
        Xc = np.zeros(X.shape)
        for i in np.arange(N):
            Xc[i,:] = X[i,:] - mu_x
        mu_y = np.mean(y)
        yc = y - mu_y
        C = Xc.T @ Xc/N
        S = Xc.T @ yc
        return N,C,S,mu_x,mu_y

    def _Arnoldi(self,A,v):
        W_rotations = np.zeros((self.n_components, self.n_features))
        W_rotations[0,:] = v/norm(v)
        for i in np.arange(1,self.n_components):
            W_rotations[i,:] = A @ W_rotations[i-1,:]
            W_rotations[i,:] = W_rotations[i,:] - \
                W_rotations[:i,:].T @ (W_rotations[:i,:] @ W_rotations[i,:])
            W_rotations[i,:] = W_rotations[i,:]/norm(W_rotations[i,:])
        return W_rotations.T

    def fit(self, X, y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        # Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n == 0:
            n_samples, n_features = X.shape
            self.n_features = n_features
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features,))

        n_new, C2, S2, mu_x2, mu_y2 = \
            self._Comp_ScatterMats(X,y)

        # update the number of sample
        n_old = self.n
        self.n += n_new
        f1,f2 = n_old/self.n, n_new/self.n
        # update the scatter matrices
        diff_mu_x = self.mu_x - mu_x2
        diff_mu_y = self.mu_y - mu_y2
        self.C = f1*self.C + f2*C2 + f1*f2*np.outer(diff_mu_x, diff_mu_x)
        self.S = self.S + S2 + (f1*f2*self.n)*diff_mu_y*diff_mu_x
        # update the mean
        self.mu_x = f1*self.mu_x + f2*mu_x2
        self.mu_y = f1*self.mu_y + f2*mu_y2
        self.W = self._Arnoldi(self.C,self.S)
        return self

    def transform(self, X, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -= self.mu_x
        return np.dot(X, self.W)

    def PLSR_coef(self,X,y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X_mean = np.mean(X,axis=0)
        for i in np.arange(X.shape[0]):
            X[i,:] -= X_mean

        yc = y-np.mean(y)

        T    = X @ self.W   # score matrix of Xc
        Qt   = T.T @ yc  # transpose of loading matrix of yc
        return  self.W  @ inv( T.T @ T ) @ Qt
