"""IPLS1."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.linalg import pinv

def _CentralizedData(x):
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    return xc, x_mean

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

        Xc, self._x_mean = _CentralizedData(X)
        yc, self._y_mean = _CentralizedData(y)
        self.C = Xc.T @ Xc
        self.S = Xc.T @ yc
        self.W = self._Arnoldi(self.C/self.n,self.S)
        return self

    def transform(self, X, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -= self._x_mean
        return np.dot(X, self.W)

    def _comp_coef(self):


        self._y_loadings = np.dot(self.S.T, self.W)

        self.coef_ =  np.dot(
            self.W,
            pinv( np.dot(self.W.T, np.dot(self.C, self.W)))
        )
        self.coef_ = np.dot(self.coef_, self._y_loadings.T)
        self.intercept_ = self._y_mean
        return self

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        Xc,_ = _CentralizedData(X)
        ypred = Xc @ self.coef_
        ypred += self.intercept_
        return ypred

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
        self._x_mean = 0
        self._y_mean = 0
        self.copy = copy

    def _Comp_ScatterMats(self,X,y):
        N = X.shape[0]
        Xc, mu_x = _CentralizedData(X)
        yc, mu_y = _CentralizedData(y)
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

        n_new, C2, S2, _x_mean2, _y_mean2 = \
            self._Comp_ScatterMats(X,y)

        # update the number of sample
        n_old = self.n
        self.n += n_new
        f1,f2 = n_old/self.n, n_new/self.n
        # update the scatter matrices
        diff_x_mean = self._x_mean - _x_mean2
        diff_y_mean = self._y_mean - _y_mean2
        self.C = f1*self.C + f2*C2 + f1*f2*np.outer(diff_x_mean, diff_x_mean)
        self.S = self.S + S2 + (f1*f2*self.n)*diff_y_mean*diff_x_mean
        # update the mean
        self._x_mean = f1*self._x_mean + f2*_x_mean2
        self._y_mean = f1*self._y_mean + f2*_y_mean2
        self.W = self._Arnoldi(self.C,self.S)
        return self

    def transform(self, X, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -= self._x_mean
        return np.dot(X, self.W)

    def _comp_coef(self):
        self._y_loadings = np.dot(self.S.T, self.W)

        self.coef_ =  np.dot(
            self.W,
            pinv( np.dot(self.W.T, np.dot(self.C*self.n, self.W)))
        )
        self.coef_ = np.dot(self.coef_, self._y_loadings.T)
        self.intercept_ = self._y_mean
        return self

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        X -= self._x_mean
        ypred = X @ self.coef_
        ypred += self.intercept_
        return ypred
