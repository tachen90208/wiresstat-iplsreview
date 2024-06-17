"""ISIMPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.linalg import pinv
from pytictoc import TicToc

def _CentralizedData(x):
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    return xc, x_mean

def _ScatterMats(X,Y):
    N = X.shape[0]
    Xc, x_mean = _CentralizedData(X)
    Yc, y_mean = _CentralizedData(Y)
    C = Xc.T @ Xc / N
    S = Xc.T @ Yc
    return N, C, S, x_mean, y_mean

class ISIMPLS(BaseEstimator):
    """ISIMPLS.

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
        self.__name__ = ' (ISIMPLS)'
        self.n_components = n_components
        self.n = 0
        self._x_mean = 0
        self._y_mean = 0
        self.copy = copy
        self.W = None
        self.C = None
        self.S = None


    def _PLS1_ProjMat(self,S):
        W_rotations = np.zeros((self.n_components,self.n_features))
        V = np.zeros((self.n_components,self.n_features))
        # compute the project matrix
        W_rotations[0,:] = S/norm(S)
        p = self.C @ W_rotations[0,:]
        V[0,:] = p/norm(p)
        S = S - V[0,:]*np.dot(V[0,:],S)
        for k in range(1,self.n_components):
            W_rotations[k,:] = S/norm(S)
            p = self.C @ W_rotations[k,:]
            V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
            V[k,:] = V[k,:]/norm(V[k,:])
            S = S - V[k,:]*np.dot(V[k,:],S)
        return W_rotations.T

    def fit(self,X,y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        y = check_array(y.reshape(-1,1), dtype=FLOAT_DTYPES, copy=self.copy)

        if self.n == 0:
            self.n_features = X.shape[1]
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features,1))

        n_new = X.shape[0]
        _x_mean2 = X.mean(axis=0)
        X -= _x_mean2
        _y_mean2 = y.mean(axis=0)
        y -= _y_mean2

        n_old = self.n
        self.n += n_new
        f1,f2 = n_old/self.n, n_new/self.n
        # update the scatter matrices
        diff_x_mean = self._x_mean - _x_mean2
        diff_y_mean = self._y_mean - _y_mean2

        self.S += X.T @ y
        self.S += (f1*f2*self.n)*np.outer(diff_x_mean, diff_y_mean)

        self.C += np.dot(X.T, X)
        self.C += (f1*f2*self.n)*np.outer(diff_x_mean, diff_x_mean)

        # update the mean
        self._x_mean = f1*self._x_mean + f2*_x_mean2
        self._y_mean = f1*self._y_mean + f2*_y_mean2
        # computer the new project matrix
        self.W = self._PLS1_ProjMat(self.S.ravel().copy())
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X -= self._x_mean
        return X @ self.x_rotations_

    def _comp_coef(self):

        self.x_rotations_ = np.dot(
            self.W,
            pinv(np.dot(self.W.T,
                         np.dot(self.C, self.W)), check_finite=False),
        )

        self.y_loadings_ = np.dot(self.S.T, self.W)

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)

        self.intercept_ = self._y_mean
        return self

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        X -= self._x_mean
        ypred = X @ self.coef_
        ypred += self.intercept_
        return ypred

class ISIMPLS2(BaseEstimator):
    """ISIMPLS.

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
    def __init__(self, n_components=10, n=0, x_mean=0, y_mean=0,
                 W=None, C=None, S=None,
                 copy=True):
        self.__name__ = ' (ISIMPLS)'
        self.n_components = n_components
        self.n = n
        self._x_mean = x_mean
        self._y_mean = y_mean
        self.copy = copy
        self.W = W
        self.C = C
        self.S = S

    def __reduce__(self):
        return (self.__class__, (self.n_components,
                                 self.n, self._x_mean, self._y_mean,
                                 self.W, self.C, self.S))

    def _PLS2_ProjMat(self,S):
        W_rotations = np.zeros((self.n_components,self.n_features))
        V = np.zeros((self.n_components,self.n_features))
        # compute the project matrix
        wrk,_,_ = svds(S, k=1, return_singular_vectors="u")

        W_rotations[0,:] = wrk.ravel()
        p = self.C @ W_rotations[0,:]
        V[0,:] = p/norm(p)
        S -= np.outer(V[0,:], (V[0,:].T @ S))
        for k in range(1,self.n_components):
            wrk,_,_ = svds(S, k= 1, return_singular_vectors="u")
            W_rotations[k,:] = wrk.ravel()
            p = self.C @ W_rotations[k,:]
            V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
            V[k,:] /= norm(V[k,:])
            S -= np.outer(V[k,:], (V[k,:].T @ S))
        return W_rotations.T

    def fit(self,X,Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy)

        if self.n==0:
            self.n_features = X.shape[1]
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features, self.n_features))

        n_new = X.shape[0]
        _x_mean2 = X.mean(axis=0)
        X -= _x_mean2
        _y_mean2 = Y.mean(axis=0)
        Y -= _y_mean2

        n_old = self.n
        self.n += n_new
        f1,f2 = n_old/self.n, n_new/self.n
        # update the scatter matrices
        diff_x_mean = self._x_mean - _x_mean2
        diff_y_mean = self._y_mean - _y_mean2

        self.S += np.dot(X.T, Y)
        self.S += (f1*f2*self.n)*np.outer(diff_x_mean, diff_y_mean)

        self.C += np.dot(X.T, X)
        self.C += (f1*f2*self.n)*np.outer(diff_x_mean, diff_x_mean)

        # update the mean
        self._x_mean = f1*self._x_mean + f2*_x_mean2
        self._y_mean = f1*self._y_mean + f2*_y_mean2
        # computer the new project matrix
        self.W = self._PLS2_ProjMat(self.S.copy())
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X -= self._x_mean
        return X @ self.x_rotations_

    def _comp_coef(self,n_components=0):
        if n_components == 0:
            n_components = self.n_components
        n_components = min(n_components, self.n_components)

        W = self.W[:,:n_components]
        self.x_rotations_ = np.dot(
            W,
            pinv(np.dot(W.T,
                         np.dot(self.C, W)), check_finite=False),
        )

        self.y_loadings_ = np.dot(self.S.T, W)

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)

        self.intercept_ = self._y_mean
        return self

    def predict(self, X, n_components=0, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        if n_components == 0:
            n_components = self.n_components
        n_components = min(n_components, self.n_components)

        self._comp_coef(n_components)
        X -= self._x_mean
        ypred = X @ self.coef_
        ypred += self.intercept_
        return ypred
