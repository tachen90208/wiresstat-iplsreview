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
        self.x_loadings_[:,0] = p
        for k in range(1,self.n_components):
            W_rotations[k,:] = S/norm(S)
            p = self.C @ W_rotations[k,:]
            V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
            V[k,:] = V[k,:]/norm(V[k,:])
            S = S - V[k,:]*np.dot(V[k,:],S)
            self.x_loadings_[:,k] = p
        return W_rotations.T

    def fit(self,X,y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        # y = check_array(y, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n == 0:
            self.n_features = X.shape[1]
            self.x_loadings_ = np.zeros((self.n_features, self.n_components))
            self.y_loadings_ = np.zeros((1, self.n_components))
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features,))

        n_new,C2,S2,_x_mean2,_y_mean2 = _ScatterMats(X,y)
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
        # computer the new project matrxi
        self.W = self._PLS1_ProjMat(self.S.copy())
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
                         np.dot(self.n*self.C, self.W)), check_finite=False),
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
        self.tic_svd = TicToc()
        self.tic_upd = TicToc()
        self.tim_svd = 0
        self.tim_upd = 0

    def _PLS2_ProjMat(self,S):
        W_rotations = np.zeros((self.n_components,self.n_features))
        V = np.zeros((self.n_components,self.n_features))
        # compute the project matrix
        # self.tic_svd.tic()
        wrk,_,_ = svds(S, k=1, return_singular_vectors="u")
        # self.tim_svd += self.tic_svd.tocvalue()

        W_rotations[0,:] = wrk.ravel()
        p = self.C @ W_rotations[0,:]
        V[0,:] = p/norm(p)
        S -= np.outer(V[0,:], (V[0,:].T @ S))
        for k in range(1,self.n_components):
            # self.tic_svd.tic()
            wrk,_,_ = svds(S, k= 1, return_singular_vectors="u")
            # self.tim_svd += self.tic_svd.tocvalue()
            W_rotations[k,:] = wrk.ravel()
            p = self.C @ W_rotations[k,:]
            V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
            V[k,:] /= norm(V[k,:])
            S -= np.outer(V[k,:], (V[k,:].T @ S))
        return W_rotations.T

    def fit(self,X,Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        # y = check_array(y, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n==0:
            self.n_features = X.shape[1]
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features, self.n_features))

        # self.tic_upd.tic()
        n_new,C2,S2,_x_mean2,_y_mean2 = _ScatterMats(X,Y)
        n_old = self.n
        self.n += n_new
        f1,f2 = n_old/self.n, n_new/self.n
        # update the scatter matrices
        diff_x_mean = self._x_mean - _x_mean2
        diff_y_mean = self._y_mean - _y_mean2

        self.S += S2 + (f1*f2*self.n)*np.outer(diff_x_mean, diff_y_mean)
        # self.S += (f1*f2*self.n)*np.outer(diff_x_mean, diff_y_mean)

        self.C *= f1
        self.C += f2*C2 + f1*f2*np.outer(diff_x_mean, diff_x_mean)
        # self.C += f1*f2*np.outer(diff_x_mean, diff_x_mean)

        # update the mean
        self._x_mean = f1*self._x_mean + f2*_x_mean2
        self._y_mean = f1*self._y_mean + f2*_y_mean2
        # self.tim_upd += self.tic_upd.tocvalue()
        # computer the new project matrix
        self.W = self._PLS2_ProjMat(self.S.copy())
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
                         np.dot(self.n*self.C, self.W)), check_finite=False),
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
