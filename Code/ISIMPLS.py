"""ISIMPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from pytictoc import TicToc

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
        self.mu_x = 0
        self.mu_y = 0
        self.copy = copy
        self.W = None
        self.C = None
        self.S = None

    def _PLS1_ScatterMats(self,X,y):
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
        # y = check_array(y, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n==0:
            self.n_features = X.shape[1]
            self.C = np.zeros((self.n_features, self.n_features))
            self.S = np.zeros((self.n_features,))

        n_new,C2,S2,mu_x2,mu_y2 = self._PLS1_ScatterMats(X,y)
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
        # computer the new project matrxi
        self.W = self._PLS1_ProjMat(self.S)
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X -= self.mu_x
        return X @ self.W

    def PLSR_coef(self,X,y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # y = check_array(y, copy=copy, dtype=FLOAT_DTYPES)

        for i in np.arange(X.shape[0]):
            X[i,:] -= self.mu_x
        yc = y - np.mean(y)

        T    = X @ self.W   # score matrix of Xc
        Qt   = T.T @ yc  # transpose of loading matrix of yc
        return self.W  @ inv( T.T @ T ) @ Qt

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
        self.mu_x = 0
        self.mu_y = 0
        self.copy = copy
        self.W = None
        self.C = None
        self.S = None
        self.tic_svd = TicToc()
        self.tic_upd = TicToc()
        self.tim_svd = 0
        self.tim_upd = 0

    def _PLS2_ScatterMats(self,X,Y):
        N = X.shape[0]
        mu_x = np.mean(X,axis=0)
        Xc = np.zeros(X.shape)
        for i in np.arange(N):
            Xc[i,:] = X[i,:] - mu_x

        mu_y = np.mean(Y,axis=0)
        Yc = np.zeros(Y.shape)
        for i in np.arange(N):
            Yc[i,:] = Y[i,:] - mu_y

        C = Xc.T @ Xc/N
        S = Xc.T @ Yc
        return N,C,S,mu_x,mu_y

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
        n_new,C2,S2,mu_x2,mu_y2 = self._PLS2_ScatterMats(X,Y)
        n_old = self.n
        self.n += n_new
        f1,f2 = n_old/self.n, n_new/self.n
        # update the scatter matrices
        diff_mu_x = self.mu_x - mu_x2
        diff_mu_y = self.mu_y - mu_y2

        self.S += S2 + (f1*f2*self.n)*np.outer(diff_mu_x, diff_mu_y)
        # self.S += (f1*f2*self.n)*np.outer(diff_mu_x, diff_mu_y)

        self.C *= f1
        self.C += f2*C2 + f1*f2*np.outer(diff_mu_x, diff_mu_x)
        # self.C += f1*f2*np.outer(diff_mu_x, diff_mu_x)

        # update the mean
        self.mu_x = f1*self.mu_x + f2*mu_x2
        self.mu_y = f1*self.mu_y + f2*mu_y2
        # self.tim_upd += self.tic_upd.tocvalue()
        # computer the new project matrxi
        self.W = self._PLS2_ProjMat(self.S.copy())
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X -= self.mu_x
        return X @ self.W

    def PLSR_coef(self,X,Y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # y = check_array(y, copy=copy, dtype=FLOAT_DTYPES)
        N = X.shape[0]
        for i in np.arange(N):
            X[i,:] -= self.mu_x

        Yc = np.zeros(Y.shape)
        for i in np.arange(N):
            Yc[i,:] = Y[i,:] - self.mu_y

        T    = X @ self.W   # score matrix of Xc
        Qt   = T.T @ Yc  # transpose of loading matrix of yc
        return self.W  @ inv( T.T @ T ) @ Qt
