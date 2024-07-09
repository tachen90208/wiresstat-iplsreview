"""SIMPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from pytictoc import TicToc
from scipy.linalg import pinv
from sklearn.metrics import r2_score

def _CentralizedData(x):
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    return xc, x_mean

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


   Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_targets, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_target, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    References:

    """
    def __init__(self, n_components=10, copy=True):
        self.__name__ = ' (SIMPLS)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.x_weights_ = None
        self._x_mean = 0
        self._y_mean = 0
        self.tic_svd = TicToc()
        self.tic_upd = TicToc()
        self.tim_svd = 0
        self.tim_upd = 0


    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        # Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy)
        n_samples, n_features = X.shape

        self.n_features = n_features
        self.x_rotations_ = np.zeros((n_features, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))

        # W_rotations = np.zeros((self.n_components,n_features))
        x_weights_ = np.zeros((self.n_components, n_features))

        if (len(Y.shape)==1):
            self.y_loadings_ = np.zeros((1, self.n_components))
        else:
            self.y_loadings_ = np.zeros((Y.shape[1], self.n_components))

        V = np.zeros((self.n_components,n_features))

        # self.tic_upd.tic()
        Xc,self._x_mean = _CentralizedData(X)
        Yc,self._y_mean = _CentralizedData(Y)
        self.S = Xc.T @ Yc
        S = self.S.copy()
        # self.tim_upd += self.tic_upd.tocvalue()
        if (len(Y.shape) == 1):
            x_weights_[0,:] = S/norm(S)

            t = Xc @ x_weights_[0,:]
            t /= norm(t)
            p = Xc.T @ t
            V[0,:] = p/norm(p)
            S -= V[0,:]*np.dot(V[0,:],S)
            self.x_loadings_[:,0] = p
            self.y_loadings_[:,0] = np.dot(Yc,t)

            for k in range(1,self.n_components):
                x_weights_[k,:] = S/norm(S)

                t = Xc @ x_weights_[k,:]
                t /= norm(t)
                p = Xc.T @ t # p
                V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
                V[k,:] /= norm(V[k,:])
                S -= V[k,:]*np.dot(V[k,:],S)
                self.x_loadings_[:,k] = p
                self.y_loadings_[:,k] = np.dot(Yc,t)

        else:
            # self.tic_svd.tic()
            wrk,_,_ = svds(S, k= 1, return_singular_vectors="u")
            # self.tim_svd += self.tic_svd.tocvalue()

            x_weights_[0,:] = wrk.ravel()
            t = Xc @ x_weights_[0,:]
            t /= norm(t)
            p = Xc.T @ t
            V[0,:] = p/norm(p)
            S -= np.outer(V[0,:], (V[0,:].T @ S))
            self.x_loadings_[:,0] = p
            self.y_loadings_[:,0] = Yc.T @ t

            for k in np.arange(1,self.n_components):
                # self.tic_svd.tic()
                wrk,_,_ = svds(S, k= 1, return_singular_vectors="u")
                # self.tim_svd += self.tic_svd.tocvalue()
                x_weights_[k,:] = wrk.ravel()
                t = Xc @ x_weights_[k,:]
                t /= norm(t)
                p = Xc.T @ t
                V[k,:] = p - V[:k,:].T @ (V[:k,:] @ p)
                V[k,:] /= norm(V[k,:])
                S -= np.outer(V[k,:], (V[k,:].T @ S))
                self.x_loadings_[:,k] = p
                self.y_loadings_[:,k] = Yc.T @ t

        self.x_weights_ = x_weights_.T

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        X -= self._x_mean
        return np.dot(X, self.x_rotations_)

    def _comp_coef(self, n_components):
        x_weights = self.x_weights_[:,:n_components]
        x_loadings = self.x_loadings_[:,:n_components]
        y_loadings = self.y_loadings_[:,:n_components]

        self.x_rotations_ = np.dot(
            x_weights,
            pinv(np.dot(x_loadings.T, x_weights), check_finite=False),
        )

        self.coef_ = np.dot(self.x_rotations_, y_loadings.T)
        self.intercept_ = self._y_mean

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

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
