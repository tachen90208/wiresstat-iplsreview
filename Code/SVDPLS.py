"""SVDPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSSVD
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.sparse.linalg import svds
from scipy.linalg import pinv

def _CentralizedData(x):
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    return xc, x_mean

class SVDPLS(PLSSVD):
    def _comp_coef(self, n_components):
        if n_components == 0:
            n_components = self.n_components
        n_components = min(n_components, self.n_components)

        x_weights = self.x_weights_[:,:n_components]
        y_weights = self.y_weights_[:,:n_components]

        self.coef_ = np.dot(x_weights, y_weights.T)
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

class ISVDPLS(BaseEstimator):
    """ISVDPLS.

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

    def __init__(self, n_components=10, copy=True):
        self.__name__ = ' (ISVDPLS)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self._x_mean = 0
        self._y_mean = 0
        self.H = 0

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        self.n_samples = X.shape[0]
        if self.n == 0:
            self.n += X.shape[0]
            self.n_features = X.shape[1]
            self.n_labels = Y.shape[1]
            self.H = min(self.n_components,
                         min(self.n_features, self.n_labels))
            self.x_weights_ = np.zeros((self.n_features, self.n_components))
            self.y_weights_ = np.zeros((self.n_labels, self.n_components))

            W_rotations = np.zeros((self.n_components, self.n_features))

            Xc, self._x_mean = _CentralizedData(X)
            Yc, self._y_mean = _CentralizedData(Y)
            self.S = Xc.T @ Yc

            # << SVD >>
            self.U,self.Del,self.Vh = svds(self.S, k=self.H)
            u  = self.U[:,0].ravel()
            vh = self.Vh[0,:]
            # << Adjusted weights step >>
            W_rotations[0,:] = u
            # << Deflation step >>
            delta = u.T @ self.S @ vh.T
            self.S -= delta*np.outer(u,vh)

            for k in range(1,min(self.n,self.n_components)):
                # << SVD >>
                u,_,vh = svds(self.S, k= 1)
                u = u.ravel()
                # << Adjusted weights step >>
                W_rotations[k,:] = u
                # << Deflation step >>
                delta = u.T @ self.S @ vh.T
                self.y_weights_[:,k] = vh.ravel()
                self.S -= delta*np.outer(u,vh)

            self.x_weights_ = W_rotations.T
        else:
            for j in range(X.shape[0]):
                self.n += 1
                x = X[j]
                y = Y[j]

                self._x_mean *= self.n-1
                self._x_mean += x
                self._x_mean /= self.n
                x -= self._x_mean

                self._y_mean *= self.n-1
                self._y_mean += y
                self._y_mean /= self.n
                y -= self._y_mean

                c = self.U.T @ x
                x_perp = x - self.U @ c
                x_perp_norm = norm(x_perp)
                d = self.Vh @ y
                y_perp = y - self.Vh.T @ d
                y_perp_norm = norm(y_perp)

                wrk = self.n/(self.n-1)*(np.diag(self.Del)+np.outer(c,d))
                Q = np.block([
                    [wrk, y_perp_norm*c.reshape((-1,1))],
                    [x_perp_norm*d, x_perp_norm*y_perp_norm]
                ])
                Q *= (self.n-1)/self.n

                x_perp /= x_perp_norm
                y_perp /= y_perp_norm

                A,self.Del,Bh = svds(Q,k=self.H)
                self.U  = np.block([self.U, x_perp.reshape((-1,1))]) @ A
                self.Vh = Bh @ np.block([[self.Vh], [y_perp]])

            self.x_weights_ = self.U
            self.y_weights_ = self.Vh.T
        return self

    def tranform(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X -= self.x_mean
        return np.dot(X, self.x_weights_)

    def _comp_coef(self, n_components):
        x_weights = self.x_weights_[:,:n_components]
        y_weights = self.y_weights_[:,:n_components]

        self.coef_ = np.dot(x_weights, y_weights.T)
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
