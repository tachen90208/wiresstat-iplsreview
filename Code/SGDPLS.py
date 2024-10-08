"""Stochastic Partial Least Squares (SGDPLS)"""

import numpy as np
from numpy.linalg import norm
from scipy import linalg

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator
import sklearn.preprocessing as skpp
import copy

def gram_schmidt_normalize(X, copy=True):
    X = check_array(X, dtype=FLOAT_DTYPES, copy=copy, ensure_2d=True)
    Q, R = np.linalg.qr(X)
    return Q

class SGDPLS(BaseEstimator):
    """Stochastic Partial Least Squares (SGDPLS).

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    eta: int or None, (default=0.1)
        The learning rate to update the projection matrices after each iteration (on sample)

    epochs: int or None, (default=100)
        The number of epochs for learning the projection matrices

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    norm : str, (default='l2')
        Normalization functions. Available:
        - 'l2': L2-norm from scikit preprocessing
        - 'gs': Gram-Schmidt using QR decomposition
	    - 'gs_numpy': Gram-Schmidt numpy implementation

    References
    Stochastic optimization for multiview representation learning using partial least squares.
    """

    def __init__(self, n_components=10, eta=0.001, epochs=100, copy=True, **kwargs):
        self.__name__ = 'Stochastic Partial Least Squares (SGDPLS)'
        self.n_components = n_components
        self.n = 0
        self.eta = eta
        self.epochs=epochs
        self.copy = copy
        self.n_features = None
        self.x_rotations = None
        self.eign_values = None
        self.x_mean = None
        self.x_loadings = None
        self.y_loadings = None

        self.normalize = gram_schmidt_normalize
        self.norm_after_update = kwargs.get('norm_after_update', False)

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if np.unique(Y).shape[0] == 2:
            Y[np.where(Y == 0)[0]] = -1

        n_samples, n_features = X.shape

        if self.n == 0:
            self.v = np.zeros((self.n_components, n_features))
            self.n_features = n_features
            self.x_rotations = np.random.randn(n_features, self.n_components)
            self.y_rotations = np.random.randn(Y.shape[1], self.n_components)
            self.eign_values = np.zeros((self.n_components))

            self.x_loadings = np.zeros((n_features, self.n_components))
            self.y_loadings = np.zeros((Y.shape[1], self.n_components))
            self.x_rotations = self.normalize(self.x_rotations)

        for i in range(self.epochs):
            for xi, yi in zip(X, Y):
                self.n = self.n + 1

                xi = xi[:, np.newaxis]
                yi = yi[:, np.newaxis]

                tmp = np.copy(self.x_rotations)
                self.x_rotations += self.eta * np.matmul(np.matmul(xi, yi.T), self.y_rotations)
                self.y_rotations += self.eta * np.matmul(np.matmul(yi, xi.T), tmp)

        self._find_projections(X,Y)
        return self

    def _find_projections(self, X, Y):
        for c in range(1, self.n_components):
            for i in range(X.shape[0]):
                t = np.dot(X[i], gram_schmidt_normalize(self.x_rotations.T[c-1][:, np.newaxis]).ravel()  )

                self.x_loadings[:, c] = self.x_loadings[:, c] + X[i] * t
                self.y_loadings[:, c] = self.y_loadings[:, c] + Y[i] * t

                # X[i] -= np.dot(t, self.x_loadings[:, c])
                # self.x_rotations[:, c] = self.x_rotations[:, c] + (X[i] * Y[i])

                x_tmp = X[i] - np.dot(t, self.x_loadings[:, c])
                self.x_rotations[:, c] = self.x_rotations[:, c] + (x_tmp * Y[i])

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        self.U = self.normalize(self.x_rotations)

        return np.dot(X, self.U)

    def _comp_coef(self):
        self.U = self.normalize(self.x_rotations)
        self.coef_ = np.dot(self.U, self.y_loadings.T)

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        ypred = X @ self.coef_
        return ypred

    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
