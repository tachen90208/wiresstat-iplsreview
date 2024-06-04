"""Covariance-free Partial Least Squares"""

# Author: Artur Jordao <arturjlcorreia[at]gmail.com>
#         Artur Jordao

import numpy as np
from scipy import linalg

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from scipy.linalg import pinv
from sklearn.metrics import r2_score
import copy

def _CentralizedData(x):
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    return xc, x_mean

class CIPLS(BaseEstimator):
    """Covariance-free Partial Least Squares (CIPLS).

    Parameters
    ----------
    n_components : int or None, (default=None)
        Number of components to keep. If ``n_components `` is ``None``,
        then ``n_components`` is set to ``min(n_samples, n_features)``.

    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    References
    Covariance-free Partial Least Squares: An Incremental Dimensionality Reduction Method
    """

    def __init__(self, n_components=10, copy=True):
        self.__name__ = 'Covariance-free Partial Least Squares'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.sum_x = None
        self.sum_y = None
        self.n_features = None
        self.x_weights_ = None
        self.x_loadings_ = None
        self.y_loadings_ = None
        self.eign_values = None
        self._x_mean = None
        self.p = []

    def normalize(self, x):
        return normalize(x[:, np.newaxis], axis=0).ravel()

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        Y = check_array(Y, dtype=FLOAT_DTYPES, copy=self.copy, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if np.unique(Y).shape[0] == 2:
            Y[np.where(Y == 0)[0]] = -1

        n_samples, n_features = X.shape

        if self.n == 0:
            x_weights_ = np.zeros((self.n_components, n_features))
            self.x_loadings_ = np.zeros((n_features, self.n_components))
            self.y_loadings_ = np.zeros((Y.shape[1], self.n_components))
            self.n_features = n_features
            self.eign_values = np.zeros((self.n_components))
            self.p = [0] * self.n_components

        for j in range(0, n_samples):
            self.n = self.n + 1
            u = X[j]
            l = Y[j]

            if self.n == 1:
                self.sum_x = u
                self.sum_y = l
            else:
                old_mean = 1 / (self.n - 1) * self.sum_x
                self.sum_x = self.sum_x + u
                mean_x = 1 / self.n * self.sum_x
                u = u - mean_x
                delta_x = mean_x - old_mean
                x_weights_[0] = x_weights_[0] - delta_x * self.sum_y
                x_weights_[0] = x_weights_[0] + (u * l)
                self.sum_y = self.sum_y + l

                t = np.dot(u, self.normalize(x_weights_[0].T))
                self.x_loadings_[:, 0] += (u * t)
                self.y_loadings_[:, 0] += (l * t)

                for c in range(1, self.n_components):
                    u -= np.dot(t, self.x_loadings_[:, c - 1])
                    l -= np.dot(t, self.y_loadings_[:, c - 1])
                    x_weights_[c]   += (u * l)
                    self.x_loadings_[:, c] += (u * t)
                    self.y_loadings_[:, c] += (l * t)
                    t = np.dot(u, self.normalize(x_weights_[c].T))

            self.x_weights_ = x_weights_.T
        return self

    def transform_orig(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data."""
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        mean = 1 / self.n * self.sum_x
        X -= mean
        self.w_rotation = np.zeros(self.x_rotations.shape)

        for c in range(0, self.n_components):
            self.w_rotation[c] = self.normalize(self.x_rotations[c])

        return np.dot(X, self.w_rotation.T)

    def _comp_coef(self):
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False),
        )
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self._x_mean = self.sum_x / self.n
        self._y_mean = self.sum_y / self.n
        self.intercept_ = self._y_mean
        return self

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        X -= self._x_mean
        ypred = X @ self.coef_
        ypred += self.intercept_
        return ypred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
