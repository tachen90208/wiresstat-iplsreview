"""SVDPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.sparse.linalg import svds
from scipy.linalg import pinv

def _svd_flip_1d(u,v):
    """Same as svd_flip but works on 1d arrays, and is inplace"""
    # svd_flip would force us to convert to 2d array and would also return 2d
    # arrays. We don't want that.
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign

def _CentralizedData(x):
    x_mean = x.mean(axis=0)
    xc = x - x_mean
    return xc, x_mean

class SVDPLS(BaseEstimator):
    """SVDPLS.

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
    # def CentralizedData(self,X):
    #     if (len(X.shape) == 1):
    #         Xc = X - np.mean(X)
    #     else:
    #         X_mean = np.mean(X,axis=0)
    #         Xc = np.zeros(X.shape)
    #         for i in np.arange(X.shape[0]):
    #             Xc[i,:] = X[i,:] - X_mean
    #     return Xc

    def __init__(self, n_components=10, copy=True):
        self.__name__ = ' (SVDPLS)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.W = None
        # self.theta_x = theta_x
        # self.theta_y = theta_y

    # def soft_x(self,v,S):
    #     wrk = S @ v
    #     u = np.maximum(np.abs(wrk)-self.theta_x, np.zeros((self.n_features,)))
    #     return u*np.sign(wrk)

    # def soft_y(self,u,S):
    #     wrk = S.T @ u
    #     v = np.maximum(np.abs(wrk)-self.theta_y, np.zeros((self.n_labels,)))
    #     return v*np.sign(wrk)

    def fit(self, X, Y):
        X = check_array(X, dtype=FLOAT_DTYPES, copy=self.copy)
        if self.n == 0:
            self.n_samples, self.n_features = X.shape
            self.n_labels = Y.shape[1]

            self.n = self.n_samples
            self.x_loadings_ = np.zeros((self.n_features, self.n_components))
            self.y_loadings_ = np.zeros((self.n_labels, self.n_components))

            # << Initialisation >>
            Xc, self._x_mean = _CentralizedData(X)
            Yc, self._y_mean = _CentralizedData(Y)
            W_rotations = np.zeros((self.n_components, self.n_features))

        S = Xc.T @ Yc
        for k in range(self.n_components):
            # << SVD >>
            u,_,vh = svds(S, k=1)
            # _svd_flip_1d(u,vh)
            u = u.ravel()
            v = vh.ravel()
            # << Adjusted weights step >>
            W_rotations[k,:] = u.ravel()
            # << Deflation step >>
            # ========================================
            # x_scores = X @ u
            # y_scores = Y @ v
            # X = X - np.outer(x_scores, u)
            # Y = Y - np.outer(y_scores, v)
            # S = X.T @ Y
            # ========================================
            self.x_loadings_[:,k] = u
            self.y_loadings_[:,k] = v
            delta = u.T @ S @ v
            S -= delta*np.outer(u,v)
            # ========================================
        self.W = W_rotations.T
        return self

    def _comp_coef(self):

        self.x_rotations_ = np.dot(
            self.W,
            pinv(np.dot(self.x_loadings_.T, self.W), check_finite=False),
        )

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.intercept_ = self._y_mean

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        X -= self._x_mean
        ypred = X @ self.coef_
        ypred += self.intercept_
        return ypred

    # def PLSR_coef(self,X,Y,copy=True):
    #     X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
    #     X = self.CentralizedData(X)
    #     Y = self.CentralizedData(Y)

    #     T  = X @ self.W  # score matrix of Xc
    #     Qt = T.T @ Y     # transpose of loading matrix of Yc
    #     return self.W @ inv(T.T @ T) @ Qt

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
        self.W = None
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
            self.H = min(self.n_features, self.n_labels)
            self.H = min(self.n_components, self.H)
            self.x_loadings_ = np.zeros((self.n_features, self.n_components))
            self.y_loadings_ = np.zeros((self.n_labels, self.n_components))

            W_rotations = np.zeros((self.n_components, self.n_features))

            Xc, self._x_mean = _CentralizedData(X)
            Yc, self._y_mean = _CentralizedData(Y)
            self.S = Xc.T @ Yc

            # << SVD >>
            self.U,self.Del,self.Vh = svds(self.S,k=self.H)
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
                self.x_loadings_[:,k] = u
                self.y_loadings_[:,k] = vh.ravel()
                self.S -= delta*np.outer(u,vh)

            self.W = W_rotations.T
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

            self.x_loadings_ = self.U
            self.y_loadings_ = self.Vh.T
            self.W = self.U
        return self

    def _comp_coef(self):
        from scipy.linalg import pinv
        self.x_rotations_ = np.dot(
            self.W,
            pinv(np.dot(self.x_loadings_.T, self.W), check_finite=False),
        )

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.intercept_ = self._y_mean

    def predict(self, X, copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        self._comp_coef()
        X -= self._x_means
        ypred = X @ self.coef_
        ypred += self.intercept_
        return ypred
