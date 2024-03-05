"""ISVDPLS."""

import numpy as np
from numpy.linalg import norm,inv,svd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from scipy.sparse.linalg import svds

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
    def CentralizedData(self,X):
        if (len(X.shape) == 1):
            Xc = X - np.mean(X)
        else:
            X_mean = np.mean(X,axis=0)
            Xc = np.zeros(X.shape)
            for i in np.arange(X.shape[0]):
                Xc[i,:] = X[i,:] - X_mean
        return Xc

    def __init__(self, n_components=10, copy=True):
        self.__name__ = ' (SVDPLS)'
        self.n_components = n_components
        self.n = 0
        self.copy = copy
        self.W = None
        self.mu_x = 0
        self.mu_y = 0
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

            W_rotations = np.zeros((self.n_components, self.n_features))

            self.mu_x = np.mean(X,axis=0)
            self.mu_y = np.mean(Y,axis=0)
            Xc = np.zeros(X.shape)
            for i in np.arange(self.n):
                Xc[i,:] = X[i,:] - self.mu_x
            Yc = np.zeros(Y.shape)
            for i in np.arange(self.n):
                Yc[i,:] = Y[i,:] - self.mu_y
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
                self.S -= delta*np.outer(u,vh)
            self.W = W_rotations.T
        else:
            for j in range(X.shape[0]):
                self.n += 1
                x = X[j]
                y = Y[j]

                self.mu_x *= self.n-1
                self.mu_x += x
                self.mu_x /= self.n
                x -= self.mu_x

                self.mu_y *= self.n-1
                self.mu_y += y
                self.mu_y /= self.n
                y -= self.mu_y

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
            self.W = self.U
        return self

    def PLSR_coef(self,X,Y,copy=True):
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        X = self.CentralizedData(X)
        Y = self.CentralizedData(Y)

        T  = X @ self.W  # score matrix of Xc
        Qt = T.T @ Y     # transpose of loading matrix of Yc
        return self.W @ inv(T.T @ T) @ Qt
