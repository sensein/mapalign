__author__ = 'satra'

import numpy as np

from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sps

def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def compute_nearest_neighbor_graph(K, n_neighbors=50):
    idx = np.argsort(K, axis=1)
    col = idx[:, -n_neighbors:].flatten()
    row = (np.array(range(K.shape[0]))[:, None] * np.ones((1, n_neighbors))).flatten().astype(int)
    A1 = sps.csr_matrix((np.ones((len(row))), (row, col)), shape=K.shape)
    A1 = (A1 + A1.transpose()) > 0
    idx1 = A1.nonzero()
    K = sps.csr_matrix((K.flat[idx1[0]*A1.shape[1] + idx1[1]],
                        A1.indices, A1.indptr))
    return K

