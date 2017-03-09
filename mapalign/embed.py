"""Generate a diffusion map embedding
"""

import numpy as np


def compute_diffusion_map(L, alpha=0.5, n_components=None, diffusion_time=0,
                          skip_checks=False, overwrite=False):
    """Compute the diffusion maps of a symmetric similarity matrix

        L : matrix N x N
           L is symmetric and L(x, y) >= 0

        alpha: float [0, 1]
            Setting alpha=1 and the diffusion operator approximates the
            Laplace-Beltrami operator. We then recover the Riemannian geometry
            of the data set regardless of the distribution of the points. To
            describe the long-term behavior of the point distribution of a
            system of stochastic differential equations, we can use alpha=0.5
            and the resulting Markov chain approximates the Fokker-Planck
            diffusion. With alpha=0, it reduces to the classical graph Laplacian
            normalization.

        n_components: int
            The number of diffusion map components to return. Due to the
            spectrum decay of the eigenvalues, only a few terms are necessary to
            achieve a given relative accuracy in the sum M^t.

        diffusion_time: float >= 0
            use the diffusion_time (t) step transition matrix M^t

            t not only serves as a time parameter, but also has the dual role of
            scale parameter. One of the main ideas of diffusion framework is
            that running the chain forward in time (taking larger and larger
            powers of M) reveals the geometric structure of X at larger and
            larger scales (the diffusion process).

            t = 0 empirically provides a reasonable balance from a clustering
            perspective. Specifically, the notion of a cluster in the data set
            is quantified as a region in which the probability of escaping this
            region is low (within a certain time t).

        skip_checks: bool
            Avoid expensive pre-checks on input data. The caller has to make
            sure that input data is valid or results will be undefined.

        overwrite: bool
            Optimize memory usage by re-using input matrix L as scratch space.

        References
        ----------

        [1] https://en.wikipedia.org/wiki/Diffusion_map
        [2] Coifman, R.R.; S. Lafon. (2006). "Diffusion maps". Applied and
        Computational Harmonic Analysis 21: 5-30. doi:10.1016/j.acha.2006.04.006
    """

    import numpy as np
    import scipy.sparse as sps

    use_sparse = False
    if sps.issparse(L):
        use_sparse = True

    if not skip_checks:
        from sklearn.manifold.spectral_embedding_ import _graph_is_connected
        if not _graph_is_connected(L):
            raise ValueError('Graph is disconnected')

    ndim = L.shape[0]
    if overwrite:
        L_alpha = L
    else:
        L_alpha = L.copy()

    if alpha > 0:
        # Step 2
        d = np.array(L_alpha.sum(axis=1)).flatten()
        d_alpha = np.power(d, -alpha)
        if use_sparse:
            L_alpha.data *= d_alpha[L_alpha.indices]
            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
            L_alpha.data *= d_alpha[L_alpha.indices]
            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())
        else:
            L_alpha = d_alpha[:, np.newaxis] * L_alpha 
            L_alpha = L_alpha * d_alpha[np.newaxis, :]

    # Step 3
    d_alpha = np.power(np.array(L_alpha.sum(axis=1)).flatten(), -1)
    if use_sparse:
        L_alpha.data *= d_alpha[L_alpha.indices]
    else:
        L_alpha = d_alpha[:, np.newaxis] * L_alpha

    M = L_alpha

    from scipy.sparse.linalg import eigsh, eigs

    # Step 4
    func = eigs
    if n_components is not None:
        lambdas, vectors = func(M, k=n_components + 1)
    else:
        lambdas, vectors = func(M, k=max(2, int(np.sqrt(ndim))))
    del M

    if func == eigsh:
        lambdas = lambdas[::-1]
        vectors = vectors[:, ::-1]
    else:
        lambdas = np.real(lambdas)
        vectors = np.real(vectors)
        lambda_idx = np.argsort(lambdas)[::-1]
        lambdas = lambdas[lambda_idx]
        vectors = vectors[:, lambda_idx]

    return _step_5(lambdas, vectors, ndim, n_components, diffusion_time)


def _step_5(lambdas, vectors, ndim, n_components, diffusion_time):
    """
    This is a helper function for diffusion map computation.

    The lambdas have been sorted in decreasing order.
    The vectors are ordered according to lambdas.

    """
    psi = vectors/vectors[:, [0]]
    diffusion_times = diffusion_time
    if diffusion_time == 0:
        diffusion_times = np.exp(1. -  np.log(1 - lambdas[1:])/np.log(lambdas[1:]))
        lambdas = lambdas[1:] / (1 - lambdas[1:])
    else:
        lambdas = lambdas[1:] ** float(diffusion_time)
    lambda_ratio = lambdas/lambdas[0]
    threshold = max(0.05, lambda_ratio[-1])

    n_components_auto = np.amax(np.nonzero(lambda_ratio > threshold)[0])
    n_components_auto = min(n_components_auto, ndim)
    if n_components is None:
        n_components = n_components_auto
    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]

    result = dict(lambdas=lambdas, vectors=vectors,
                  n_components=n_components, diffusion_time=diffusion_times,
                  n_components_auto=n_components_auto)
    return embedding, result


def compute_diffusion_map_psd(
        X, alpha=0.5, n_components=None, diffusion_time=0):
    """
    This variant requires L to be dense, positive semidefinite and entrywise
    positive with decomposition L = dot(X, X.T).

    """
    from scipy.sparse.linalg import svds

    # Redefine X such that L is normalized in a way that is analogous
    # to a generalization of the normalized Laplacian.
    d = X.dot(X.sum(axis=0)) ** (-alpha)
    X = X * d[:, np.newaxis]

    # Decompose M = D^-1 X X^T
    # This is like
    # M = D^-1/2 D^-1/2 X (D^-1/2 X).T D^1/2
    # Substituting U = D^-1/2 X we have
    # M = D^-1/2 U U.T D^1/2
    # which is a diagonal change of basis of U U.T
    # which itself can be decomposed using svd.
    d = np.sqrt(X.dot(X.sum(axis=0)))
    U = X / d[:, np.newaxis]

    if n_components is not None:
        u, s, vh = svds(U, k=n_components+1, return_singular_vectors=True)
    else:
        k = max(2, int(np.sqrt(ndim)))
        u, s, vh = svds(U, k=k, return_singular_vectors=True)

    # restore the basis and the arbitrary norm of 1
    u = u / d[:, np.newaxis]
    u = u / np.linalg.norm(u, axis=0, keepdims=True)
    lambdas = s*s
    vectors = u

    # sort the lambdas in decreasing order and reorder vectors accordingly
    lambda_idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[lambda_idx]
    vectors = vectors[:, lambda_idx]

    return _step_5(lambdas, vectors, X.shape[0], n_components, diffusion_time)


def main():
    # run a test

    from numpy.testing import assert_allclose

    def _nonnegative_corrcoef(X):
        return (np.corrcoef(X) + 1) / 2.0

    def _factored_nonnegative_corrcoef(X):
        X = X - X.mean(axis=1, keepdims=True)
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        U = np.hstack([U, np.ones((U.shape[0], 1))])
        return U / np.sqrt(2)

    X = np.random.randn(100, 20)
    L = _nonnegative_corrcoef(X)
    U = _factored_nonnegative_corrcoef(X)

    assert_allclose(L, U.dot(U.T))

    alpha = 0.2
    n_components = 7
    diffusion_time = 2.0

    stuff_a = compute_diffusion_map(L, alpha, n_components, diffusion_time)
    embedding_a, result_a = stuff_a

    stuff_b = compute_diffusion_map_psd(U, alpha, n_components, diffusion_time)
    embedding_b, result_b = stuff_b

    # The embeddings should be the same up to coordinate signs.
    # In other words, if the x coordinate in one embedding
    # is interpreted as the -x coordinate in another embedding,
    # then the embeddings are not really different.
    assert_allclose(
            embedding_a / np.sign(embedding_a[0]),
            embedding_b / np.sign(embedding_b[0]))

    # Same thing for vectors.
    assert_allclose(
            result_a['vectors'] / np.sign(result_a['vectors'][0]),
            result_b['vectors'] / np.sign(result_b['vectors'][0]))

    # Check the other stuff.
    for x in 'lambdas', 'diffusion_time', 'n_components', 'n_components_auto':
        assert_allclose(result_a[x], result_b[x])


if __name__ == '__main__':
    main()
