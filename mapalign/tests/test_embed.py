from ..embed import (compute_diffusion_map, compute_diffusion_map_psd,
                     has_sklearn)

import numpy as np
from pytest import mark

def _nonnegative_corrcoef(X):
    return (np.corrcoef(X) + 1) / 2.0


def _factored_nonnegative_corrcoef(X):
    X = X - X.mean(axis=1, keepdims=True)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    U = np.hstack([U, np.ones((U.shape[0], 1))])
    return U / np.sqrt(2)


def test_psd_with_nonpsd():
    X = np.random.randn(100, 20)
    L = _nonnegative_corrcoef(X)
    U = _factored_nonnegative_corrcoef(X)

    assert np.allclose(L, U.dot(U.T))

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
    assert np.allclose(embedding_a / np.sign(embedding_a[0]),
                       embedding_b / np.sign(embedding_b[0]))

    # Same thing for vectors.
    assert np.allclose(result_a['vectors'] / np.sign(result_a['vectors'][0]),
                       result_b['vectors'] / np.sign(result_b['vectors'][0]))

    # Check the other stuff.
    for x in 'lambdas', 'diffusion_time', 'n_components', 'n_components_auto':
        assert np.allclose(result_a[x], result_b[x])


@mark.skipif(not has_sklearn, reason="scikit-learn not installed")
def test_sklearn_adapter():
    from ..embed import DiffusionMapEmbedding
    X = np.random.randn(100, 20)
    L = _nonnegative_corrcoef(X)
    U = _factored_nonnegative_corrcoef(X)

    assert np.allclose(L, U.dot(U.T))

    alpha = 0.2
    n_components = 7
    diffusion_time = 2.0

    embedding_a = DiffusionMapEmbedding(alpha=alpha, n_components=n_components,
                                        diffusion_time=diffusion_time,
                                        affinity='precomputed',
                                        use_variant=False).fit_transform(L)
    embedding_b = DiffusionMapEmbedding(alpha=alpha, n_components=n_components,
                                        diffusion_time=diffusion_time,
                                        affinity='precomputed',
                                        use_variant=True).fit_transform(U)

    # The embeddings should be the same up to coordinate signs.
    # In other words, if the x coordinate in one embedding
    # is interpreted as the -x coordinate in another embedding,
    # then the embeddings are not really different.
    assert np.allclose(embedding_a / np.sign(embedding_a[0]),
                       embedding_b / np.sign(embedding_b[0]))
