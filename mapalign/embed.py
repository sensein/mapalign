"""Generate a diffusion map embedding of a 2D dataset
"""

def compute_embeddings(K, n_components=50, diffusion_time=0):
    import numpy as np
    import scipy.sparse as sps

    from sklearn.manifold.spectral_embedding_ import _graph_is_connected

    if not _graph_is_connected(K):
        raise ValueError('Graph is disconnected')

    ndim = K.shape[0]
    v = np.array(np.sqrt(K.sum(axis=1))).flatten()
    A = K.copy()
    del K
    A.data /= v[A.indices]
    A = sps.csr_matrix(A.transpose().toarray())
    A.data /= v[A.indices]
    A = sps.csr_matrix(A.transpose().toarray())

    from sklearn.utils.arpack import eigsh, eigs

    func = eigs
    if n_components is not None:
        lambdas, vectors = func(A, k=n_components + 1)
    else:
        lambdas, vectors = func(A, k=max(2, int(np.sqrt(ndim))))
    del A

    if func == eigsh:
        lambdas = lambdas[::-1]
        vectors = vectors[:, ::-1]
    else:
        lambdas = np.real(lambdas)
        vectors = np.real(vectors)
        lambda_idx = np.argsort(lambdas)[::-1]
        lambdas = lambdas[lambda_idx]
        vectors = vectors[:, lambda_idx]

    psi = vectors/vectors[:, [0]]
    if diffusion_time <= 0:
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

    result = dict(lambdas=lambdas, vectors=vectors, embedding=embedding,
                  n_components=n_components, diffusion_time=diffusion_time,
                  n_components_auto=n_components_auto)
    return result

"""
    from nipype.utils.filemanip import split_filename

    import nibabel as nb
    img = nb.load(filename)
    ntimepoints, nsamples = img.data.shape

    K = (np.corrcoef(img.data[:, :img.header.matrix.mims[1].brainModels[2].indexOffset].T) + 1) / 2.
    del img
"""