irange = range
from scipy.cluster.vq import *
from pylab import *
import numpy as np
# import faiss
import mkl
def clusterpixels(im, k):
    im = np.array(im)
    h, w = im.shape
    im = im.astype(np.float).reshape(-1)
    # 聚类， k是聚类数目
    centroids, variance = kmeans(im, k)
    code, distance = vq(im, centroids)
    code = code.reshape(h, w)
    fc = sum(code[:, 0])
    lc = sum(code[:, -1])
    fr = sum(code[0, :])
    lr = sum(code[-1, :])
    num = int(fr > w // 2) + int(lr > w // 2) + int(fc > h // 2) + int(lc > h // 2)
    if num >= 3:
        return 1 - code
    else:
        return code

def preprocess_features(npdata, pca=128):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')
    mkl.get_max_threads()
    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata
def run_kmeans(x, pca, nmb_clusters, use_pca=True, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    if use_pca:
        x = preprocess_features(x, pca)
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.centroids)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return I