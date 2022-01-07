#! /usr/bin/python3

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib

x, labels = make_blobs(random_state = 1, n_samples = 100, n_features = 2, cluster_std = 0.7, centers = 5)

def plot_clusters(x, clst_idx = None, centers = None):
    plt.figure(figsize=(5, 5))
    if clst_idx is None:
        plt.scatter(x[:, 0], x[:, 1], s=50, marker='o', label="data points")
    else:
        plt.scatter(x[:, 0], x[:, 1], s = 50, marker = 'o', c = clst_idx, cmap = "Paired", label = "data points", norm = matplotlib.colors.Normalize(vmin=-0, vmax=len(centers)))
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], s = 50, c = [i for i in range(len(centers))], cmap = "Paired", marker = 'x', label = "centers", norm = matplotlib.colors.Normalize(vmin=-0, vmax=len(centers)))
    plt.legend()
    plt.show()

# Define The Number of Clusters

n_clust = 5

def init_random(X, n_clust):
    centers = np.zeros((n_clust, X.shape[1]))
    np.random.seed(seed = 32)
    centers[:, 0] = np.random.randint(low = min(X[:, 0]), high = max(X[:, 0]), size = n_clust)
    centers[:, 1] = np.random.randint(low = min(X[:, 1]), high = max(X[:, 1]), size = n_clust)
    return centers

centers = init_random(x, n_clust)
plot_clusters(x, None, centers)

def kmeans(X, n_clust, centers, n_iter = 10):
    clst_idx = np.zeros(X.shape[0])
    dist = np.zeros((X.shape[0], n_clust))
    for iter in range(n_iter):
        for i in range(X.shape[0]):
            for k in range(n_clust):
                dist[i][k] = np.sum(np.square(centers[k, :] - X[i, :]))
        clst_idx = np.argmin(dist, axis = 1)
        plot_clusters(x, clst_idx, centers)

        for j in range(n_clust):
            centers[j, :] = np.mean(X[clst_idx == j], axis = 0)
    
    return clst_idx

clst_idx = kmeans(x, n_clust, centers, n_iter = 10)

plot_clusters(x, clst_idx, centers)

def init_kmpp(X, n_clust):
    np.random.seed(seed = 32)
    indice = np.random.choice(range(X.shape[0]), n_clust)
    centers = X[indice, :]
    return centers

centers = init_kmpp(x, n_clust)
plot_clusters(x, None, centers)

clst_idex = kmeans(x, n_clust, centers)

plot_clusters(x, clst_idx, centers)