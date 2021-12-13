#! /usr/bin/python3

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


x, labels = make_blobs(random_state=0, n_samples=50, n_features=2, cluster_std=0.5, centers=3)
plt.scatter(x[:, 0], x[:, 1], s=50, marker='o')
plt.show()


def CalcEuclideanDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def CalcDist(data, idx1, idx2, f_distance=CalcEuclideanDistance):
    vec1 = data[idx1, :]
    vec2 = data[idx2, :]
    return f_distance(vec1, vec2)


def FindMostSimilarPoint(data, idx):
    id = []
    for i in range(0, 50):
        id.append(CalcDist(data, idx, i))
    id.pop(idx)
    y = max(id) + 1
    id.insert(idx, y)
    return id.index(min(id))


query = input("Please input query number: ")
query = int(query)
nn_idx = FindMostSimilarPoint(x, query)


plt.scatter(x[:, 0], x[:, 1], s=50, marker='o', c='black')
plt.scatter(x[query, 0], x[query, 1], s=50, marker='o', c='blue')
plt.scatter(x[nn_idx, 0], x[nn_idx, 1], s=50, marker='o', c='red')
plt.show()
