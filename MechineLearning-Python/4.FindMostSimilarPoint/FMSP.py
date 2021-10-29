from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

x, labels = make_blobs(random_state=0, n_samples=50, n_features=2, cluster_std=0.5, centers=3)

plt.scatter(x[:, 0], x[:, 1], s=50, marker='o')


def CalcEuclideanDistance(vec1, vec2):

  return np.sqrt(np.sum(np.square(vec1 - vec2)))


def CalcDist(data, idx1, idx2, f_distance=CalcEuclideanDistance):
  vec1 = data[idx1, :]
  vec2 = data[idx2, :]
  return f_distance(vec1, vec2)


def FindMostSimilarPoint(data, idx):

  for i in range(0, 49):

    if i != idx:
      d = CalcDist(data, idx, i)
      min = CalcDist(data, idx, 0)
      idx_n = 0
      if d < min:
          min = d
          idx_n = i
      else:
          continue
    else:
      continue
    i = i + 1

  return idx_n



query_idx = 10
nn_idx = FindMostSimilarPoint(x, query_idx)

plt.scatter(x[:, 0], x[:, 1], s=50, marker='o', c='black')
plt.scatter(x[query_idx, 0], x[query_idx, 1], s=50, marker='o', c='blue')
plt.scatter(x[nn_idx, 0], x[nn_idx, 1], s=50, marker='o', c='red')
