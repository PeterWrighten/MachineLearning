from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

x = np.concatenate([make_blobs(random_state=0, n_samples=100, n_features=2, cluster_std=1.0, centers=2)[0], make_blobs(random_state=0, n_samples=100, n_features=2, cluster_std=0.3, centers=1)[0]])

def CalcEuclideanDistance(vec1, vec2):
  return np.sqrt(np.sum(np.square(vec1 - vec2)))


def CalcDist(data, idx1, idx2, f_distance=CalcEuclideanDistance): 
  vec1 = data[idx1, :]
  vec2 = data[idx2, :]
  return f_distance(vec1, vec2)

def knnSearch(data, idx, k): 
  id = []
  num = []

  for i in range(0, data.shape[0]):
    id.append(CalcDist(data, idx, i))

  id.pop(idx)
  m = max(id) + 1
  id.insert(idx, m)

  for j in range(0, k):
    a = id.index(min(id))
    num.append(a)
    id.pop(a)
    id.insert(a, m)

  return num

def local_density(data, idx, k):
  knn = knnSearch(data, idx, k)
  d = 0.0
  for i in knn:
    d += CalcDist(data, idx, i, f_distance=CalcEuclideanDistance)
  return k / d

def calc_LOF(data, idx, k):
  ld_por = 0.0
  knn = knnSearch(data, idx, k)
  for i in knn:
    ld_por += local_density(data, i, k) / local_density(data, idx, k)
  lof = ld_por / k
  return lof

def do_LOF(data, k, threshold=2.0):
  results = []
  for idx in range(data.shape[0]):
    lof = calc_LOF(data, idx, k)
    if lof > threshold:
      results.append(-1)
    else:
      results.append(0)
  return np.array(results)

k = 50
lof_res = do_LOF(x, k)

def draw_clustering_result(ax, N_CLUSTERS, X, pred, centers=None, title = "clustring result", show_legend = True):
    for i in range(N_CLUSTERS):
        labels = X[pred == i]
        if len(labels) > 0:
            ax.scatter(labels[:, 0], labels[:, 1], label="inlier" )
    ourlier_labels = X[pred == -1]
    if len(ourlier_labels) > 0:
        ax.scatter(ourlier_labels[:, 0], ourlier_labels[:, 1], color="k", s=8, label="outlier")
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], s=100,
                facecolors='none', edgecolors='black')
    ax.set_title(title)
    if show_legend:
        ax.legend()

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
axL.scatter(x[:,0],x[:,1])
axL.set_title("original data")
draw_clustering_result(axR, 2, x, lof_res, title = "LOF result")
fig.show()

clf = IsolationForest(random_state=0).fit(x)
if_res = clf.predict(x)
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
axL.scatter(x[:,0],x[:,1])
axL.set_title("original data")
draw_clustering_result(axR, 2, x, if_res, title = "LOF result")
fig.show()

clf1 = LocalOutlierFactor(n_neighbors=3, contamination=0.1)
l_res = clf.fit_predict(x)
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
axL.scatter(x[:,0],x[:,1])
axL.set_title("original data")
draw_clustering_result(axR, 2, x, l_res, title = "LOF result")
fig.show()