#! /usr/bin/python3

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


x, labels = make_blobs(random_state=0, n_samples=50, n_features=2, cluster_std=0.5, centers=3)




def CalcEuclideanDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def CalcDist(data, idx1, idx2, f_distance=CalcEuclideanDistance):
    vec1 = data[idx1, :]
    vec2 = data[idx2, :]
    return f_distance(vec1, vec2)


def knnSearch(data, idx, k):
    id = []
    num = []

    for i in range(0, 50):
        id.append(CalcDist(data, idx, i))

    id.pop(idx)
    y = max(id) + 1
    id.insert(idx, y)

    for j in range(0, k):
        a = id.index(min(id))
        num.append(a)
        id.pop(a)
        id.insert(a, y)

    return num


query = input("Please input index of data you wanna query: ")
query = int(query)
k = input("How many most similar points do you wanna query? >> ")
k = int(k)
neighbors = knnSearch(x, query, k)


plt.scatter(x[:, 0], x[:, 1], s=50, marker='o', c='black')
plt.scatter(x[query, 0], x[query, 1], s=50, marker='o', c='blue')
plt.scatter(x[[neighbors], 0], x[[neighbors], 1], s=50, marker='o', c='red')
plt.show()


def knnClassification(data, labels, idx, k):
    num = knnSearch(data, idx, k)
    l = labels[num]
    return stats.mode(l)





plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='autumn', marker='o')
plt.colorbar()
plt.scatter(x[query, 0], x[query, 1], c=labels[query], s=50, marker='o', edgecolors="black")
plt.show()
predicted = knnClassification(x, labels, query, k)
print("Classified Result:", predicted)
