# MechineLearning

This is a MachineLearning course powered by Python 3.


**Syntax**

- KNN Search 

Sort

```python
import numpy as np
sample_arr = np.array([4, 2, 6, 1, 3, 0])
np.sort(sample_arr)
```

Argsort(Descent sequence to display index)

```python
np.argsort(sample_arr)
```

Choose Multi-variable

```python
sample_list = np.array([5, 4, 3, 2, 1, 0])
indices = [0, 2, 4]
sample_list[indices]
```
Mode

```python
import scipy.stats as stats
sample_list = np.array([0, 1, 2, 1, 1, 1])
mode, _ = stats.mode(sample_list)
mode[0]
```




# Lecture 1: Supervised Learning

## Part 1: Classification Analysis

### Nearest Neighbor Algorithms


Firstly, define function which could calculate Euclid Distance of 2 vectors;

```python
import numpy as np;
def CalcEuclideanDistance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def CalcDist(data, idx1, idx2, f_distance=CalcEuclideanDistance):
    vec1 = data[idx1, :]
    vec2 = data[idx2, :]
    return f_distance(vec1, vec2)
```

Then, Find most similar point and return index of it;

```python
def FindMostSimilarPoint(data, idx):
    id = []
    for i in range(50):
        id.append(CalcDist(data, idx, i))
    id.pop(idx)
    y = max(id) + 1
    id.insert(idx, y)
    return id.index(min(id))
```



### K Nearest Neighbor Algorithms

Return the list of indices which are most similar with input.

```python
def knnSearch(data, idx, k):
    id = []
    num = []

    for i in range(50):
        id.append(CalcDist(data, idx, i))

    id.pop(idx)
    y = max(id) + 1
    id.insert(idx, y)

    for j in range(k):
        a = id.index(min(id))
        num.append(a)
        id.pop(a)
        id.insert(a, y)

    return num
```



### Perceptron

```Predict```: input original data and output predictions calculated by weights multiply data.

```python
def _predict(weights, one_data):
    output = np.sum(weights * one_data)
    return 1 if output >= 0.5 else 0
```

```fit```: optimize ```w``` by gradient descent.

```python
def _fit(weights, X_train, label_train, l_rate, n_epoch):
    loss_history = [ ]
    for epoch in range(n_epoch):
        total_loss = 0.0
        update_weights = np.zeros_like(weights)
        for i in X_train:
            y_hat = _predict(weights, i[0])
            total_loss += (y_hat - i[1])**2
            for j in weights:
                update_weights += (y_hat - i[1]) * i[0]
        loss_history.append((total_loss/(2 * X_train.shape[0])))
        weights = update_weights
    return weights, loss_history
```

```evaluate```: F-Measure

```python 
def evaluate(self, X_test, label_test):
        predicts = []
        for one_data, one_label in zip(X_test, label_test):
            pred = self.predict(one_data)
            predicts.append(pred)
        predicts = np.array(predicts)
        print("[Show performance Metrics]")
        print(classification_report(label_test, predicts, target_names=["0", "1"]))
        cm = confusion_matrix(label_test, predicts)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.show()
        print("[Predicts]")
        plt.scatter(X_test[predicts == 0, 0], X_test[predicts == 0, 1], s=50, marker='o', label = "predicted as 0")
        plt.scatter(X_test[predicts == 1, 0], X_test[predicts == 1, 1], s=50, marker='o', label = "predicted as 1")
        plt.legend()
        plt.show()
```

**Summary**: ```class SimplePerceptron```:

```python
def _predict(weights, one_data):
    output = np.sum(weights * one_data)
    return 1 if output >= 0.5 else 0


def _fit(weights, X_train, label_train, l_rate, n_epoch):
    loss_history = [ ]
    for epoch in range(n_epoch):
        total_loss = 0.0
        update_weights = np.zeros_like(weights)
        for i in X_train:
            y_hat = _predict(weights, i[0])
            total_loss += (y_hat - i[1])**2
            for j in weights:
                update_weights += (y_hat - i[1]) * i[0]
        loss_history.append((total_loss/(2 * X_train.shape[0])))
        weights = update_weights
    return weights, loss_history


class SimplePerceptron:
    def __init__(self, n_dim):
        self.weights = np.zeros(n_dim)
    def predict(self, one_data):
        return _predict(self.weights, one_data)
    def fit(self, X_train, label_train, l_rate=0.1, n_epoch=10):
        self.weights, losses = _fit(self.weights, X_train, label_train, l_rate, n_epoch)
        print("[Error Propagation]")
        plt.plot(losses)
        plt.xlabel("epochs")
        plt.ylabel("loss (MSE)")
    def evaluate(self, X_test, label_test):
        predicts = []
        for one_data, one_label in zip(X_test, label_test):
            pred = self.predict(one_data)
            predicts.append(pred)
        predicts = np.array(predicts)
        print("[Show performance Metrics]")
        print(classification_report(label_test, predicts, target_names=["0", "1"]))
        cm = confusion_matrix(label_test, predicts)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.show()
        print("[Predicts]")
        plt.scatter(X_test[predicts == 0, 0], X_test[predicts == 0, 1], s=50, marker='o', label = "predicted as 0")
        plt.scatter(X_test[predicts == 1, 0], X_test[predicts == 1, 1], s=50, marker='o', label = "predicted as 1")
        plt.legend()
        plt.show()
```

## Part 2: Regression Analysis

### Linear Regression

**Normal Equation**

```python
import numpy as np
def calculateRegression(X, Y):
  size = len(Y)
  y_avg = np.sum(Y) / size
  x_avg = np.sum(X) / size
  s_x2 = (1/size) * np.sum(X[:,0] ** 2 - x_avg ** 2)
  s_xy = (1/size) * np.sum(X[:,0] * Y - x_avg * y_avg)
  b = s_xy / s_x2 
  a = y_avg - b * x_avg
  return a, b
```

**Gradient Descent**

```python
def optimizeRegression(X, Y, l_rate=0.005, n_epoch=10):
  loss_history = [] 
  a = 0 
  b = 0 
  for epoch in range(n_epoch):
    total_loss = (1 / 2) * (a + b * X[:,0] - Y) ** 2 
    update_a = np.sum(a + b * X[:, 0] - Y) 
    update_b = np.sum((a + b * X[:, 0] - Y) * X[:, 0]) 
    a -= l_rate * update_a
    b -= l_rate * update_b   
    loss_history.append((total_loss/(2*X.shape[0])))
  return a, b
```

**MultiRegression**


**Lasso Regression**

In order to implement Lasso Regression, we would utilize Python Module ```from sklearn import linear_model```, and implement API ```Lasso.fit(X, Y)```, then ```Lasso.coef_``` is weights, and ```Lasso.intercept_``` is intercept.

# Lecture 2: Unsupervised Learning

## Part 1: Clustering

### K-means Algorithm

*Algorithm's Basis:* Minimum Spanning Tree in Graph.

*Prerequisites:* 

1. The number of clusters necessary.
2. All axis's data necessary

*Advantage:* Benifitial to grasp the data structure.

*Method:* 

- Initialization: Acollate centers randomly.
- 1. Calculate EuclidDistance between *centers* and *data*. 
- 2. classify spot to its nearest center's cluster
- 3. Recenter cluster (New Mean)
- Loop 1, 2, 3

**Initialization**

1. Random Init

```python
n_clust = 5

def init_random(X, n_clust):
    centers = np.zeros((n_clust, X.shape[1]))
    np.random.seed(seed = 32)
    centers[:, 0] = np.random.randint(low = min(X[:, 0]), high = max(X[:, 0]), size = n_clust)
    centers[:, 1] = np.random.randint(low = min(X[:, 1]), high = max(X[:, 1]), size = n_clust)
    return centers
```

2. K-means++: seperate clusters's centers.

*Method:*

Define probablity to choose next cluster's center:

- Proportional to square of distances among chosen centers.

![picture 1](images/b9e1a778bd02fb01594f69c4ace72a8314a1071eeaca02b5348cb337faac6ffa.png) 

- Numerator: square of distance between current spot and chosen center(s).
- Denominator: N means the dataset which waits to be chosen.

![picture 2](images/4e5fcb95cbc40f64dabb831948736277263c6272a1aa23c4d70b6a32971712ae.png)  

- Orange: chosen center(s)
- Blue: Waitlist


```python 
def init_kmpp(X, n_clust):
    np.random.seed(seed = 32)
    indice = np.random.choice(range(X.shape[0]), n_clust)
    centers = X[indice, :]
    return centers
```

**K-means**

```python
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
```

## Part 2: Outlier Detection

### Local Outlier Factor
**LOF: Essence**

![picture 1](images/00523bb5b28948dbe4c585944677b935d6003a0bc4a358b3dfed62b35e839e90.png)  

![picture 2](images/4797e8aa5d3b39a578e2c6657cd8be7f9472263dcc38b08a6f25e4ab7f40f68e.png)  

when LOF(p) > threshold, it was seperated.
