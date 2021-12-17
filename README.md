# MechineLearning

This is a MachineLearning course powered by Python 3.



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
    for i in range(0, 50):
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

