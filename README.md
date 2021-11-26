# MechineLearning

This is a MachineLearning course powered by Python 3.

# Lecture 1: Nearest Neighbor Algorithms


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



# Lecture 2: K Nearest Neighbor Algorithms

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



# Lecture 3: Perceptron

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