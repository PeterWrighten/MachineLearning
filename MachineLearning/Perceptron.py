#! /usr/bin/python3

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

x, labels = make_blobs(random_state = 1, n_samples = 100, n_features = 2, cluster_std = 1.2, centers = 2)

plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='autumn', marker='o' )
plt.colorbar()
plt.show()

X_train, X_test, label_train, label_test = train_test_split(x, labels, train_size=0.7, random_state=0)

plt.scatter(X_train[:, 0], X_train[:, 1], s=50, marker='o', label="train")
plt.scatter(X_test[:, 0], X_test[:, 1], s=50, marker='o', label="test")
plt.legend()

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




perceptron = SimplePerceptron(X_train.shape[1])
perceptron.fit(X_train, label_train)
perceptron.evaluate(X_test, label_test)





