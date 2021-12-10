from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

X, Y, coef = make_regression(random_state = 12, n_samples = 100, n_features = 1, n_informative = 1, noise = 10.0, bias = 2.0, coef = True)

plt.scatter(X[:,0], Y, s = 50, marker = 'o')
plt.xlabel("X0")
plt.ylabel("y")
plt.show()

def calculateRegression(X, Y):
    size = len(Y)
    x_avg = np.sum(X[:,0]) / size
    y_avg = np.sum(Y) / size
    s_xy = (1 / size) * np.sum(X[:,0] * Y - x_avg * y_avg)
    s_x2 = (1 / size) * np.sum(X[:,0] ** 2 - x_avg ** 2)
    b = s_xy / s_x2
    a = y_avg - b * x_avg
    return a, b

def drawRegressionLine(X, Y, a, b):
    x_max = np.max(X) * 1.1
    x_min = np.min(X) * 1.1 
    x = np.linspace(x_min, x_max, 1000)
    y = a + b * x  
    plt.plot(x, y, label='Regression line', color = "red")
    plt.scatter(X, Y, s = 50, marker='o', label = "data")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

a, b = calculateRegression(X, Y)

def evaluateRegression(X, Y, a, b):
    pred = a + X * b
    return r2_score(Y, pred)

print("Bias: ", evaluateRegression(X, Y, a, b))

drawRegressionLine(X, Y, a, b)

def optimizeRegression(X, Y, l_rate=0.005, n_epoch=100):
    loss_history = [ ]
    a = 0
    b = 0
    for epoch in range(n_epoch):
        total_loss = (1 / 2) * (a + b * X[:, 0] - Y) ** 2
        update_a = np.sum(a + b * X[:, 0] - Y)
        update_b = np.sum((a + b * X[:, 0] - Y) * X[:, 0])
        a -= l_rate * update_a
        b -= l_rate * update_b
        loss_history.append((total_loss / (2*X.shape[0])))

    plot_loss(loss_history)
    plt.show()
    return a, b   

def plot_loss(losses):
    print("[Bias Plot]")
    plt.plot(losses)
    plt.xlabel("epoches")
    plt.ylabel("loss (MSE)")

a1, b1 = optimizeRegression(X, Y)

drawRegressionLine(X, Y, a1, b1)

print("Bias: ", evaluateRegression(X, Y, a1, b1))