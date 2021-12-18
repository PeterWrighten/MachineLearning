#! /usr/bin/python3

from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

X, Y, coef = make_regression(random_state=10, n_samples=100, n_features=2, n_informative=1, noise=10.0, bias=2.0, coef=True)

def plot_3D_Reg(X, Y, a=None, b=None, angle=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y, c = 'blue', marker = 'o', alpha = 1)
    if a is not None:
        x_surf = np.arange(-3, 3, 0.1)
        y_surf = np.arange(-3, 3, 0.1)
        x_surf, y_surf = np.meshgrid(x_surf, y_surf)
        out = a + b[0] * x_surf + b[1] * y_surf
        ax.plot_surfaces(x_surf, y_surf, out.reshape(x_surf.shape), rstride=1, cstride=1, color='None', alpha = 0.4)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_zlabel('Y')
        if angle is not None:
            ax.view_init(30, angle)
        plt.show()

plot_3D_Reg(X, Y, angle=0)
plot_3D_Reg(X, Y, angle=45)
plot_3D_Reg(X, Y, angle=90)
plot_3D_Reg(X, Y, angle=135)
plot_3D_Reg(X, Y, angle=180)

plt.scatter(X[:,0], Y, s=50, marker='o')
plt.xlabel("X0")
plt.ylabel("Y")
plt.show()

plt.scatter(X[:,1], Y, s=50, marker='o')
plt.xlabel("X1")
plt.ylabel("Y")
plt.show()

#Gradient descent

def optimizeMultiRegression(X, Y, l_rate=0.005, n_epoch=10):
    loss_history = [ ]
    a = 0
    b = np.zeros_like(X.shape[1])
    for epoch in range(n_epoch):
        total_loss = (1 / 2) * np.sum((a + (b * X)[:, 0 + 1] - Y) ** 2)
        update_a = np.sum(a + (b * X)[:, 0 + 1] - Y) 
        update_b = np.zeros_like(b) 
        update_b[0] = np.sum((a + (b * X)[:, 0 + 1] - Y) * X[:,0])
        update_b[1] = np.sum((a + (b * X)[:, 0 + 1] - Y) * X[:,1])
        a = a - l_rate * update_a
        b = b - l_rate * update_b   
        loss_history.append((total_loss/(2*X.shape[0])))
     
    return a, b, loss_history

def plot_loss(losses):
    print("[Bias]")
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss(MSE)")
    plt.show()

a, b, loss_history = optimizeMultiRegression(X, Y)
plot_loss(loss_history)

plot_3D_Reg(X, Y, a, b)
plot_3D_Reg(X, Y, a, b, angle = 0)
plot_3D_Reg(X, Y, a, b, angle = 45)
plot_3D_Reg(X, Y, a, b, angle = 90)
plot_3D_Reg(X, Y, a, b, angle = 135)
plot_3D_Reg(X, Y, a, b, angle = 180)

def evaluate_regression(X, Y, a, b):
  pred = a + np.dot(X, b)
  print(pred.shape)
  return r2_score(Y, pred) 

print("R2:", evaluate_regression(X, Y, a, b))

#Lasso Regression      

def optimizeMultiRegressionLASSO(X, Y):
    ls = linear_model.Lasso(alpha = 0.005)
    ls.fit(X, Y)
    a = ls.intercept_
    b = ls.coef_
    return a, b 

a, b = optimizeMultiRegressionLASSO(X, Y)

plot_3D_Reg(X, Y, a, b)
plot_3D_Reg(X, Y, a, b, angle = 0)
plot_3D_Reg(X, Y, a, b, angle = 45)
plot_3D_Reg(X, Y, a, b, angle = 90)
plot_3D_Reg(X, Y, a, b, angle = 135)
plot_3D_Reg(X, Y, a, b, angle = 180)

print("R2:", evaluate_regression(X, Y, a, b))