import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from matplotlib import cm
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler


boston = load_boston()
df = pd.DataFrame(boston["data"], columns=boston["feature_names"])
df["MEDV"] = boston["target"]
X = df[["LSTAT", "RM"]].values
y = df["MEDV"].values


# This class is a heavily inspired by GWU MLI professor James' class  
class LinReg:

    def __init__(self, n_iter=50, eta=10 ** -5, random_state=0):

        self.n_iter = n_iter
        self.eta = eta
        self.random_state = random_state
        self.w = None
        
    def fit(self, X, y, batch=True, shuffle=False):
        
        self.w = [-6]*X.shape[1]
        weights_1, weights_2 = [], []
        weights_1.append(self.w[0])
        weights_2.append(self.w[1])
        grads_w1 = []
        grads_w2 = []
        mse = []
        for _ in range(self.n_iter):
            if batch:
                delta_w = [0]*X.shape[1]
            e = 0
            for i in range(X.shape[0]): 
                net_input = self.net_input(X, i)
                error = y[i] - net_input
                e += error**2
                for j in range(X.shape[1]):
                    if batch:
                        delta_w[j] += self.eta * error * X[i][j]
                    else:
                        self.w[j] += self.eta * error * X[i][j]
            if batch:
                grads_w1.append(-delta_w[0]/self.eta)
                grads_w2.append(-delta_w[1]/self.eta)
            else:
                grads_w1.append(-self.w[0]/self.eta)  # This doesn't really make sense
                grads_w2.append(-self.w[1]/self.eta)
            if batch:   
                for j in range(X.shape[1]):
                    self.w[j] += delta_w[j]
            weights_1.append(self.w[0])
            weights_2.append(self.w[1])
            mse.append(e/X.shape[0])
            if not batch and shuffle:
                feat_target = np.hstack((X, y.reshape(-1, 1)))
                np.random.shuffle(feat_target)
                X = feat_target[:, :-1]
                y = feat_target[:, -1].reshape(-1)
            
        return weights_1, weights_2, grads_w1, grads_w2, mse

    def net_input(self, X, i):
        
        weighted_sum = 0
        for j in range(X.shape[1]):
            weighted_sum += X[i][j] * self.w[j]

        return weighted_sum

    def predict(self, X):
        
        y_pred = []
        for i in range(X.shape[0]):
            net_input = self.net_input(X, i)
            y_pred.append(net_input)

        return y_pred
    
    def score(self, X, y):

        y_pred = self.predict(X)
        e_res = ((y - y_pred)**2).sum()
        e_tot = ((y - y.mean())**2).sum()

        return 1 - e_res/e_tot


def J(W1, W2, X, y):
    error = 0
    for feat, target in zip(X, y):
        y_hat = W1*feat[0] + W2*feat[1]
        error += (target - y_hat)**2
    return error/2


W1, W2 = np.meshgrid(np.linspace(-100, 100, 30), np.linspace(-100, 100, 30))
F = J(W1, W2, X, y)

# Shows the objective function in the weight space
plt.figure(figsize=[15, 11])
ax = plt.axes(projection='3d')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.contour3D(W1, W2, F, 100, cmap=cm.coolwarm)
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('$J(w_1, w_2)$')
title1 = "Objective function for Linear Regression model on the Boston Housing Dataset"
title2 = "\nusing only LSTAT = x and RM = $x^*$ as predictors, and with no intercept"
title3 = "\n$J(w_1, w_2) = 0.5 \cdot \sum_i^n (y_i - w_1x_i + w_2x^*_i)^2$"
plt.title(title1 + title2 + title3)
ax.view_init(10, 40)
plt.show()


def plot_obj_func_and_learn_process(eta, n_iter, show_lines=False, batch=True):

    lr = LinReg(eta=eta, n_iter=n_iter)
    w_1, w_2, grads_w1, grads_w2, mse = lr.fit(X, y, batch=batch)

    w1 = np.linspace(np.min(w_1), np.max(w_1), 30)
    W1, W2 = np.meshgrid(np.linspace(np.min([-6, np.min(w_1)]), np.max([6, np.max(w_1)]), 30),
                         np.linspace(np.min([-6, np.min(w_2)]), np.max([6, np.max(w_2)]), 30))
    F = J(W1, W2, X, y)

    plt.figure(figsize=[15, 11])
    ax = plt.axes(projection='3d')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.contour3D(W1, W2, F, 100, cmap=cm.coolwarm)
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_zlabel('$J(w_1, w_2)$')
    ax.view_init(10, 40)
    z = []
    z.append(J(w_1[0], w_2[0], X, y))
    ax.scatter3D(w_1[0], w_2[0], z[0], marker="^", color="k", label="starting point")
    if batch:
        if n_iter == 1:
            ax.quiver(w_1[0], w_2[0], 0, -grads_w1[0], -grads_w2[0], 0, length=1, normalize=True, color="r", label="$-$ Gradient")
            ax.quiver(0, 0, 0, w_1[0], w_2[0], 0, length=1, color="k")
            ax.quiver(0, 0, 0, w_1[1], w_2[1], 0, length=1, color="k")
        else:
            ax.quiver(w_1[0], w_2[0], 0, -grads_w1[0], -grads_w2[0], 0, length=1, normalize=True, color="r", label="$-$ Gradient")
    for i in range(1, len(w_1)):
        m = "*" if i%2 == 0 else "."
        z.append(J(w_1[i], w_2[i], X, y))
        ax.scatter3D(w_1[i], w_2[i], z[i], marker=m, color="k")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        if batch:
            ax.quiver(w_1[i-1], w_2[i-1], 0, -grads_w1[i-1], -grads_w2[i-1], 0, length=1, normalize=True, color="r")
        mi, ma = np.min([w_1[i-1], w_1[i]]), np.max([w_1[i-1], w_1[i]])
        ran = w1[(w1 >= mi) & (w1 <= ma)]
        y_line = (ran-w_1[i])/(w_1[i-1]-w_1[i])*(w_2[i-1]-w_2[i]) + w_2[i]
        z_line = (ran-w_1[i])/(w_1[i-1]-w_1[i])*(z[i-1]-z[i]) + z[i]
        if show_lines:
            if i == 1:
                ax.plot3D(ran, y_line, z_line, 'gray', label="Gradient Descent Path")
            else:
                ax.plot3D(ran, y_line, z_line, 'gray')
    plt.legend()
    bs = "Batch" if batch else "Stochastic"
    title1 = bs + " Gradient Descent Optimization for $\eta=${} and {} epochs\n".format(eta, n_iter)
    title2 = "Final $J(w)$: {}, final weights: $w_1 = ${}, $w_2 = ${}\n".format(round(z[-1], 2), round(lr.w[0], 2), round(lr.w[1], 2))
    plt.title(title1 + title2 + "Final $R^2 = ${}".format(round(lr.score(X, y), 2)))
    plt.show()

    #plt.plot(range(len(z)), z)
    #plt.show()



# Shows how we update the weights for the first epoch
#plot_obj_func_and_learn_process(eta=10**(-6), n_iter=1)
# Shows the gradient descent path 
plot_obj_func_and_learn_process(eta=10**(-6), n_iter=500)
# Idem for bigger learning rate
#plot_obj_func_and_learn_process(eta=10**(-5), n_iter=50, show_lines=True)
# Even bigger 
#plot_obj_func_and_learn_process(eta=1.52*10**(-5), n_iter=50, show_lines=True)
# Too big
#plot_obj_func_and_learn_process(eta=1.68*10**(-5), n_iter=10, show_lines=True)
# Shows Batch and Shows stochastic
#plot_obj_func_and_learn_process(eta=10**(-6), n_iter=50)
#plot_obj_func_and_learn_process(eta=10**(-6), n_iter=50, batch=False)
# This are not good examples :/ It's because we only have two features?


#std = StandardScaler()
#X = std.fit_transform(X)
#y = std.fit_transform(y.reshape(-1, 1)).reshape(-1)
X = X/2
y = y/2
plot_obj_func_and_learn_process(eta=10**(-6), n_iter=2000)



def show_errors():
    X = df.drop(["MEDV", "B", "TAX"], axis=1).values
    y = df["MEDV"].values

    plt.title("MSE along the training process")
    lr = LinReg(eta=1.7*10**(-4), n_iter=20)
    w_1, w_2, grads_w1, grads_w2, mse = lr.fit(X, y, batch=False)
    plt.plot(range(len(mse)), mse, label="Stochastic MSE")
    plt.xlabel("epoch"), plt.ylabel("MSE")
    plt.legend()
    plt.show()

    plt.title("MSE along the training process")
    lr = LinReg(eta=1.3*10**(-7), n_iter=20)
    w_1, w_2, grads_w1, grads_w2, mse = lr.fit(X, y, batch=False)
    plt.plot(range(len(mse)), mse, label="Stochastic MSE")
    lr = LinReg(eta=1.3*10**(-7), n_iter=20)
    w_1, w_2, grads_w1, grads_w2, mse = lr.fit(X, y)
    plt.plot(range(len(mse)), mse, label="Batch MSE", linestyle="dashed")
    plt.xlabel("epoch"), plt.ylabel("MSE")
    plt.legend()
    plt.show()

show_errors()