import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

'''
Credits -
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py

Lot of this has been inspired by / taken from above link
I have tried to explain little bit deeper
'''


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, x_train, y_train):
        # Initalising superclass -> nn.Module
        super(LinearRegression, self).__init__()

        # Training Data
        self.x_train = x_train
        self.y_train = y_train

        # Extract input and output dimensions
        input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        # Prepare Linear Model
        # nn.Linear --> y = mx + b
        self.model = nn.Linear(input_size, output_size)
        # If GPU available
        if torch.cuda.is_available():
            self.model.cuda()


    # Train the model, given learning rate and no of epochs
    def train(self, num_epochs, learning_rate):
        # Convert from numpy arrays to torch tensors
        # requires_grad=False : We don't need to caluculate
        #                       gradient w.r.t x and y, as we don't update input
        #                     : We only need to update Weights and bias
        #                       So, we calculate grad wrt W and b only
        if torch.cuda.is_available():
            x = Variable(torch.from_numpy(self.x_train).float().cuda(), requires_grad=False)
            y = Variable(torch.from_numpy(self.y_train).float().cuda(), requires_grad=False)
        else:
            x = Variable(torch.from_numpy(self.x_train).float(), requires_grad=False)
            y = Variable(torch.from_numpy(self.y_train).float(), requires_grad=False)

        # Loss : Mean Square Error
        criterion = nn.MSELoss()

        # SGD Optimiser
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Array that will store loss at each epoch
        # so we can plot it later
        self.loss_array = list()
        for epoch in range(num_epochs):
            # Forward Pass + Backward Pass (backprop) + Optimize

            # zero_grad() : Clears the gradients of all optimized Variables
            optimizer.zero_grad()

            # model.forward(x) : Given x, return y
            #                    from equation  y = mx + b
            outputs = self.model.forward(x)

            # criterion -> nn.MSELoss() : Calculates MSE in y_pred and y_original
            #                             loss = sigma(y_pred - y_original)^2
            loss = criterion(outputs, y)

            # Calculates gradients with backpropogation
            loss.backward()

            # optimizer.step() : Performs a single optimization step
            #                    parameter update : w = w - lr * grad(w)
            optimizer.step()

            self.loss_array.append(loss.data[0])

        return(self.loss_array, self.model)


    # Plots loss with each epoch
    def plot_loss(self):
        plt.plot(range(len(self.loss_array)), self.loss_array)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.show()

    # Plots 2D graph with predicted values : Case - 1
    def plot_model(self, x_test, y_test):
        if torch.cuda.is_available():
            predicted = self.model(Variable(torch.from_numpy(x_test).cuda())).data.numpy()
        else:
            predicted = self.model(Variable(torch.from_numpy(x_test))).data.numpy()
        plt.plot(x_test, y_test, 'r', label='Original data')
        plt.plot(x_test, predicted, label='Fitted line')
        plt.legend()
        plt.show()

    # Plots 3D graph with predicted values : Case - 2
    def plot_3D_Model(self, x_test, y_test):
        # Plot the graph
        pred = self.model.forward(Variable(torch.from_numpy(x_test))).data.numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x_test[:,0], x_test[:,1], zs=y_test[:,0], label='Original data')
        ax.plot(x_test[:,0], x_test[:,1], zs=pred[:,0], label='Fitted line')

        ax.set_xlabel('X 1')
        ax.set_ylabel('X 2')
        ax.set_zlabel('Y')
        plt.legend()

        plt.show()


# ------------------------------------------------------------------------
# --------------------------- CASE : 1 -----------------------------------
# --------------- input : (n, 1) --- output : (n, 1)) --------------------
# ------------------- Simple case of y = mx + c --------------------------
def case_1():
    num_epochs = 100
    learning_rate = 0.0001

    # numpy array of shape (n, 1)
    x_train = np.array(np.mgrid[1:101:1].reshape(1, -1).T, dtype=np.float32)
    y_train = np.array([2*i+1 for i in x_train], dtype=np.float32)

    x_test = np.array(np.mgrid[200:250:1].reshape(1, -1).T, dtype=np.float32)
    y_test = np.array([2*i+1 for i in x_test], dtype=np.float32)

    linear_model = LinearRegression(x_train, y_train)
    linear_model.train(num_epochs, learning_rate)
    linear_model.plot_loss()
    linear_model.plot_model(x_test, y_test)

# ------------------------------------------------------------------------
# --------------------------- CASE : 2 -----------------------------------
# --------------- input : (n, 2) --- output : (n, 1)) --------------------
# ---------------------- y = m1*x1 + m2*x2 + c ---------------------------
def case_2():
    num_epochs = 100
    learning_rate = 0.001

    # numpy array of shape (n, 2)
    x = np.float32(np.mgrid[1:10:1, 1:10:1].reshape(2, -1).T)
    # np_array -> (n,)
    # np_array[:,np.newaxis] -> (n, 1)
    y = np.float32([2*i[0] + 3*i[1] + 4 for i in x])[:,np.newaxis]

    linear_model = LinearRegression(x, y)
    linear_model.train(num_epochs, learning_rate)
    linear_model.plot_loss()
    # We will need to plot 3D figure in this case
    # as there are 3 Dimensions - x1, x2, y
    linear_model.plot_3D_Model(x, y)


# ------------------------------------------------------------------------
# --------------------------- CASE : 1 -----------------------------------
# --------------- input : (n, 6) --- output : (n, 3)) --------------------

# ---------------- y1 = m1*x1 + m2*x2 + m3*x3 + c ------------------------
# ---------------- y2 = m4*x3 + m5*x4 + m6*x5 + c ------------------------
# ---------------- y1 = m7*x5 + m8*x6 + m9*x1 + c ------------------------
# ------------------------- y = [y1  y2  y3] -----------------------------
def case_3():
    num_epochs = 1000
    learning_rate = 0.001

    xx = np.float32(np.mgrid[1:10:1, 1:10:1, 1:10:1, 1:10:1, 1:10:1, 1:10:1].reshape(6, -1).T)

    a = np.float32([2*i[0] + 3*i[1] + 4*i[2] + 6 for i in xx])[:,np.newaxis]
    b = np.float32([7*i[2] + 9*i[3] + 5*i[4] + 4 for i in xx])[:,np.newaxis]
    c = np.float32([4*i[4] + 1*i[5] + 9*i[0] + 1 for i in xx])[:,np.newaxis]
    yy = np.hstack((a, b, c))

    linear_model = LinearRegression(xx, yy)
    linear_model.train(num_epochs, learning_rate)
    linear_model.plot_loss()


if __name__ == "__main__":
    case_1()
    case_2()
    case_3()
