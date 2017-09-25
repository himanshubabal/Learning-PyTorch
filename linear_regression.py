import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, x_train, y_train):
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

    def train(self, num_epochs, learning_rate):
        # Convert from numpy arrays to torch tensors
        # requires_grad=False : We don't need to caluculate
        #                       gradient w.r.t x and y
        x = Variable(torch.from_numpy(self.x_train).float(), requires_grad=False)
        y = Variable(torch.from_numpy(self.y_train).float(), requires_grad=False)

        # Loss : Mean Square Error
        criterion = nn.MSELoss()
        # SGD Optimiser
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

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

    # Plots 2D graph with predicted values
    def plot_model(self, x_test, y_test):
        predicted = self.model(Variable(torch.from_numpy(x_test))).data.numpy()
        plt.plot(x_test, y_test, 'r', label='Original data')
        plt.plot(x_test, predicted, label='Fitted line')
        plt.legend()
        plt.show()

    # Plots 3D graph with predicted values
    def plot_3D_Model(self, x_test, y_test):
        # Plot the graph
        pred = self.model(Variable(torch.from_numpy(x_test))).data.numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x_test[:,0], x_test[:,1], zs=y_test[:,0], label='Original data')
        ax.plot(x_test[:,0], x_test[:,1], zs=pred[:,0], label='Fitted line')

        ax.set_xlabel('X 1')
        ax.set_ylabel('X 2')
        ax.set_zlabel('Y')
        plt.legend()

        plt.show()



num_epochs = 100
learning_rate = 0.0001

x_train = np.array(np.mgrid[1:101:1].reshape(1, -1).T, dtype=np.float32)
y_train = np.array([2*i+1 for i in x_train], dtype=np.float32)

x_test = np.array(np.mgrid[200:250:1].reshape(1, -1).T, dtype=np.float32)
y_test = np.array([2*i+1 for i in x_test], dtype=np.float32)

linear_model = LinearRegression(x_train, y_train)
linear_model.train(num_epochs, learning_rate)
linear_model.plot_loss()
linear_model.plot_model(x_test, y_test)


num_epochs = 100
learning_rate = 0.001

x = np.float32(np.mgrid[1:10:1, 1:10:1].reshape(2, -1).T)
y = np.float32([2*i[0] + 3*i[1] + 4 for i in x])[:,np.newaxis]

linear_model = LinearRegression(x, y)
linear_model.train(num_epochs, learning_rate)
linear_model.plot_loss()
linear_model.plot_3D_Model(x, y)



num_epochs = 1000
learning_rate = 0.001

xx = np.float32(np.mgrid[1:10:1, 1:10:1, 1:10:1, 1:10:1, 1:10:1, 1:10:1].reshape(6, -1).T)

a = np.float32([2*i[0] + 3*i[1] + 4*i[1] + 6 for i in xx])[:,np.newaxis]
b = np.float32([7*i[2] + 9*i[3] + 5*i[4] + 4 for i in xx])[:,np.newaxis]
c = np.float32([4*i[4] + 1*i[5] + 9*i[0] + 1 for i in xx])[:,np.newaxis]
yy = np.hstack((a, b, c))

linear_model = LinearRegression(xx, yy)
linear_model.train(num_epochs, learning_rate)
linear_model.plot_loss()
