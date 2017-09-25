import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pandas as pd


class LogisticRegression(nn.Module):
    def __init__(self, x_train, y_train):
        # Initalising superclass -> nn.Module
        super(LogisticRegression, self).__init__()

        # Training Data
        self.x_train = x_train
        self.y_train = y_train

        # Extract input dimensions and number of classes
        input_size = x_train.shape[1]
        num_classes = len(np.unique(y_train))

        # Prepare Linear Model
        # nn.Linear --> y = mx + b
        self.model = nn.Linear(input_size, num_classes)


    # Train the model, given learning rate, no of epochs and batch_size
    def train(self, batch_size, num_epochs, learning_rate):
        # number of steps taken in order to iterate through complete data
        num_steps = len(self.y_train)//batch_size

        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()

        # SGD Optimiser
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            for step in range(num_steps - 1):
                # Convert from numpy arrays to torch tensors
                # requires_grad=False : We don't need to caluculate
                #                       gradient w.r.t x and y, as we don't update input
                #                     : We only need to update Weights and bias
                #                       So, we calculate grad wrt W and b only
                # Divide data into batches
                batch_x = Variable(torch.from_numpy(self.x_train[step*batch_size:(step+1)*batch_size]).float(), requires_grad=False)
                batch_y = Variable(torch.from_numpy(self.y_train[step*batch_size:(step+1)*batch_size]).long(), requires_grad=False)

                # Forward Pass + Backward Pass (backprop) + Optimize

                # zero_grad() : Clears the gradients of all optimized Variables
                optimizer.zero_grad()

                # model.forward(x) : Given x, return y
                #                    from equation  y = mx + b
                outputs = self.model.forward(batch_x)

                # criterion -> nn.CrossEntropyLoss() : Calculates Sigmoid probablity
                #                                      and then loss b/w expected and predicted class
                loss = criterion(outputs, batch_y)

                # Calculates gradients with backpropogation
                loss.backward()

                # optimizer.step() : Performs a single optimization step
                #                    parameter update : w = w - lr * grad(w)
                optimizer.step()

        return(self.model)

    # Test the Model
    def test(self, x_test, y_test):
        correct, total = 0, 0

        # Get x and y. We don't need y as a variable as we need not compute loss
        test_x = Variable(torch.from_numpy(x_test).float(), requires_grad=False)
        test_y = torch.from_numpy(y_test).long()

        # Get softmax probablities from model
        outputs = self.model.forward(test_x)

        # Get label form max probablity
        _, predicted = torch.max(outputs.data, 1)
        total += test_y.size(0)

        # check how many are predicted correctly
        correct += (predicted == test_y).sum()

        print('Accuracy of the model : %d %%' % (100 * correct / total))


# Detailed explanation : logistic_regression.ipynb
def try_model():
    # Read csv
    data = pd.read_csv('census.csv')

    # Get label
    y = data[['income']]
    # Remove label from data
    x = data.drop(['income'], axis=1, errors='ignore')
    # Columns which are categorical, hence will be one-hot encoded
    one_hot_encode_columns = ['workclass', 'education_level', 'marital-status', 'occupation', 'relationship',
                              'race', 'sex', 'native-country']
    # One-hot encode said columns
    x = pd.get_dummies(x, columns=one_hot_encode_columns)
    # 'Binarise' labels
    y = y.apply(lambda x: 1 if x['income'] == '>50K' else 0, axis=1)

    # Divide into train-test datasets
    x_train, x_test = np.asarray(x[:44000], dtype=np.float32), np.asarray(x[44000:], dtype=np.int32)
    y_train, y_test = np.asarray(y[:44000], dtype=np.float32), np.asarray(y[44000:], dtype=np.int32)

    num_epochs = 50
    batch_size = 100
    learning_rate = 0.001

    # Train and test models
    logistic_model = LogisticRegression(x_train, y_train)
    logistic_model.train(batch_size, num_epochs, learning_rate)
    logistic_model.test(x_test, y_test)


if __name__ == "__main__":
    try_model()
