# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:41:55 2017

@author: Henrique
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Regression(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        # from data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # will be estimated/calculated
        self.y_hat_train = None
        self.y_hat_test = None
        self.w_hat = None

    def calculateY_Train(self):
        self.y_hat_train = np.dot(self.x_train, self.w_hat)
        return self.y_hat_train

    def calculateY_Test(self):
        self.y_hat_test = np.dot(self.x_test, self.w_hat)
        return self.y_hat_test

    def basicPlots(self, method):
        # y_hat_train vs y_train
        plt.figure()
        plt.plot(self.y_train, self.y_hat_train, marker='.', linestyle='None')
        plt.xlabel('Reference value')
        plt.ylabel('Estimated value')
        plt.title('total_UPDRS - Training set - Y vs Y_hat - ' + method)
        plt.margins(0.01,0.1)# leave some space between the max/min value and the frame
        plt.grid()
        plt.show()

        #y_hat_test vs y_test
        plt.figure()
        plt.plot(self.y_test, self.y_hat_test, marker='.', linestyle='None')
        plt.xlabel('Reference value')
        plt.ylabel('Estimated value')
        plt.title('total_UPDRS - Test set - Y vs Y_hat - ' + method)
        plt.margins(0.01,0.1)# leave some space between the max/min value and the frame
        plt.grid()
        plt.show()

        # histogram of y_train-y_hat_train
        dif_train = pd.Series(self.y_train - self.y_hat_train)
        plt.figure()
        plt.xlabel('Difference between refence and estimated value')
        plt.title('total_UPDRS - Training set - Y-Y_hat - ' + method)
        dif_train.plot.hist(bins=50)

        # histogram of y_test-y_hat_test
        dif_test = pd.Series(self.y_test - self.y_hat_test)
        plt.figure()
        plt.xlabel('Difference between refence and estimated value')
        plt.title('total_UPDRS - Test set - Y-Y_hat - ' + method)
        dif_test.plot.hist(bins=50)

        # weights w
        plt.figure()
        plt.plot(self.w_hat, marker='o', linestyle='None')
        plt.xlabel('Weight index')
        plt.ylabel('Estimated weight')
        plt.title('total_UPDRS - Weights - ' + method)
        plt.margins(0.01,0.1)# leave some space between the max/min value and the frame
        plt.grid()
        plt.show()

class MSE(Regression):

    def __init__(self, x_train, x_test, y_train, y_test):
        Regression.__init__(self, x_train, x_test, y_train, y_test)

    def estimateW(self):
        self.w_hat = np.dot(np.dot(np.linalg.inv(np.dot(self.x_train.transpose(), self.x_train)), self.x_train.transpose()), self.y_train)
        return self.w_hat


class Gradient(Regression):

    def __init__(self, x_train, x_test, y_train, y_test):
        Regression.__init__(self, x_train, x_test, y_train, y_test)

    def estimateW(self, w_initial_guess, max_error, max_iterations, learn_coeff):
        self.w_initial_guess = w_initial_guess
        self.max_error = max_error
        self.max_iterations = max_iterations
        self.learn_coeff = learn_coeff
        self.count = 0
        self.stop_cond = 100000

        w_hat_prev = self.w_initial_guess

        while self.stop_cond > self.max_error and self.count < self.max_iterations:
            self.count = self.count + 1

            grad = (-2) * np.dot(self.x_train.transpose(), self.y_train) + \
                    2 * np.dot((np.dot(self.x_train.transpose(), self.x_train)), \
                    w_hat_prev)

            self.w_hat = w_hat_prev - self.learn_coeff * grad

            self.stop_cond = (np.dot(self.w_hat - w_hat_prev, self.w_hat - w_hat_prev))**(1/2)

            w_hat_prev = self.w_hat

            if (self.count%1000) == 0:
                print ('Iteration: ' + str(self.count))
                print ('Error: ' + str(self.stop_cond))

        print ('End total_UPDRS - Gradient')
        print ('Iteration: ' + str(self.count))
        print ('Error: ' + str(self.stop_cond))


class SteepestDescent(Regression):

    def __init__(self, x_train, x_test, y_train, y_test):
        Regression.__init__(self, x_train, x_test, y_train, y_test)

    def estimateW(self, w_initial_guess, max_error, max_iterations):
        self.w_initial_guess = w_initial_guess
        self.max_error = max_error
        self.max_iterations = max_iterations
        self.count = 0
        self.stop_cond = 100000

        h = 4 * np.dot(self.x_train.transpose(), self.x_train)

        w_hat_prev = self.w_initial_guess

        while self.stop_cond > self.max_error and self.count < self.max_iterations:
               self.count = self.count + 1

               grad = (-2) * np.dot(self.x_train.transpose(), self.y_train) + \
                                    2 * np.dot((np.dot(self.x_train.transpose(), self.x_train)), \
                                    w_hat_prev)

               self.learn_coeff = np.dot(grad, grad) / (np.dot(np.dot(grad.transpose(), h), grad))

               self.w_hat = w_hat_prev - self.learn_coeff * grad

               self.stop_cond = (np.dot(self.w_hat - w_hat_prev, self.w_hat - w_hat_prev))**(1/2)

               w_hat_prev = self.w_hat

               if (self.count%1000) == 0:
                      print ('Iteration: ' + str(self.count))
                      print ('Error: ' + str(self.stop_cond))

        print ('End total_UPDRS - Steepest Descent')
        print ('Iteration: ' + str(self.count))
        print ('Error: ' + str(self.stop_cond))


class Ridge(Regression):

    def __init__(self,x_train, x_test, y_train, y_test):
           Regression.__init__(self, x_train, x_test, y_train, y_test)

    def estimateW(self, lamb):
           size = np.shape(self.x_train)
           identity = np.identity(size[1])

           self.w_hat = np.dot(np.dot(np.linalg.inv(np.dot(self.x_train.transpose(), self.x_train) \
                       + lamb*identity), self.x_train.transpose()), self.y_train)

    # def optimizeLambda(self, lambda_min, lambda_max, step):


if __name__ == '__main__':

       #%% Main
       # seed used to generate always the same initial guess
       seed = 10
       np.random.seed(seed)

       # parameters for Gradient and Steepest Descent
       w_initial_guess = np.random.rand(16,)
       max_error = 10**(-6)
       max_iterations = 10**6

       #%% Load and prepare data
       # load
       data = pd.read_csv('parkinsons_updrs.data')

       # bring time reference to zero (some negative values for time were measured)
       data.test_time = data.test_time - data.test_time.min()

       # round time
       data.test_time = data.test_time.round()

       # aggregate data considering pacient and time
       # more than one measurement (5-6) was taken for each time instant
       data_aggreg = data.groupby(['subject#', 'test_time']).mean()

       # we want to split the data in two sets: training and test
       # training: 1-36
       # test: 37-42
       data_train = data_aggreg.loc[data_aggreg.index.get_level_values('subject#') < 37]
       data_test = data_aggreg.loc[data_aggreg.index.get_level_values('subject#') > 36]

       # we want to normalize both data sets to have mean 0 and variance 1 for each column
       # we'll use the mean and variance from the training set to normalize the test set
       mean_train = data_train.apply('mean')
       var_train = data_train.apply('var')
       data_train_norm = data_train.apply(lambda x : (x - np.mean(x))/(np.var(x)**(1/2.0)))
       # test data
       data_test_norm = pd.DataFrame()
       for i in data_test.keys():
           data_test_norm[i] = data_test[i].apply(lambda x : (x - mean_train[i])/(var_train[i]**(1/2.0)))

       # generate the vector Y which contains the values of the parameter we want to estimate
       # training set
       y_train_df = data_train_norm.loc[:,'total_UPDRS']
       y_train = y_train_df.as_matrix()
       #test set
       y_test_df = data_test_norm.loc[:,'total_UPDRS']
       y_test = y_test_df.as_matrix()

       # generate the matrix X by keeping the features that will be used in the estimation
       # trainign set
       x_train_df = data_train_norm[data_train_norm.columns[4:20]]
       x_train = x_train_df.as_matrix()
       # test set
       x_test_df = data_test_norm[data_test_norm.columns[4:20]]
       x_test = x_test_df.as_matrix()

       #%% Prepare data for cross validation
       # parameter
       k = 5
       num_patients = 42

       #subset_size = int(data_aggreg.shape[0] / k)
       subset_size = int(42/k)

       data_train_k = []
       data_test_k = []

       for i in  range(k):
              test_k = data_aggreg.loc[(i*subset_size+1):((i+1)*subset_size)]
              train_k = data_aggreg.loc[data_aggreg.index.difference(test_k.index)]
              # we want to normalize both data sets to have mean 0 and variance 1 for each column
              # we'll use the mean and variance from the training set to normalize the test set
              mean_train_k = train_k.apply('mean')
              var_train_k = train_k.apply('var')
              train_k_norm = train_k.apply(lambda x : (x - np.mean(x))/(np.var(x)**(1/2.0)))
              # test data
              test_k_norm = pd.DataFrame()
              for j in test_k.keys():
                  test_k_norm[j] = test_k[j].apply(lambda x : (x - mean_train_k[j])/(var_train_k[j]**(1/2.0)))
       
              data_train_k.append(train_k_norm)
              data_test_k.append(test_k_norm)

       #%% MSE
       if False:
              mse = MSE(x_train, x_test, y_train, y_test)
              mse.estimateW()
              mse.calculateY_Train()
              mse.calculateY_Test()
              mse.basicPlots('MSE')

       #%% Gradient
       #parameters
       learn_coeff = 10**(-8)

       if False:
              gradient = Gradient(x_train, x_test, y_train, y_test)
              gradient.estimateW(w_initial_guess, max_error, max_iterations, learn_coeff)
              gradient.calculateY_Train()
              gradient.calculateY_Test()
              gradient.basicPlots('Gradient')

       #%% Steepest Descent
       if False:
              steepest = SteepestDescent(x_train, x_test, y_train, y_test)
              steepest.estimateW(w_initial_guess, max_error, max_iterations)
              steepest.calculateY_Train()
              steepest.calculateY_Test()
              steepest.basicPlots('Steepest Descent')

       #%% Ridge
       # parameters
       lamb_min = 0.1
       lamb_max = 100000
       step = 1.

       if False:
              ridge = Ridge(x_train, x_test, y_train, y_test)
              best_lamb = lamb_min
              best_error = 10**10
              lamb = lamb_min

              while lamb < lamb_max:
                     ridge.estimateW(lamb)
                     ridge.calculateY_Test()

                     error = np.dot(ridge.y_hat_test - ridge.y_test, ridge.y_hat_test - ridge.y_test)

                     if error < best_error:
                            best_error = error
                            best_lamb = lamb

                     lamb = lamb + step

              ridge.estimateW(best_lamb)
              ridge.calculateY_Train()
              ridge.calculateY_Test()
              ridge.basicPlots('Ridge')

