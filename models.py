# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:43:37 2020

@author: Prathamesh
"""

import sys
sys.path.append(r'F:\ML_DL_Implementations')

import learning_algorithms as la
import numpy as np

class LogisticRegression:
    """This is a logistic regression model that can be used for binary classification.
    It can also be used for multi-class classification using one-vs-all method."""
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def initialize_parameters(self,n):
        """Initialize parameters w to a vector of zeros and b to a scalar 0.
        n is the number of features excluding bias term 1.
        w is a vector of shape (n,1).
        b is a scalar."""
        self.w = np.zeros(n).reshape(n,1) # initialize w to a vector of zeros.
        self.b = 0 # initialize scalar b to 0.
        
    def get_prob(self,X,y):
        """compute and return probability p(y=1|x) for 1 or more examples."""
        m,n = X.shape # number of examples,number of features.
        X2 = X.T # transpose of feature matrix for algebra convenience.
        y = y.reshape(m,1) # reshape if in case, y is of shape (m,).
        z = (np.dot(self.w.T,X2) + self.b).reshape(m,1) # vector z = w.x + b
        y_hat = (1 / (1+np.exp(-z))).reshape(m,1) # vector y_hat = sigmoid(z).
        return y_hat # return vector of conditional probabilities.
    
    def get_class(self,X,y):
        """compute and return class prediction for 1 or more examples.
        X is the feature matrix of shape (m,n).
        y is the label vector of shape (m,1)."""
        m = X.shape[0] # # number of examples
        y_hat = self.get_prob(X,y) # vector of conditional probabilities.
        y_pred = np.array(y_hat>=0.5,dtype='uint8').reshape(m,1) # convert to vector of classes with threshold 0.5
        return y_pred # return vector of classes
        
    def get_cost(self,X,y):
        """compute and return cost function J = (1/m)*(sum of loss function L over all examples).
        loss function L = - ( y*ln(y_hat)) + (1-y)*ln(1-y_hat) ).
        y_hat is the estimation of p(y=1|x).
        X is the feature matrix of shape (m,n).
        y is the label vector of shape (m,1)."""
        m = X.shape[0] # number of examples
        y = y.reshape(m,1) # reshape if in case, y is of shape (m,).
        y_hat = self.get_prob(X,y) # vector of conditional probabilities.
        L = (-1) * ( np.multiply(y,np.log(y_hat))
                    + np.multiply(1-y,np.log(1-y_hat))) # vector L 
        J = (np.sum(L))/m # cost = mean of the loss function values.
        return J # return cost.
    
    def get_gradients(self,X,y):
        """compute and return partial derivatives of cost function J w.r.t all parameters w and b.
        These partial derivatives are used by learning algorithm to update the parmeters w and b."""
        m = X.shape[0] # number of examples
        X2 = X.T # transpose of feature matrix for algebra convenience.
        y = y.reshape(m,1) # reshape if in case, y is of shape (m,).
        y_hat = self.get_prob(X,y) # vector of conditional probabilities.
        dJ_dw = (1/m)*(np.dot(X2,y_hat-y)) # vector of derivatives of cost function J w.r.t all the parameters in w
        dJ_db = (1/m)*(np.sum(y_hat-y)) # derivative of cost function J w.r.t b
        return dJ_dw,dJ_db # return partial derivatives.

    def train(self,X,y,alpha,noi,method='sgd',bs=10):
        """learn the parameters w and b.
        X is the feature matrix of shape (m,n).
        y is the label vector of shape (m,1).
        alpha is the learning rate.
        noi is the number of iterations."""
        m,n = X.shape # number of examples,number of features
        self.initialize_parameters(n) # initialize parameters w and b before starting the iterative learning algorithm
        if method=='gd':
            la.gradient_descent(X,y,self,alpha,noi) # learn the parameters by minimizing cost function J.
        elif method=='sgd':
            la.stochastic_gradient_descent(X,y,self,alpha,noi)
        elif method=='mbgd':
            la.mini_batch_gradient_descent(X,y,self,alpha,noi,bs)
        
    def __str__(self):
        s = ('Logistic Regression\n' + 
             'w : {}\n'.format(self.w) + 
             'b : {}\n'.format(self.b))
        return s