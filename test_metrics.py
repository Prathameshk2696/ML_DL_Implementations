# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:10:20 2020

@author: Prthamesh
"""

import numpy as np

def accuracy(y,y_pred):
    """compute and return classification accuracy.
    accuracy = number of correct predictions / total number of predictions."""
    y_pred = y_pred.astype(y.dtype)
    ac =  np.sum(y == y_pred) / y.shape[0]
    return ac

def precision(y,y_pred):
    """compute and return precision.
    precision = # true positives / (# true positives + # false positives)"""
    y_pred = y_pred.astype(y.dtype)
    prec = np.sum(y)/(np.sum(np.bitwise_or(y,y_pred)))
    return prec

def recall(y,y_pred):
    """compute and return recall.
    recall = # true positives / (# true positives + # false negatives)"""
    y_pred = y_pred.astype(y.dtype)
    rec = np.sum(y)/(np.sum(y)+np.sum(np.bitwise_and(y,1-y_pred)))
    return rec


