#!/usr/bin/python
# -*- coding: ascii -*-
#   __      _____             _              __        _       
# <(o )___ |  _  |___ ___ ___| |_ ___ ___   |  |   ___| |_ ___ 
# ( ._> /  |   __|  _| -_|_ -|  _| -_|_ -|  |  |__| .'| . |_ -|
#  `---'   |__|  |_| |___|___|_| |___|___|  |_____|__,|___|___|
#==============================================================================
# Title          : hparameter optimization visualization.py
# Description    : hyper-parameter optimization with visualization (Parfit) on 
#                  Random Forest Classifier.
# Author         : Isaias V. Prestes <isaias.prestes@gmail.com>
# Date           : 201803
# Version        : 0.0.1
# Usage          : python hparameter optimization visualization.py
# Notes          : Based on author code https://github.com/jmcarpenter2/parfit
# Python_version : 3.6  
#==============================================================================

# hyper-parameter optimization with visualization (Parfit) on Random Forest Classifier
# https://github.com/jmcarpenter2/parfit

# Cross-validation using GridSearchCV optimizes on the training data! As 
# mentioned in Rachel’s post, this is dangerous and not applicable to most 
# real-world modeling problems (especially time series data). Instead, we 
# want to optimize our hyper-parameters on the validation set.
# 
# Process
#
# * Create an effective validation set
# * Choose your model(s)
# * Create a parameter grid over which to evaluate your model on the validation set
# * Use parfit to visualize the scores over the grid and select the best model
# * Re-train model on full training set, using best parameters from parfit
# * Apply re-trained model to test set
#
# parfit_ex.ipynb - examples of how to use for varying 1, 2, and 3 parameters on a generated data set.
# https://github.com/jmcarpenter2/parfit/blob/master/examples/parfit_ex.ipynb

# pip install parfit

###############################################################################
# Import the modules needed to run the script.
###############################################################################
import parfit.parfit as pf
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

###############################################################################
# Body
###############################################################################

paramGrid = ParameterGrid({
    'min_samples_leaf': [1,3,5,10,15,25,50,100,125,150,175,200],
    'max_features': ['sqrt', 'log2', 0.4, 0.5, 0.6, 0.7],
    'n_estimators': [60],
    'n_jobs': [-1],
    'random_state': [42]
})

best_model, best_score, all_models, all_scores = pf.bestFit(RandomForestClassifier, paramGrid, 
     X_train, y_train, X_val, y_val, 
     metric=roc_auc_score, bestScore='max', scoreLabel='AUC')
print(best_model)

