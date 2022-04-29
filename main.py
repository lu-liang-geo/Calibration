import glob
import pandas as pd
import datetime
import os
import numpy as np
import math
from sklearn.metrics import r2_score

from parameters import *
from algorithms import runAllAlgorithms, runAllAlgorithms_noKFold
# import create_figures
import preprocessing

""" 
Inputs
"""
runAlgs_varyInputs = False
runAlgs_varySampleSize = False

""" 
Read Data
"""

master_df = pd.read_csv("master_df.csv")

if runAlgs_varyInputs:
    """ 
    Run algorithms
    """

    # Create datasets based on list of feature sets
    datasets = []
    for featureSet in datasetFeatures: datasets.append(master_df[featureSet])

    # Create dataframe to hold all results
    results_df = pd.DataFrame(columns=['Algorithm', 'Parameters', 'Features', 'RMSE', 'R2'])

    for dataset, datasetFeature in zip(datasets[-1:], datasetFeatures[-1:]):
        # Separate Features and Labels for current dataset
        X = dataset.drop('PM25FM', axis=1).values
        Y = dataset.PM25FM.values.reshape((-1, 1))
        print(X)

        results = runAllAlgorithms(
            X, Y, nn_params, svm_params, rf_params, knn_params, MLR_params, dt_params,
            ridge_params, lasso_params, bayes_params, ada_params, grad_params
        )

        results.insert(loc = 2, column = 'Features', value = [datasetFeature] * len(results))

        results_df = results_df.append(results, ignore_index=True)

    results_df.RMSE = results_df.RMSE.apply(min, args=(10,))
    results_df.R2 = results_df.R2.apply(max, args=(0,))

    results_df.to_csv("results_features_algorithms.csv")

if runAlgs_varySampleSize:

    datasetFeatures = ['ID', *datasetFeatures[-1]]

    dataset = master_df[datasetFeatures]

    dataset = dataset[dataset.ID.isin(siteToKeep)]

    dataset = dataset.drop('ID', axis=1)

    X = dataset.drop('PM25FM', axis=1).values
    Y = dataset.PM25FM.values.reshape((-1, 1))

    results_df = runAllAlgorithms_noKFold(
        X, Y, nn_params, svm_params, rf_params, knn_params, MLR_params, dt_params,
        ridge_params, lasso_params, bayes_params, ada_params, grad_params
    )

    results_df.RMSE = results_df.RMSE.apply(min, args=(10,))
    results_df.R2 = results_df.R2.apply(max, args=(0,))

    results_df.to_csv("results_amount_of_data.csv")
