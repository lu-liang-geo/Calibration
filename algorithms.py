import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from scipy import stats

import statsmodels.api as sm

def neural_network(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]

    regr = MLPRegressor(hidden_layer_sizes=params[0], random_state=0, max_iter=5000)
    regr.fit(x_train, y_train)
        
    y_pred = regr.predict(x_test)

    return y_pred

def svm(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]

    clf = SVR(kernel=params[0])
    clf.fit(x_train, y_train)
    
    return clf.predict(x_test)

def knn(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    
    neigh = KNeighborsRegressor(n_neighbors=params[0], weights=params[1], n_jobs=6)
    neigh.fit(x_train, y_train)

    return neigh.predict(x_test)

def DT(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def linear_regression(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    
    reg = LinearRegression()
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def ridge_regress(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    
    reg = Ridge(alpha=params[0])
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def lasso_regress(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    
    reg = Lasso(alpha=params[0])
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def bayes_ridge(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    
    reg = BayesianRidge(n_iter=1000)
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def random_forest(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]

    regr = RandomForestRegressor(n_estimators=params[0], random_state=0, n_jobs=6)
    regr.fit(x_train, y_train)

    return regr.predict(x_test)

def ada_boost(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]

    reg = AdaBoostRegressor(random_state=0, n_estimators=params[0])
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def grad_boost(data, params):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]

    reg = GradientBoostingRegressor(
        random_state=0, 
        n_estimators=params[0], 
        subsample=params[1]
    )
    reg.fit(x_train, y_train)

    return reg.predict(x_test)

def chooseAlgorithm(data, alg_name, params_i):
    if alg_name == 'NN':
        pred = neural_network(data, params_i)
    elif alg_name == 'SVM':
        pred = svm(data, params_i)
    elif alg_name == 'RF':
        pred = random_forest(data, params_i)
    elif alg_name == 'KNN':
        pred = knn(data, params_i)
    elif alg_name == 'MLR':
        pred = linear_regression(data, params_i)
    elif alg_name == 'DT':
        pred = DT(data, params_i)
    elif alg_name == 'RR':
        pred = ridge_regress(data, params_i)
    elif alg_name == 'LR':
        pred = lasso_regress(data, params_i)
    elif alg_name == 'BR':
        pred = bayes_ridge(data, params_i)
    elif alg_name == 'AB':
        pred = ada_boost(data, params_i)
    elif alg_name == 'GTB':
        pred = grad_boost(data, params_i)
    else: 
        print("Algorithm name not recognized: " + str(alg_name))

    return pred

def runAlgorithm(alg_name, params, x_train_k, y_train_k, \
    x_test_k, y_test_k_inv, TargetVarScaler, results_df):
    for i, params_i in enumerate(params):

        if alg_name == 'NN':
            params_str = "[" + str(len(params_i[0])) + ", " + str(params_i[0][0]) + "]"
        else:
            params_str = params_i
        results_rmse = 0
        results_r2 = 0
        for j in range(len(x_train_k)):
            data = [x_train_k[j], y_train_k[j], x_test_k[j]]
            
            pred = chooseAlgorithm(data, alg_name, params_i)

            pred_inv = TargetVarScaler.inverse_transform(pred.reshape(-1, 1))
            results_rmse = results_rmse + math.sqrt(mean_squared_error(y_test_k_inv[j], pred_inv))
            results_r2 = results_r2 + r2_score(y_test_k_inv[j], pred_inv)

        result_rmse_avg = results_rmse / 10
        results_r2_avg = results_r2 / 10
        
        results_df.loc[len(results_df.index)] = [
            alg_name, params_str, result_rmse_avg, results_r2_avg
        ]

    print("Finished " + alg_name)

def runAllAlgorithms(
    X, Y, 
    nn_params, svm_params, rf_params, knn_params, MLR_params, dt_params,
    ridge_params, lasso_params, bayes_params, ada_params, grad_params
):

    '''
    Standardize Data
    '''
    PredictorScaler=StandardScaler()
    TargetVarScaler=StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)
    TargetVarScalerFit=TargetVarScaler.fit(Y)

    # Generating the standardized values of X and y
    X=PredictorScalerFit.transform(X)
    Y=TargetVarScalerFit.transform(Y)

    ''' 
    Shuffle Data
    '''
    data = np.append(X, Y, axis=1)

    # Set seed so that shuffle is done same way each time
    np.random.seed(42)
    # Shuffle data
    np.random.shuffle(data)


    X = data[:, :-1]
    Y = data[:, -1:]

    ''' 
    Make Data for K-Fold
    '''
    # The training data split
    x_train_k = []
    y_train_k = []
    x_test_k = []
    y_test_k = []

    y_test_k_inv = []

    x_k = np.array_split(X, 10)
    y_k = np.array_split(Y, 10)

    for i in range(10):
        x_train_k.append(np.vstack(x_k[0:i] + x_k[min(i + 1, 10):-1]))
        x_test_k.append(x_k[i])
        y_train_k.append(np.vstack(y_k[0:i] + y_k[min(i + 1, 10):-1]).ravel())
        y_test_k.append(y_k[i].ravel())

        # Get Inverse-Transformed Y
        y_test_k_inv.append(TargetVarScaler.inverse_transform(y_test_k[i].reshape(-1, 1)))

    ''' 
    Make df for results
    '''
    results_df = pd.DataFrame(columns=['Algorithm', 'Parameters', 'RMSE', 'R2'])

    ''' 
    Run Algorithms
    '''

    runAlgorithm('MLR', MLR_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('RR', ridge_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('LR', lasso_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('BR', bayes_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('RF', rf_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('AB', ada_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('GTB', grad_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)

    runAlgorithm('NN', nn_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('KNN', knn_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)
    runAlgorithm('SVM', svm_params, x_train_k, y_train_k, x_test_k, y_test_k_inv, TargetVarScaler, results_df)

    return results_df

def runAlgorithm_noKFold(num_datasets, alg_name, dataUsed, params, x_train, y_train, \
    x_test, y_test_inv, TargetVarScaler, results_df):
    for i, params_i in enumerate(params): 
        if alg_name == 'NN':
            params_str = "[" + str(len(params_i[0])) + ", " + str(params_i[0][0]) + "]"
        else:
            params_str = params_i

        data = [x_train, y_train, x_test]
        
        pred = chooseAlgorithm(data, alg_name, params_i)

        pred_inv = TargetVarScaler.inverse_transform(pred.reshape(-1, 1))
        
        rmse =  math.sqrt(mean_squared_error(y_test_inv, pred_inv))
        r2 = r2_score(y_test_inv, pred_inv)
        
        results_df.loc[len(results_df.index)] = [
            alg_name, params_str, dataUsed, rmse, r2
        ]

        # print("Finished " + alg_name + " " + str(i+1) + "/" + str(len(params)))

    print("Finished " + alg_name)

def runAllAlgorithms_noKFold(
    X, Y, nn_params, svm_params, rf_params, knn_params, MLR_params, dt_params,
    ridge_params, lasso_params, bayes_params, ada_params, grad_params
):

    '''
    Standardize Data
    '''
    PredictorScaler=StandardScaler()
    TargetVarScaler=StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit=PredictorScaler.fit(X)
    TargetVarScalerFit=TargetVarScaler.fit(Y)

    # Generating the standardized values of X and y
    X=PredictorScalerFit.transform(X)
    Y=TargetVarScalerFit.transform(Y)

    ''' 
    Get test data
    '''
    # Get indices for sampling data to get test data
    # This will get 10% of data (rounded up) for testing
    testIndices = random.sample(range(0, len(Y)), math.ceil(len(Y) * 0.1))

    x_test = X[testIndices]
    x_train = np.delete(X, testIndices, axis=0)
    y_test = Y[testIndices]
    y_train = np.delete(Y, testIndices, axis=0)

    y_test_inv = TargetVarScaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    ''' 
    Get Datasets for Varying Amounts of Data
    '''

    datasets_x = []
    datasets_y = []

    x_split = np.array_split(x_train, 10)
    y_split = np.array_split(y_train, 10)

    for i in range(0, 10):
        datasets_x.append(np.vstack(x_split[0:i+1]))
        datasets_y.append(np.vstack(y_split[0:i+1]).ravel())

    ''' 
    Make df for results
    '''
    results_df = pd.DataFrame(columns=['Algorithm', 'Parameters', '% of Data', 'RMSE', 'R2'])

    ''' 
    Run Algorithms
    '''

    for i in range(len(datasets_y)):

        # Specify the percent of data being used
        dataUsed = (i + 1) * 10

        runAlgorithm_noKFold('RF', dataUsed, rf_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('MLR', dataUsed, MLR_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('RR', dataUsed, ridge_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('LR', dataUsed, lasso_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('BR', dataUsed, bayes_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('AB', dataUsed, ada_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('GTB', dataUsed, grad_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('KNN', dataUsed, knn_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('SVM', dataUsed, svm_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)
        runAlgorithm_noKFold('NN', dataUsed, nn_params, datasets_x[i], datasets_y[i], x_test, y_test_inv, TargetVarScaler, results_df)

        print(f"Finished {dataUsed}%")

    return results_df