import csv
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  make_scorer, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

#Load Data
def load_data(filename, columns=None):
    data = pd.read_csv(filename)
    return data

#Create processed data
def preprocess(data):
    standard_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(data),
      columns=data.columns
    )
    return x_train_scaled

    minmax_scale = MinMaxScaler()
    x_train = pd.DataFrame(
      minmax_scale.fit_transform(x_train_scaled),
      columns=data.columns
    )
    return x_train

#Model classification
def create_grid_search(X,Y, model,metric):
    # Create a grid search based model
    gs_model = GridSearchCV(estimator = model["model"], param_grid = model['parameters'], scoring=metric, n_jobs = -1,return_train_score=True)
    gs_model.fit(X.values,Y.values)
    return gs_model

def write_cv_results(grid, regressor):
    ## Results from grid search
    mean_test = grid.cv_results_['mean_test_score']
    mean_test_std = grid.cv_results_['std_test_score']
    mean_train = grid.cv_results_['mean_train_score']
    mean_train_std = grid.cv_results_['std_train_score']
    
    d = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(zip(grid.cv_results_["mean_test_score"],grid.cv_results_["std_test_score"], grid.cv_results_["mean_train_score"], grid.cv_results_["std_train_score"]),  columns=["Test Accuracy","Test STD","Train Accuracy","Train STD"])],axis=1)
    d.to_csv(f"results-{regressor}.csv")


if __name__ == "__main__":
    model_names=['Support Vector', 'Random Forest', 'Decision Tree', 'Multi Layer Perceptron', 'Extreme Gradient Boosting',]
    #Mainlines
    raw_data = load_data('data.csv')
    data = preprocess(raw_data)
    X,Y = data.iloc[:,:-1],data.iloc[:,-1]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0)
    #Scorer for grid search
    scorer = make_scorer(r2_score, greater_is_better = True)

    #Models definitions
    model_s = dict()
    model_s['SVR']= { 'model':SVR(), 'parameters': { 
        "kernel": ["rbf","linear"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]
        }
    }
    model_s['RF']= { 'model':RandomForestRegressor(), 'parameters': {
        'max_features': [2, 3],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'n_estimators': [10, 20, 50, 100]} }

    model_s['MLP'] = {'model': MLPRegressor(), 'parameters': {
        'hidden_layer_sizes': [(5,1),(10,1),(15,1), (5,2), (10,2), (15,2)],
        'activation': ['relu','tanh','logistic'],
        'alpha': [0.001, 0.01, 0.035, 0.1],
        'max_iter' : [10000]
    }}

    model_s['XGB'] = {'model':XGBRegressor(),'parameters': {
        'eta':[ 0.1, 0.2, 0.4, 0.6],
        'n_estimators': [10, 20, 50, 100]
        }}

    model_s['DT']= { 'model':DecisionTreeRegressor(), 'parameters' : {
        "criterion":["squared_error","absolute_error"],
        "max_depth" : [1,2,4,8],
        "min_samples_leaf":[1,2,3,4,5],
    }}

    #Definitions
    model_performances = dict()
    predicts = dict()
    score_functions = dict()


    #Score functions
    score_functions["R2"] = r2_score
    score_functions["MAE"] = mean_absolute_error
    score_functions["MSE"] = mean_squared_error
    score_functions["RMSE"] = mean_squared_error


    #model_performances
    performances = dict()
    for k,v in model_s.items():
        cvmodel = create_grid_search(X_train,Y_train, v,metric=scorer)
        write_cv_results(cvmodel, k)
        filename = f"final-version-{k}.sav"
        joblib.dump(cvmodel.best_estimator_, filename)
        model_performances[k] = cvmodel
        y_true, predicts[k] = Y_test, model_performances[k].best_estimator_.predict(X_test)
        for s,f in score_functions.items():
            if s == "MSE":
                performances[k, s] = f(y_true,predicts[k],squared=False) 
            else:
                performances[k, s] = f(y_true,predicts[k]) 
            print(performances[k, s])
        print(f"Best configuration of {k}: {model_performances[k].best_params_} ")

    print(performances)