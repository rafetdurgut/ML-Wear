import joblib
import pandas as pd
from pyparsing import col
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
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
    minmax_scale = MinMaxScaler()
    x_train = pd.DataFrame(
      minmax_scale.fit_transform(x_train_scaled),
      columns=data.columns
    )
    return x_train


#Model classification
def create_grid_search(X,Y, model,metric):
    # Create a grid search based model
    gs_model = GridSearchCV(estimator = model["model"], param_grid = model['parameters'],verbose = 2, scoring=metric, n_jobs = -1)
    gs_model.fit(X.values,Y.values)
    return gs_model

#SVR, RF, DT, MLP, GBM, XGBoost, ELM

model_names=['Support Vector', 'Random Forest', 'Decision Tree', 'Multi Layer Perceptron', 'Extreme Gradient Boosting',]
#Mainlines
raw_data = load_data('data.csv')
data = preprocess(raw_data)
X,Y = data.iloc[:,:-1],data.iloc[:,-1]

#Scorer for grid search
scorer = make_scorer(r2_score, greater_is_better = True)

#Models definitions
model_s = dict()
model_s['SVR']= { 'model':SVR(), 'parameters': [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]},
    {"kernel": ["linear"], "C": [1, 10, 100]},
] }
model_s['RF']= { 'model':RandomForestRegressor(), 'parameters': {
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [10, 20, 50, 100]} }

model_s['MLP'] = {'model': MLPRegressor(), 'parameters': {
    'hidden_layer_sizes': [(5,1),(10,1),(15,1), (5,2), (10,2), (15,2)],
    'activation': ['relu','tanh','logistic'],
    'alpha': [0.035, 0.001, 0.01],
    'max_iter':[5000]
}}
model_s['XGB'] = {'model':XGBRegressor(),'parameters': {
    'eta':[ 0.1, 0.2, 0.4, 0.6],
    'n_estimators': [10, 20, 50, 100]
    }}
model_s['DT']= { 'model':DecisionTreeRegressor(), 'parameters' : {
    "splitter":["best","random"],
    "max_depth" : [1,5,7,10],
    "min_samples_leaf":[1,2,5],
    "min_weight_fraction_leaf":[0.1,0.2,0.5],
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
    model_performances[k] = create_grid_search(X,Y, v,metric=scorer)
    y_true, predicts[k] = Y, model_performances[k].best_estimator_.predict(X)
    for s,f in score_functions.items():
        if s == "RMSE":
            performances[k, s] = f(y_true,predicts[k],squared=False) 
        else:
            performances[k, s] = f(y_true,predicts[k]) 
        print(performances[k, s])
    print(f"Best configuration of {k}: {model_performances[k].best_params_} ")
    filename = f"final-version-{k}.sav"
    joblib.dump(model_performances[k].best_estimator_, filename)

print(performances)