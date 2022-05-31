import joblib
from matplotlib.font_manager import json_dump
import pandas as pd
from Model  import *
import json 

model_names=['SVR', 'RF', 'DT', 'MLP', 'XGB',]

raw_data = load_data('data.csv')
data = preprocess(raw_data)
X,Y = data.iloc[:,:-1],data.iloc[:,-1]


performances = dict()
predicts = dict()
score_functions = dict()
score_functions["R2"] = r2_score
score_functions["MAE"] = mean_absolute_error
score_functions["MSE"] = mean_squared_error
score_functions["RMSE"] = mean_squared_error

for k in model_names:
    filename = f"final-version-{k}.sav"
    loaded_model = joblib.load(filename)
    y_true, predicts[k] = Y, loaded_model.predict(X)
    performances[k] = dict()
    for s,f in score_functions.items():
        if s == "MSE":
            performances[k][s] = f(y_true,predicts[k],squared=False) 
        else:
            performances[k][s] = f(y_true,predicts[k]) 
        print(performances[k][s])
print(performances)

with open("results.json", "w") as outfile:
    json.dump(performances, outfile)