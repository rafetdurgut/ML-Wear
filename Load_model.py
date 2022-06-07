import joblib
from matplotlib.font_manager import json_dump
import pandas as pd
import sklearn
from Model  import *
import json 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate

model_names=['SVR', 'RF', 'DT', 'MLP', 'XGB',]

def re_standardise(real_data, standardised_data):
    standard_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(real_data),
      columns=data.columns
    )
    dummy = pd.DataFrame(standard_scaler.inverse_transform(standardised_data), columns=real_data.columns) 
    return dummy.iloc[:,-1]

def plot_performance(Xr, Xp, title=None):
    plt.figure()
    plt.title(title)
    sns.regplot(x=Xr, y=Xp, color="g")
    plt.xlabel("Actual Volume Loss")
    plt.ylabel("Predicted Volume Loss")
    plt.show()
raw_data = load_data('data.csv')
raw_X, raw_Y = raw_data.iloc[:,:-1],raw_data.iloc[:,-1]
data = preprocess(raw_data)
X,Y = data.iloc[:,:-1],data.iloc[:,-1]
indices = np.arange(len(Y))
X_train, X_test, Y_train, Y_test, i_train, i_test = train_test_split(X,Y,indices,random_state=42)
predicted_data = data.iloc[i_test,:]


performances = dict()
predicts = dict()
score_functions = dict()
score_functions["R2"] = r2_score
score_functions["MAE"] = mean_absolute_error
score_functions["MSE"] = mean_squared_error
score_functions["RMSE"] = mean_squared_error
results = dict()

for k in model_names:
    filename = f"final-version-{k}.sav"
    loaded_model = joblib.load(filename)
    # Cross-validation yap.
    scoring = {'r2': 'r2',
            'rmse':'neg_root_mean_squared_error',
           'mse': 'neg_mean_squared_error',
           'mae': 'neg_mean_absolute_error'}

    cv_results = cross_validate(loaded_model, X_train, Y_train, cv=5, scoring=scoring, return_train_score=True)
    for v in scoring.keys():
        results[f"{k}-test_{v}_mean"] = np.mean(cv_results[f"test_{v}"])
        results[f"{k}-test_{v}_std"] = np.std(cv_results[f"test_{v}"])
    for v in scoring.keys():
        results[f"{k}-train_{v}_mean"] = np.mean(cv_results[f"train_{v}"])
        results[f"{k}-train_{v}_std"] = np.std(cv_results[f"train_{v}"])

    y_true, predicts[k] = Y_test, loaded_model.predict(X_test)
    predicted_data.iloc[:,-1] = predicts[k]
    plot_performance(raw_data.iloc[i_test,-1], re_standardise(raw_data.iloc[i_test,:],predicted_data), k)
    


with open("results.json", "w") as outfile:
    json.dump(results, outfile)