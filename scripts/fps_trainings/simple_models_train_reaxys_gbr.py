import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

# Load the datasets
df_us = pd.read_csv('../df_us_bh_rxnfp_mhfp6_no_dupl.csv')
df_rx = pd.read_csv('../df_rx_bh_rxnfp_mhfp6_no_dupl.csv')
df_hte = pd.read_csv('../df_hte_bh_rxnfp_mhfp6_no_dupl.csv')
df_az = pd.read_csv('../df_az_bh_rxnfp_mhfp6_no_dupl.csv')

delimiter = '.'
df_us['rxn'] = df_us.apply(lambda row: delimiter.join(row[['CONDITIONS', 'NO_MAP_NO_COND']]), axis=1)
df_us.dropna(subset = ['rxn'], inplace=True)
df_rx['rxn'] = df_rx.apply(lambda row: delimiter.join(row[['CONDITIONS', 'NO_MAP_NO_COND']]), axis=1)
df_rx.dropna(subset = ['rxn'], inplace=True)
df_hte['rxn'] = df_hte.apply(lambda row: delimiter.join(row[['CONDITIONS', 'NO_MAP_NO_COND']]), axis=1)
df_hte.dropna(subset = ['rxn'], inplace=True)
df_az['rxn'] = df_az.apply(lambda row: delimiter.join(row[['CONDITIONS', 'NO_MAP_NO_COND']]), axis=1)
df_az.dropna(subset = ['rxn'], inplace=True)

scaler = StandardScaler()
# Split the datasets into features and target variables
X4 = np.array([json.loads(x) for x in df_az["RXNFP_FP"]])
scaler.fit(X4)
X4 = scaler.transform(X4)
X4_2 = joblib.load("az_eln_drfp.pickle")
scaler.fit(X4_2)
X4_2 = scaler.transform(X4_2)
y4 = np.array(df_az.YIELD)


X1 = np.array([json.loads(x) for x in df_us["RXNFP_FP"]])
scaler.fit(X1)
X1 = scaler.transform(X1)
X1_2 = joblib.load("uspto_drfp.pickle")
scaler.fit(X1_2)
X1_2 = scaler.transform(X1_2)
y1 = np.array(df_us.YIELD)



X2 = np.array([json.loads(x) for x in df_rx["RXNFP_FP"]])
scaler.fit(X2)
X2 = scaler.transform(X2)
X2_2 = joblib.load("reaxys_drfp.pickle")
scaler.fit(X2_2)
X2_2 = scaler.transform(X2_2)
y2 = np.array(df_rx.YIELD)



X3 = np.array([json.loads(x) for x in df_hte["RXNFP_FP"]])
scaler.fit(X3)
X3 = scaler.transform(X3)
X3_2 = joblib.load("bh_hte_drfp.pickle")
scaler.fit(X3_2)
X3_2 = scaler.transform(X3_2)
y3 = np.array(df_hte.YIELD)


print("Data was loaded")


def evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, model_name, train_dataset_name, test_dataset_name, fps_name):
    print(f"{model_name}_{train_dataset_name}_{test_dataset_name}_{fps_name}")
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=30, verbose =2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - {train_dataset_name} - {test_dataset_name} - {fps_name}\nTrue vs Predicted')
    plt.savefig(f'{model_name}_{train_dataset_name}_{test_dataset_name}_{fps_name}_plot.png')
    plt.close()

    results = {
        'model_name': model_name,
        'train_dataset_name': train_dataset_name,
        'test_dataset_name': test_dataset_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'best_params': best_model.get_params()
    }

    with open(f'{model_name}_{train_dataset_name}_{test_dataset_name}_{fps_name}_results.json', 'w') as file:
        json.dump(results, file, indent=4)
        
    model_filename = f'{model_name}_{train_dataset_name}_{test_dataset_name}_{fps_name}_model.pkl'
    joblib.dump(best_model, model_filename)
    return best_model, r2, rmse, mae

def test(model, X, y, best_params, train_dataset_name, test_dataset_name, fps_name, model_name):
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - {train_dataset_name} - {test_dataset_name} - {fps_name}\nTrue vs Predicted')
    plt.savefig(f'{model_name}_{train_dataset_name}_{test_dataset_name}_{fps_name}_plot.png')
    plt.close()

    results = {
        'model_name': model_name,
        'test_dataset_name': test_dataset_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'best_params': best_params
    }

    with open(f'{model_name}_{train_dataset_name}_{test_dataset_name}_{fps_name}_results.json', 'w') as file:
        json.dump(results, file, indent=4)

    return r2, rmse, mae

param_grid_rfr = {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 80, 90],
    'min_samples_split': [2, 5, 20]
}

param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.01, 0.001]
}

param_grid_gbr = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
X2_2train, X2_2test, y2_2train, y2_2test = train_test_split(X2_2, y2, test_size=0.2, random_state=42)

# Create instances of the models

gbr = GradientBoostingRegressor(random_state=42)


best_gbr2, r2_gbr2, rmse_gbr2, mae_gbr2 = evaluate_model(gbr, param_grid_gbr, X2_train, y2_train, X2_test, y2_test, 'GBR', 'reaxys', 'reaxys', 'rxnfp')
best_gbr2_2, r2_gbr2_2, rmse_gbr2_2, mae_gbr2_2 = evaluate_model(gbr, param_grid_gbr, X2_2train, y2_2train, X2_2test, y2_2test, 'GBR', 'reaxys', 'reaxys', 'drfp')


#best_gbr2 = joblib.load("GBR_reaxys_reaxys_rxnfp_model.pkl")
#best_gbr2_2 = joblib.load("GBR_reaxys_reaxys_drfp_model.pkl")

r2_eval_gbr_3, rmse_eval_gbr_3, mae_eval_gbr_3 = test(best_gbr2, X3, y3, best_gbr2.get_params(), 'reaxys', 'hte_bh', 'RXNFP', 'GBR')
r2_eval_gbr_3_2, rmse_eval_gbr_3_2, mae_eval_gbr_3_2 = test(best_gbr2_2, X3_2, y3, best_gbr2_2.get_params(), 'reaxys', 'hte_bh', 'DRFP', 'GBR')



r2_eval_gbr_4, rmse_eval_gbr_4, mae_eval_gbr_4 = test(best_gbr2, X4, y4, best_gbr2.get_params(), 'reaxys', 'az_eln', 'RXNFP', 'GBR')
r2_eval_gbr_4_2, rmse_eval_gbr_4_2, mae_eval_gbr_4_2 = test(best_gbr2_2, X4_2, y4, best_gbr2_2.get_params(), 'reaxys', 'az_eln', 'DRFP', 'GBR')


r2_eval_gbr_1, rmse_eval_gbr_1, mae_eval_gbr_1 = test(best_gbr2, X1, y1, best_gbr2.get_params(), 'reaxys', 'uspto', 'RXNFP', 'GBR')
r2_eval_gbr_1_2, rmse_eval_gbr_1_2, mae_eval_gbr_1_2 = test(best_gbr2_2, X1_2, y1, best_gbr2_2.get_params(), 'reaxys', 'uspto', 'DRFP', 'GBR')
