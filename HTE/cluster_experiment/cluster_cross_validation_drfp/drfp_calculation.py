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
from drfp import DrfpEncoder

# Load the datasets

df_hte_train = pd.read_csv('df_hte_cluster_drfp_14.csv')


delimiter = '.'
df_hte_train['rxn'] = df_hte_train.apply(lambda row: delimiter.join(row[['CONDITIONS', 'NO_MAP_NO_COND']]), axis=1)
df_hte_train.dropna(subset = ['rxn'], inplace=True)


hte_drfp_train = DrfpEncoder.encode(df_hte_train['rxn'], show_progress_bar=False)

joblib.dump(hte_drfp_train, "df_hte_cluster_labels_drfp.pickle")

