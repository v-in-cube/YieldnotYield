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

us_drfp = DrfpEncoder.encode(df_us['rxn'], show_progress_bar=True)
joblib.dump(us_drfp, "uspto_drfp.pickle")

rx_drfp = DrfpEncoder.encode(df_rx['rxn'], show_progress_bar=True)
joblib.dump(rx_drfp, "reaxys_drfp.pickle")

hte_drfp = DrfpEncoder.encode(df_hte['rxn'], show_progress_bar=True)
joblib.dump(hte_drfp, "bh_hte_drfp.pickle")

az_drfp = DrfpEncoder.encode(df_az['rxn'], show_progress_bar=True)
joblib.dump(az_drfp, "az_eln_drfp.pickle")