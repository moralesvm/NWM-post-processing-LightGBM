# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:05:27 2023

@author: Mirce Morales

Objective: ML post-processing correction model to improve NWM hourly streamflow
predictions over the Sleepers River W5 catchment in Vermont, US

Model 1 [LightGBM]: LightGBM without antecedent water level
Model 2 [LightGBM_WL]: LightGBM including antecedent water level

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from sklearn.pipeline import Pipeline

# Libraries to apply LightGBM
import lightgbm
from lightgbm import LGBMRegressor
from sklearn import set_config
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
set_config(transform_output="pandas") 

# Main directories
dataPath = './Input/'
Path = "./Output/"

# Specify model to perform
modelID = "M2"

# Load data:
path_dict = {
    "M1": f'{dataPath}ml_features.csv', # Dataset for Model 1
    "M2": f'{dataPath}ml_wWL_features.csv' # Dataset for Model 2
    }   

filename =  path_dict[modelID]
final_features = pd.read_csv(filename, index_col=["dateTime"], dtype={'siteID': 'string',})
# Cast datetime variable in datetime format.
final_features.index = pd.to_datetime(final_features.index,format='mixed')
final_features = final_features.sort_values(by=["siteID", "dateTime"]) # Sort to ensure the time series groups are contiguous. 

#---------------------------------
# Select the siteID to work with
#---------------------------------
final_features['siteID'].unique()
site = '01135300'
final_features = final_features.loc[final_features['siteID'] == site]

final_features['siteID'].unique()
df_temp = final_features

# Create a directory to save the results if it doesn't exist already
savePath = Path + site
if not os.path.exists(savePath):
   os.makedirs(savePath)
   
#----------------------
# Periodic features
#----------------------
dtf = DatetimeFeatures(
    # the datetime variable
    variables="index",

    # the features we want to create
    features_to_extract=[
        #"year",
        "month",
        #"day_of_month",
    ],
)

# Imputer for dropping missing data
#----------------------------------
# Drop missing data
imputer = DropMissingData(missing_only=False)

#----------------------------------
# DEFINE MODEL
#----------------------------------
# Define the dataset to work with
df_ml = df_temp
# Sanity check: data span.
print(df_ml.index.min(), df_ml.index.max())

#----------------------------------
# DEFINE PIPELINE
#----------------------------------
# Features to drop
to_drop = ["siteID","nwm_precip_window_8_sum","q_nwm_window_4_mean","nwm_precip_window_24_exp_weighted_mean",
           "Basin_Slope","Basin_area","Basin_Relief","Elongation_Ratio","streamorder",
           "Longest_Flowpath_Length_mi", "Longest_Flow_path_Slope"]

df_ml = df_ml.drop(columns=to_drop)
        
pipe = Pipeline(
    [("datetime_features", dtf),
     ("dropna", imputer),])

# To double check column names or features
print(list(df_ml.columns.values))

#-----------------------------------------
# APPLY ML MODEL
#-----------------------------------------
# Define our target value
target = ["q_obs"]

# Rescale our features
pipeline = make_pipeline(pipe,MinMaxScaler())

# Let's see the pipeline to double-check
pipeline

# Let's check how our feature engineering pipeline behaves
pipeline.fit_transform(df_ml)
# We can use `clone` to return an unfitted version
# of the pipeline.
pipeline = clone(pipeline)

# --- CONFIG --- #
# Define time of first prediction, this determines our train / test split
prediction_start_time = pd.to_datetime("2016-01-01 00:00:00")

# Define the model.
model = LGBMRegressor(
                      boosting = "gbdt",
                      linear_tree=True, 
                      linear_lambda=0.1, 
                      n_estimators=300,
                      importance_type='split',
                     )
                                                    
# --- CREATE TRAINING & TESTING DATAFRAME  --- #
# Ensure we only have training data up to the start
# of the prediction.
df_train = df_ml.loc[df_ml.index < prediction_start_time].copy()
df_test = df_ml.loc[df_ml.index >= prediction_start_time].copy()

# --- FEATURE ENGINEERING--- #
# Create X_train and y_train
y_train = df_train.dropna()[target]
X_train = pipeline.fit_transform(df_train).drop(columns=["q_obs"])

# --- MODEL TRAINING---#
model.fit(X_train, y_train)

#----- PREDICT USING ALL THE TESTING PERIOD----#
X_test = pipeline.transform(df_test).drop(columns=["q_obs"])
# Predict one step ahead.
y_pred = model.predict(X_test)
    
# --- GET PREDICTION AND TEST VALUES --- #
y_test = df_test.dropna()[target]

# Save the results to postprocess and assess results
#---------------------------------------------------------
df_y_pred = df_test.dropna()[["q_nwm","q_obs"]]
df_y_pred['y_pred'] = y_pred.tolist()
df_y_pred.to_csv(f'{savePath}/df_y_pred_{modelID}_{site}_.csv')

# See feature importance
#------------------------
ax = lightgbm.plot_importance(model)
figure = plt.gcf()
figure.set_size_inches(7, 8)
plt.savefig(f'{savePath}/FImportance_{modelID}_{site}.png',dpi=(100), bbox_inches='tight')









