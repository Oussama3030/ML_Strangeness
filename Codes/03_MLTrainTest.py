# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https:#alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
#
# ML training/test code
# ================
#
# This code train and test a BDT model with the best set of hiperparameters
#
#    Comments, questions, complaints, suggestions?
#    Please write to:
#    gianni.shigeru.setoue.liveraro@cern.ch
#    romain.schotter@cern.ch
#    david.dobrigkeit.chinellato@cern.ch

#____________________________
# Imports
from __future__ import division
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import binarize
from xgboost import XGBClassifier
import xgboost as xgb
import shap
import onnx
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import pickle
import json
import joblib
import time
t0 = time.time() # Initial time

#---------------------------  MAIN CONFIGURATIONS -----------------------------

##----------------------- PATHS AND LOADING SETTINGS----------------------------
MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/' # Project main directory path
RESULTS_PATH = MAIN_PATH + 'Results/'

print('Which ML Run would you like to load?. Available Runs: \n', os.listdir(RESULTS_PATH))
RunNumber = str (input())
RUN_PATH = RESULTS_PATH+'Run {}'.format(RunNumber)

# Loading dict with predictions
with open(RUN_PATH+"/RunConfig.pkl", 'rb') as f:
    RunConfig = pickle.load(f)

# ##--------------------------------- DATASET ----------------------------------
DatasetName = RunConfig['Dataset']['DatasetName']
FeaturesToTrain = RunConfig['Dataset']['Features']
Class_name = RunConfig['Dataset']['Class']

# Loading train test
DatasetTrain = pd.read_parquet(MAIN_PATH+'Dataset/Processed/{}_Train.parquet'.format(DatasetName))
X_train = DatasetTrain[FeaturesToTrain]
y_train = DatasetTrain[[Class_name]]

# Loading test test
DatasetTest = pd.read_parquet(MAIN_PATH+'Dataset/Processed/{}_Test.parquet'.format(DatasetName))
X_test = DatasetTest[FeaturesToTrain]
y_test = DatasetTest[[Class_name]]

# Loading validation set: only monitored during training (learning curves), never fitted on
DatasetVal = pd.read_parquet(MAIN_PATH+'Dataset/Processed/{}_Val.parquet'.format(DatasetName))
X_val = DatasetVal[FeaturesToTrain]
y_val = DatasetVal[[Class_name]]

# ##--------------------------------- TRAINING ----------------------------------
BDT_Classifier = XGBClassifier(**RunConfig['BDT'])
BDT_Classifier.set_params(eval_metric=['logloss', 'auc']) # monitoring only: no early stopping, the fit is unchanged
BDT_Classifier.fit(X_train.values, y_train.values.ravel(),
                   eval_set=[(X_train.values, y_train.values.ravel()), (X_val.values, y_val.values.ravel())],
                   verbose=False)

# ##---------------------------- LEARNING CURVES --------------------------------
# Train and validation metrics after each boosting iteration, plotted in 04_MLAnalysis.py
EvalsResult = BDT_Classifier.evals_result()
LearningCurvesDF = pd.DataFrame({'Iteration': np.arange(1, len(EvalsResult['validation_0']['logloss'])+1)})
for SetName, SetKey in [('Train', 'validation_0'), ('Val', 'validation_1')]:
    for Metric, Values in EvalsResult[SetKey].items():
        LearningCurvesDF['{}_{}'.format(SetName, Metric)] = Values
LearningCurvesDF.to_parquet(RUN_PATH+"/LearningCurves.parquet")

# ##------------------------------- SAVING MODEL --------------------------------
# Save trained model in JSON format
BDT_Classifier.save_model(RUN_PATH+"/{}_BDTModel.json".format(DatasetName))

# Save trained model in ONNX format
initial_type = [('float_input', FloatTensorType([None, X_train.values.shape[1]]))]
onnx_model = convert_xgboost(BDT_Classifier, initial_types=initial_type, target_opset=12)
onnx_model.graph.doc_string = "Converted from XGBoost ver."+xgb.__version__
onnx.checker.check_model(onnx_model)
with open(RUN_PATH+'/{}_BDTModel.onnx'.format(DatasetName), "wb") as f:
    f.write(onnx_model.SerializeToString())
f.close()
#exit()

print("Mean Absolute Error (Train sample):", mean_absolute_error(y_train.values.ravel(), BDT_Classifier.predict(X_train)))
print("Mean Absolute Error (Test sample):", mean_absolute_error(y_test.values.ravel(), BDT_Classifier.predict(X_test)))

# ##--------------------------------- TEST --------------------------------------
# Predictions:
PredictionProb = BDT_Classifier.predict_proba(X_test)

PredictionsDF = pd.DataFrame({
    'fIsLStar':DatasetTest['fIsLStar'].values.ravel(), 
    'fLStarPt':DatasetTest['fLStarPt'].values.ravel(), 
    'Prediction':PredictionProb[:,1], 
    'GroundTruth':y_test.values.ravel()})

# Saving predictions for analysis:
PredictionsDF.to_parquet(RUN_PATH+"/Predictions.parquet")

# Train-set predictions, for the overtraining check (train vs test output) in 04_MLAnalysis.py
PredictionProbTrain = BDT_Classifier.predict_proba(X_train)

PredictionsTrainDF = pd.DataFrame({
    'fIsLStar':DatasetTrain['fIsLStar'].values.ravel(),
    'fLStarPt':DatasetTrain['fLStarPt'].values.ravel(),
    'Prediction':PredictionProbTrain[:,1],
    'GroundTruth':y_train.values.ravel()})

PredictionsTrainDF.to_parquet(RUN_PATH+"/PredictionsTrain.parquet")

# ##------------------------------- SHAP VALUES --------------------------------
# SHAP values for feature importance
explainer = shap.TreeExplainer(BDT_Classifier)
shap_values = explainer.shap_values(X_test)
# save shap values
shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
shap_values_df.to_parquet(RUN_PATH+"/SHAPvalues.parquet")

# Feature values of the same test candidates, row-aligned with SHAPvalues.parquet (beeswarm and
# dependence plots need them). Class, pT and mass are not features: kept to colour the dependence plots.
SHAPColumns = FeaturesToTrain + [c for c in [Class_name, 'fLStarPt', 'fLStarMass'] if c not in FeaturesToTrain]
DatasetTest[SHAPColumns].reset_index(drop=True).to_parquet(RUN_PATH+"/SHAPFeatures.parquet")

# End
t1 = time.time() - t0
print("_________________________________________")
print("Total time elapsed (min): ", t1/60)
print("_________________________________________")

