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
# ML Test only code
# ================
#
# This code loads a trained BDT model and tests it with a test set.
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
import onnx
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
import pickle
import json
import joblib
import time
t0 = time.time() # Initial time

##----------------------- PATHS AND LOADING SETTINGS----------------------------
MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/'
RESULTS_PATH = MAIN_PATH + 'Results/'

print('Which ML Run would you like to load?. Available Runs: \n', os.listdir(RESULTS_PATH))
RunNumber = str (input())
RUN_PATH = RESULTS_PATH+'Run {}'.format(RunNumber)
# Loading dict with predictions
with open(RUN_PATH+"/RunConfig.pkl", 'rb') as f:
    RunConfig = pickle.load(f) 

# ##--------------------------------- DATASET ----------------------------------
DatasetName = RunConfig['Dataset']['DatasetName']
DatasetTestName = 'LambdaStar_Test'
FeaturesToTrain = RunConfig['Dataset']['Features']
Class_name = RunConfig['Dataset']['Class']

# Loading test test
DatasetTest = pd.read_parquet(MAIN_PATH + 'Dataset/Processed/{}.parquet'.format(DatasetTestName))
X_test = DatasetTest[FeaturesToTrain]
y_test = DatasetTest[[Class_name]]

# ##------------------------------- OPENING MODEL --------------------------------
# Load trained model in JSON format
BDT_Classifier = xgb.XGBClassifier()
BDT_Classifier.load_model(RUN_PATH+"/{}_BDTModel.json".format(DatasetName))

# ##--------------------------------- TEST --------------------------------------
# Predictions:
PredictionProb = BDT_Classifier.predict_proba(X_test)
PredictionsDF = pd.DataFrame({'fLStarPt':DatasetTest['fLStarPt'].values.ravel(), 'Prediction':PredictionProb[:,1], 'GroundTruth':y_test.values.ravel()})

# Saving predictions for analysis:
PredictionsDF.to_parquet(RUN_PATH+"/Predictions_{}.parquet".format(DatasetTestName))

# End
t1 = time.time() - t0
print("_________________________________________")
print("Total time elapsed (min): ", t1/60)
print("_________________________________________")

