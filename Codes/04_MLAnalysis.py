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
# ML analysis code
# ================
#
# BDT Performance Analysis using the test set predictions
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_absolute_error, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import binarize
import pickle
import json
import time
import shap
from matplotlib import pyplot as plt
t0 = time.time() # Initial time

#---------------------------  MAIN CONFIGURATIONS -----------------------------

##----------------------- PATHS AND LOADING SETTINGS----------------------------
# Change these paths to ones in your own machine!
MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/' # Project main directory path
RESULTS_PATH = MAIN_PATH + 'Results/'

##--------------------------- MASTER SWITCHES -----------------------------------
fPlotSHAP = True # if True, plots SHAP values
fPlotAUCVspT = True  # if True, plots AUC Vs pT 
fPlotConfusionMatrix = True # if True, plots confusion matrix vs pT
BestThreshold = 0.5

fUseAnotherTestDataset = False # if True, uses another test dataset (not the one used in the training). If False, uses the same test dataset used in the training.
DatasetTestName = 'LambdaStar' 

if not fUseAnotherTestDataset: DatasetTestName = '' # if you want to use the same test dataset used in the training, leave it empty

print('Which ML Run would you like to load?. Available Runs: \n', os.listdir(RESULTS_PATH))
RunNumber = str (input())
RUN_PATH = RESULTS_PATH+'Run {}'.format(RunNumber)
# Loading dict with predictions
with open(RUN_PATH+"/RunConfig.pkl", 'rb') as f:
    RunConfig = pickle.load(f)


PredictionsDF = pd.read_parquet(RUN_PATH+"/Predictions{}.parquet".format(DatasetTestName))

print('N signal candidates: ', len(PredictionsDF[PredictionsDF.GroundTruth==1]))
print('N bkg candidates: ', len(PredictionsDF[PredictionsDF.GroundTruth==0]))

# Calculating True Positive Rate (TPR) and False Positive Rate (FPR) for different threshold values
FPR, TPR, thresholds = roc_curve(PredictionsDF['GroundTruth'].values.ravel(), PredictionsDF['Prediction'].values, drop_intermediate=False) 

#--------------------------------------
# ROC Curve:
## Area Under the Curve
auc = roc_auc_score(PredictionsDF['GroundTruth'].values.ravel(), PredictionsDF['Prediction'].values) 

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(FPR, TPR, label="AUC ="+np.format_float_positional(auc, precision=3))
ax.plot([0,1], [0,1], linestyle='--', label='Random Selection, AUC = 0.5')
ax.set_xlabel('False Positive Rate', fontsize=15)
ax.set_ylabel('True Positive Rate', fontsize=15)
ax.set_title('BDT ROC Curve', fontsize=15)
ax.grid(linestyle='--')
ax.legend(loc='lower right')
plt.savefig(RUN_PATH+'/ROC Curve{}.png'.format(DatasetTestName), bbox_inches='tight')
plt.show()

plt.clf() 

#--------------------------------------
# Plot TPR x Threshold
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(thresholds, TPR)
ax.set_xlabel('Decision Threshold', fontsize=15)
ax.set_ylabel('True Positive Rate', fontsize=15)
ax.set_title('TPR x Decision Threshold', fontsize=15)
ax.grid(linestyle='--')
ax.set_xlim((0,1.0))
plt.savefig(RUN_PATH+'/TPR{}.png'.format(DatasetTestName), bbox_inches='tight')
plt.show()

plt.clf() 

#--------------------------------------
# Plot FPR x Threshold
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(thresholds, FPR)
ax.set_xlabel('Decision Threshold', fontsize=15)
ax.set_ylabel('False Positive Rate', fontsize=15)
ax.set_title('FPR x Decision Threshold', fontsize=15)
ax.grid(linestyle='--')
ax.set_xlim((0,1.0))
plt.savefig(RUN_PATH+'/FPR{}.png'.format(DatasetTestName), bbox_inches='tight')
plt.show()
plt.clf() 

# ML output histogram
fig = plt.figure()
plt.hist(PredictionsDF[PredictionsDF.GroundTruth==0].Prediction, label='Wrongly Associated', bins=100, color='blue', alpha=0.5)
plt.hist(PredictionsDF[PredictionsDF.GroundTruth==1].Prediction, label='Correctly Associated', bins=100, color='red', alpha=0.5)
plt.yscale('log')

plt.title('BDT output: candidates probability distribution')
plt.xlabel('BDT output (P)',fontsize='x-large')

plt.ylabel('Counts',fontsize='x-large')
plt.legend(fontsize='x-large')
ax.grid(linestyle='--')
#plt.figtext(0.70, 0.75, r"$\bf{ALICE\/ Performance}$" + '\n Pb-Pb at $\sqrt{s}=5.02$ $TeV$ \n $(0-100)$%', ha='center')
plt.savefig(RUN_PATH+'/ProbabilityPlot{}.png'.format(DatasetTestName), bbox_inches='tight')
plt.show()
plt.clf()

#--------------------------------------
# FEATURE IMPORTANCE
if fPlotSHAP:
    #Open shap values
    SHAPDF = pd.read_parquet(RUN_PATH+"/SHAPvalues.parquet")
    SHAP_Array = np.array(SHAPDF)
    FeaturesToTrain = SHAPDF.columns.tolist()
    # Create the plot
    shap.summary_plot(SHAP_Array, features=FeaturesToTrain, feature_names=FeaturesToTrain, plot_type='bar', plot_size=(10,5), show=False)
    #shap.summary_plot(shap_values, X_test, show=False)  # <--- Do not show immediately

    # Save it
    plt.savefig(RUN_PATH+"/shap_beeswarm.png", bbox_inches='tight', dpi=300)

    plt.show()
    plt.clf() 


# -----------------------------------
# Confusion matrix per pT 
if fPlotConfusionMatrix:
    pT_bins = [0.0, 30.0]

    for i in range(len(pT_bins)-1):
        pT_min = pT_bins[i]
        pT_max = pT_bins[i+1]
        
        # Filter predictions for the current pT bin
        
        PredictionsDF_pT_bin = PredictionsDF[(PredictionsDF['fLStarPt'] >= pT_min) & (PredictionsDF['fLStarPt'] < pT_max)]
        PredictionsDF_pT_bin["Classification"] = 0
        PredictionsDF_pT_bin["Classification"][PredictionsDF_pT_bin.Prediction>= BestThreshold] = 1

        cm = confusion_matrix(PredictionsDF_pT_bin['GroundTruth'].values.ravel(), PredictionsDF_pT_bin["Classification"].values.ravel())
        cm_percent = cm.astype('float') / cm.sum() * 100
        cm_percent = np.round(cm_percent, 2)
        
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bkg", "Signal"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix, {} < pT < {}, Threshold {}'.format(pT_min, pT_max, BestThreshold))

        plt.savefig(RUN_PATH+"/ConfusionMatrix_{}-{}.png".format(pT_min, pT_max), bbox_inches='tight', dpi=300)
        plt.clf()



#--------------------------------------
# AUC vs pT Plot
if fPlotAUCVspT:
    # Define list of pT bins
    pT_bins = [0.0,30.0]


    # Loop over pT bins and calculate AUC for each bin
    AUCs = []
    for i in range(len(pT_bins)-1):
        pT_min = pT_bins[i]
        pT_max = pT_bins[i+1]
        
        # Filter predictions for the current pT bin
        PredictionsDF_pT_bin = PredictionsDF[(PredictionsDF['fLStarPt'] >= pT_min) & (PredictionsDF['fLStarPt'] < pT_max)]
        
        if len(PredictionsDF_pT_bin) > 0:  # Check if there are predictions in this bin
            auc = roc_auc_score(PredictionsDF_pT_bin['GroundTruth'], PredictionsDF_pT_bin['Prediction'])
            AUCs.append(auc)
        else:
            AUCs.append(np.nan)  # If no predictions, append NaN

    # Compute bin centers and half-widths (for x-error bars)
    bin_centers = [(pT_bins[i] + pT_bins[i+1]) / 2 for i in range(len(pT_bins) - 1)]
    x_errors = [(pT_bins[i+1] - pT_bins[i]) / 2 for i in range(len(pT_bins) - 1)]

    # Convert to numpy arrays for consistency with matplotlib
    bin_centers = np.array(bin_centers)
    x_errors = np.array(x_errors)
    AUCs = np.array(AUCs)

    # Plot AUC vs pT with horizontal error bars
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        bin_centers,
        AUCs,
        xerr=x_errors,
        fmt='o',
        color=(0.1, 0.1, 0.9),     # ROOT-style blue
        ecolor=(0.1, 0.1, 0.9),
        elinewidth=1.5,
        capsize=3,
        markersize=4              # Smaller marker
    )

    # Labels
    ax.set_xlabel('pT (GeV/c)', fontsize=16)
    ax.set_ylabel('AUC', fontsize=16)

    # Ticks styling
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
    ax.tick_params(axis='both', which='minor', labelsize=10, direction='in', length=3)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Optional: set y limits
    # ax.set_ylim(0.5, 1.0)

    # Layout and save
    plt.tight_layout()
    plt.savefig(RUN_PATH + '/AUC_vs_pT.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
