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
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
t0 = time.time() # Initial time

#---------------------------  MAIN CONFIGURATIONS -----------------------------

##----------------------- PATHS AND LOADING SETTINGS----------------------------
# Change these paths to ones in your own machine!
MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/' # Project main directory path
RESULTS_PATH = MAIN_PATH + 'Results/'

##--------------------------- MASTER SWITCHES -----------------------------------
fPlotSHAP = True # if True, plots SHAP values (bar chart, beeswarm and dependence plots)
fPlotOvertraining = True # if True, plots train vs test BDT output per class, with KS p-values
fPlotLearningCurves = True # if True, plots train and validation logloss/AUC vs boosting iteration
fPlotAUCVspT = True  # if True, plots AUC Vs pT
fPlotConfusionMatrix = True # if True, plots confusion matrix vs pT
BestThreshold = 0.5

SignalLabel = r'MC true $\Lambda^*$'  # class 1
BkgLabel = 'data mass sideband'       # class 0

# Features that get a SHAP dependence plot, and the column colouring the points
# (any feature, or fLStarPt, fLStarMass, fIsLStar)
SHAPDependenceFeatures = ['fOPAngle', 'fPhotonDCAPosPV', 'fPhotonDCANegPV', 'fLambdaDCAPosPV', 'fLambdaDCANegPV']
SHAPDependenceColor = 'fLStarPt'
SentinelValue = 900 # |value| above this is the 999 "not available" marker (photon daughter DCA to PV)

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
# OVERTRAINING CHECK: BDT output on the train and test sets of 03_MLTrainTest.py, per class.
# Uses Predictions.parquet (03's own test set) whatever DatasetTestName is.
if fPlotOvertraining:
    if not os.path.exists(RUN_PATH+"/PredictionsTrain.parquet"):
        print('[WARNING]: no PredictionsTrain.parquet in {}: rerun 03_MLTrainTest.py for the overtraining check'.format(RUN_PATH))
    else:
        TrainDF = pd.read_parquet(RUN_PATH+"/PredictionsTrain.parquet")
        TestDF = pd.read_parquet(RUN_PATH+"/Predictions.parquet")

        AUCTrain = roc_auc_score(TrainDF['GroundTruth'], TrainDF['Prediction'])
        AUCTest = roc_auc_score(TestDF['GroundTruth'], TestDF['Prediction'])

        bins = np.linspace(0, 1, 51)
        bin_centers = 0.5*(bins[1:] + bins[:-1])

        fig, ax = plt.subplots(figsize=(8, 6))
        for GroundTruth, Label, Color in [(0, BkgLabel, 'blue'), (1, SignalLabel, 'red')]:
            ClassName = 'signal' if GroundTruth else 'background'
            ScoresTrain = TrainDF[TrainDF.GroundTruth==GroundTruth].Prediction.values
            ScoresTest = TestDF[TestDF.GroundTruth==GroundTruth].Prediction.values
            KS = ks_2samp(ScoresTrain, ScoresTest) # unbinned
            print('[INFO]: overtraining check, {}: KS D = {:.4f}, p = {:.3g} (N train = {}, N test = {})'.format(
                ClassName, KS.statistic, KS.pvalue, len(ScoresTrain), len(ScoresTest)))

            # Test: filled histogram. Train: points with Poisson errors. Both normalised to unit area.
            ax.hist(ScoresTest, bins=bins, density=True, histtype='stepfilled', color=Color, alpha=0.3,
                    label='{} (test)'.format(Label))
            Counts, _ = np.histogram(ScoresTrain, bins=bins)
            Norm = len(ScoresTrain)*np.diff(bins)
            Filled = Counts > 0 # empty bins can't be drawn on a log scale
            ax.errorbar(bin_centers[Filled], (Counts/Norm)[Filled], yerr=(np.sqrt(Counts)/Norm)[Filled],
                        fmt='o', color=Color, markersize=4, elinewidth=1,
                        label='{} (train), KS p = {:.3g}'.format(Label, KS.pvalue))

        ax.set_yscale('log')
        ax.set_ylim(top=ax.get_ylim()[1]*30) # headroom for the legend
        ax.set_xlim(0, 1)
        ax.set_xlabel('BDT output (P)', fontsize=15)
        ax.set_ylabel('Normalised counts', fontsize=15)
        ax.set_title('Overtraining check: AUC train = {:.4f}, test = {:.4f}'.format(AUCTrain, AUCTest), fontsize=13)
        ax.grid(linestyle='--', alpha=0.6)
        ax.legend(loc='upper center', fontsize=10)
        plt.savefig(RUN_PATH+'/Overtraining.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)

#--------------------------------------
# LEARNING CURVES: train and validation metrics vs boosting iteration (eval_set of 03_MLTrainTest.py)
if fPlotLearningCurves:
    if not os.path.exists(RUN_PATH+"/LearningCurves.parquet"):
        print('[WARNING]: no LearningCurves.parquet in {}: rerun 03_MLTrainTest.py for the learning curves'.format(RUN_PATH))
    else:
        LearningCurvesDF = pd.read_parquet(RUN_PATH+"/LearningCurves.parquet")
        Metrics = [c[len('Train_'):] for c in LearningCurvesDF.columns if c.startswith('Train_')]

        # Iteration where the validation logloss stops improving: well before the last one means
        # n_estimators is larger than the data supports
        BestRow = LearningCurvesDF['Val_logloss'].idxmin()
        BestIteration = LearningCurvesDF['Iteration'][BestRow]
        print('[INFO]: validation logloss is minimal at iteration {} of {} ({:.4f}, {:.4f} at the last iteration)'.format(
            BestIteration, len(LearningCurvesDF), LearningCurvesDF['Val_logloss'][BestRow], LearningCurvesDF['Val_logloss'].iloc[-1]))

        fig, axes = plt.subplots(1, len(Metrics), figsize=(6*len(Metrics), 5), squeeze=False)
        for ax, Metric in zip(axes[0], Metrics):
            ax.plot(LearningCurvesDF['Iteration'], LearningCurvesDF['Train_'+Metric], color='blue', label='Train')
            ax.plot(LearningCurvesDF['Iteration'], LearningCurvesDF['Val_'+Metric], color='red', label='Validation')
            ax.axvline(BestIteration, linestyle='--', color='grey', label='Min. validation logloss (iteration {})'.format(BestIteration))
            ax.set_xlabel('Boosting iteration', fontsize=15)
            ax.set_ylabel({'logloss': 'Log loss', 'auc': 'AUC'}.get(Metric, Metric), fontsize=15)
            ax.grid(linestyle='--', alpha=0.6)
            ax.legend(fontsize=10)
        fig.suptitle('Learning curves', fontsize=15)
        plt.tight_layout()
        plt.savefig(RUN_PATH+'/LearningCurves.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)

#--------------------------------------
# FEATURE IMPORTANCE
if fPlotSHAP:
    #Open shap values
    SHAPDF = pd.read_parquet(RUN_PATH+"/SHAPvalues.parquet")
    SHAP_Array = np.array(SHAPDF)
    FeaturesToTrain = SHAPDF.columns.tolist()
    # Mean |SHAP| per feature
    shap.summary_plot(SHAP_Array, features=FeaturesToTrain, feature_names=FeaturesToTrain, plot_type='bar', plot_size=(10,5), show=False)

    # Save it
    plt.savefig(RUN_PATH+"/shap_bar.png", bbox_inches='tight', dpi=300)

    plt.show()
    plt.clf()

    if not os.path.exists(RUN_PATH+"/SHAPFeatures.parquet"):
        print('[WARNING]: no SHAPFeatures.parquet in {}: rerun 03_MLTrainTest.py for the SHAP beeswarm and dependence plots'.format(RUN_PATH))
    else:
        SHAPFeaturesDF = pd.read_parquet(RUN_PATH+"/SHAPFeatures.parquet")
        if len(SHAPFeaturesDF) != len(SHAPDF):
            raise ValueError('[ERROR]: SHAPFeatures.parquet ({} rows) and SHAPvalues.parquet ({} rows) are not row-aligned'.format(
                len(SHAPFeaturesDF), len(SHAPDF)))

        # Beeswarm: colour = feature value. The 999 marker would squash the colour scale of the
        # photon DCAs, so it is set to NaN, which SHAP draws in grey.
        FeatureValues = SHAPFeaturesDF[FeaturesToTrain]
        FeatureValues = FeatureValues.where(FeatureValues.abs() < SentinelValue)
        shap.summary_plot(SHAP_Array, features=FeatureValues, feature_names=FeaturesToTrain, plot_size=(10,7), show=False)
        plt.title('SHAP beeswarm (grey: 999 "not available")')
        plt.savefig(RUN_PATH+"/shap_beeswarm.png", bbox_inches='tight', dpi=300)
        plt.show()
        plt.clf()

        # Dependence plots: SHAP value vs feature value, one panel per feature
        DependenceFeatures = [f for f in SHAPDependenceFeatures if f in FeaturesToTrain]
        if DependenceFeatures:
            ColorValues = SHAPFeaturesDF[SHAPDependenceColor].values.astype(float)
            ColorMin, ColorMax = np.percentile(ColorValues, [5, 95])
            ColorMap = LinearSegmentedColormap.from_list('Blues_truncated', plt.cm.Blues(np.linspace(0.3, 1.0, 256))) # light end stays visible

            nCols = min(3, len(DependenceFeatures))
            nRows = int(np.ceil(len(DependenceFeatures)/nCols))
            fig, axes = plt.subplots(nRows, nCols, figsize=(6*nCols, 4.5*nRows), squeeze=False, constrained_layout=True)
            for ax, Feature in zip(axes.flat, DependenceFeatures):
                x = SHAPFeaturesDF[Feature].values
                y = SHAPDF[Feature].values
                IsSentinel = np.abs(x) > SentinelValue
                Points = ax.scatter(x[~IsSentinel], y[~IsSentinel], c=ColorValues[~IsSentinel], cmap=ColorMap,
                                    vmin=ColorMin, vmax=ColorMax, s=4, rasterized=True)
                ax.axhline(0, linestyle='--', color='grey', linewidth=0.8)
                if Feature.endswith('PV'):
                    ax.axvline(0, linestyle='--', color='grey', linewidth=0.8) # sign of the signed DCA

                # x range: 0.5-99.5 % quantiles, so a few far tails don't squash the bulk
                xLow, xHigh = np.percentile(x[~IsSentinel], [0.5, 99.5])
                ax.set_xlim(xLow - 0.05*(xHigh - xLow), xHigh + 0.05*(xHigh - xLow))
                ax.set_xlabel(Feature, fontsize=13)
                ax.set_ylabel('SHAP value (log-odds)', fontsize=13)
                ax.grid(linestyle='--', alpha=0.6)
                if IsSentinel.any(): # what the model does with the "not available" marker (not drawn)
                    ax.set_title('999 (n/a, not drawn): {:.1f}% of candidates, mean SHAP {:+.2f}'.format(
                                 100*IsSentinel.mean(), y[IsSentinel].mean()), fontsize=10)
            for ax in axes.flat[len(DependenceFeatures):]:
                ax.set_visible(False)
            fig.colorbar(Points, ax=axes, label=SHAPDependenceColor, fraction=0.03)
            fig.suptitle('SHAP dependence (colour: {}, test set)'.format(SHAPDependenceColor), fontsize=15)
            plt.savefig(RUN_PATH+"/shap_dependence.png", bbox_inches='tight', dpi=300)
            plt.show()
            plt.close(fig)


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
