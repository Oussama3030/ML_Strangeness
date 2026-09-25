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
# ProcessData
# ================
#
# This code reads a TTree and creates training and testing samples for ML
#
#    Comments, questions, complaints, suggestions?
#    Please write to:
#    gianni.shigeru.setoue.liveraro@cern.ch
#    romain.schotter@cern.ch
#    david.dobrigkeit.chinellato@cern.ch

#____________________________
# Imports
import numpy as np
import pandas as pd
import uproot
import os
from sklearn.model_selection import train_test_split
import time
t0 = time.time() # Initial time

#---------------------------  MAIN CONFIGURATIONS ----------------------------
fSplitTrainTest = False  # if true, perform train/test/validation split. 
fSeparateAO2Ds = True    # else if true, merge signal and bacgkround ao2ds, then perform perform train/test/validation split
                         # else, the entire input dataset is converted to an unique .parquet file

# Number of signal and bkg candidates in the total dataset (before splitting)
NSignal = 2600
NBkg = 20000


##--------------------------------- PATHS ------------------------------------
# Change these paths to ones in your own machine!
MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/'
TABLES_TO_KEEP = ['O2sigma0cores', 'O2sigma0lambda', 'O2sigma0photon', 'O2sigma0mccores'] # Tables with training variables that needs to be accessed e.g. O2tracks
  
##--------------------------------- DATASET ----------------------------------
DatasetName = 'Dataset/Raw/AO2DMerged.root' # .root TTree from O2Physics 
OutputPrefix = "LambdaStar"

SignalFile = 'Dataset/Raw/AO2DMerged_Sig.root'     # MC: signal = candidates PDG-matched to the L*
BackgroundFile = 'Dataset/Raw/AO2DMerged_Bkg.root' # data: background = candidates in the mass sidebands

Class_name = "fIsLStar" # Class name (target)

LStarPDG = 3124

# Background is taken from data OUTSIDE this invariant mass window
MassWindowMin = 1.45
MassWindowMax = 1.59

# Stop reading a file after this many raw candidates (0 = read it all).
NRowsSignalRead = 0
NRowsBkgRead = 2000000

def add_signal_lstar(data_frame):
  if 'fPDGCode' not in data_frame.columns: # data: no MC table, class is set by hand later
    data_frame[Class_name] = False
    return data_frame
  data_frame[Class_name] = np.abs(data_frame['fPDGCode']) == LStarPDG
  return data_frame

def add_opening_angle(data_frame):
  dot = (data_frame['fPhotonPx'] * data_frame['fLambdaPx']
         + data_frame['fPhotonPy'] * data_frame['fLambdaPy']
         + data_frame['fPhotonPz'] * data_frame['fLambdaPz'])
  magPhoton = np.sqrt(data_frame['fPhotonPx']**2 + data_frame['fPhotonPy']**2 + data_frame['fPhotonPz']**2)
  magLambda = np.sqrt(data_frame['fLambdaPx']**2 + data_frame['fLambdaPy']**2 + data_frame['fLambdaPz']**2)
  ptot = magPhoton * magLambda
  arg = np.divide(dot, ptot, out=np.zeros_like(dot, dtype=float), where=ptot > 0)
  arg = np.clip(arg, -1.0, 1.0)
  data_frame['fOPAngle'] = np.arccos(arg)
  return data_frame

def add_lstar_mass(data_frame):
  #   M = sqrt( (E_photon + E_Lambda)^2 - |p_photon + p_Lambda|^2 )
  MassPhoton = 0.0            
  MassLambda = 1.115683 
  ePhoton = np.sqrt(data_frame['fPhotonPx']**2 + data_frame['fPhotonPy']**2 + data_frame['fPhotonPz']**2 + MassPhoton**2)
  eLambda = np.sqrt(data_frame['fLambdaPx']**2 + data_frame['fLambdaPy']**2 + data_frame['fLambdaPz']**2 + MassLambda**2)
  E = ePhoton + eLambda
  px = data_frame['fPhotonPx'] + data_frame['fLambdaPx']
  py = data_frame['fPhotonPy'] + data_frame['fLambdaPy']
  pz = data_frame['fPhotonPz'] + data_frame['fLambdaPz']
  m2 = E**2 - (px**2 + py**2 + pz**2)
  data_frame['fLStarMass'] = np.sqrt(np.clip(m2, 0.0, None))
  return data_frame

def add_lstar_pt(data_frame):
  px = data_frame['fPhotonPx'] + data_frame['fLambdaPx']
  py = data_frame['fPhotonPy'] + data_frame['fLambdaPy']
  data_frame['fLStarPt'] = np.sqrt(px**2 + py**2)
  return data_frame

def filter_df(data_frame):
  """
  Filter columns; e.g.
  Geometry = 1
  data_frame = data_frame[(data_frame.fLUTConfigId == Geometry)]
  """
  return data_frame

def create_pd_df(dataset_name, n_rows_max=0):
  rfile = uproot.open(MAIN_PATH+"{}".format(dataset_name))

  # Get the list of directories (TDirectory) in the ROOT file
  keys = rfile.keys()


  if not TABLES_TO_KEEP:
      directory_names = [x.split(';')[0] for x in keys if "/" not in x]
      group_names = [x.split(';')[0] for x in keys if "/" in x]
  else:
      directory_names = [x.split(';')[0] for x in keys if '/' not in x and any(t in x for t in TABLES_TO_KEEP)]
      group_names = [x.split(';')[0] for x in keys if '/' in x and any(t in x for t in TABLES_TO_KEEP)]

  Subgroups = np.unique(np.array([x.split('/')[1] for x in group_names]))

  print("\n_________________________________________")
  print('[INFO]: The input dataset has {} directories'.format(len(directory_names)))
  print('[INFO]: The input dataset has {} subgroups'.format(len(Subgroups)))
  print('[INFO]: The input dataset has {} groups'.format(len(group_names)))
  print('[INFO]: The input dataset has {} keys'.format(len(keys)))

  # Creating Pandas dataframes from TTrees
  tables_per_directory = {}
  for dir in group_names:
    tables_per_directory.setdefault(dir.split('/')[0], []).append(dir)

  dataframesPerDirectory = []
  for directory, table_names in tables_per_directory.items():
    tables = [rfile[t].arrays(library='pd') for t in table_names]
    tables = [t for t in tables if len(t) > 0] # e.g. O2sigma0mccores is present but empty in data

    nrows = {len(t) for t in tables}
    if len(nrows) > 1:
      raise ValueError(
          '[ERROR]: tables in {} have different lengths {} and cannot be joined '
          'column-wise: {}'.format(directory, sorted(nrows), table_names))

    dataframesPerDirectory.append(pd.concat(tables, axis=1))

    if n_rows_max > 0 and sum(len(d) for d in dataframesPerDirectory) >= n_rows_max:
      print('[INFO]: stopping after {} directories ({} candidates read)'.format(
          len(dataframesPerDirectory), sum(len(d) for d in dataframesPerDirectory)))
      break

  dataframeFinal = pd.concat(dataframesPerDirectory, axis=0, ignore_index=True)
  del dataframesPerDirectory

  if 'fPhotonPx' not in dataframeFinal.columns:
    raise ValueError('[ERROR]: {} has no O2sigma0cores table: without it there are no candidate '
                     'momenta, so the invariant mass, fOPAngle and fDCADaughters cannot be built. '
                     'Add "AOD/SIGMA0CORES/0" to the OutputDirector and reproduce this AO2D.'.format(dataset_name))
    
  dataframeFinal = add_opening_angle(dataframeFinal)
  dataframeFinal = add_signal_lstar(dataframeFinal)
  dataframeFinal = add_lstar_mass(dataframeFinal)
  dataframeFinal = add_lstar_pt(dataframeFinal)

  print("\n_________________________________________")
  print('[INFO]: Dataframe created for {}'.format(dataset_name))

  return filter_df(dataframeFinal)

#--------------------------------- PROCESSING ---------------------------------
def main():
  if fSplitTrainTest:

    # LOADING DATA 
    dataframeFinal = create_pd_df(DatasetName, NSignal)

    print("\n-------------------------------------------")
    print('[INFO]: The input dataset has a total of {} signal candidates, {} will be selected: '.format(len(dataframeFinal[dataframeFinal[Class_name]==True]), NSignal))
    print('[INFO]: The input dataset has a total of {} bkg candidates, {} will be selected'.format(len(dataframeFinal[dataframeFinal[Class_name]==False]), NBkg))

    SignalCandidates = dataframeFinal[dataframeFinal[Class_name]==True].sample(n=NSignal, random_state=42) 
    BkgCandidates = dataframeFinal[dataframeFinal[Class_name]==False].sample(n=NBkg, random_state=42) 
    TotalCandidates = pd.concat([SignalCandidates, BkgCandidates], axis=0).sample(frac=1) # merging and shuffling

    # Creating training, validation and testing sets:
    Data_TrainVal, Data_Test = train_test_split(TotalCandidates, test_size=0.30, random_state=42, stratify=TotalCandidates[Class_name])
    Data_Train, Data_Val = train_test_split(Data_TrainVal, test_size=0.10, random_state=42, stratify=Data_TrainVal[Class_name])

    #----------------------------- SAVING DATASETS ---------------------------------
    Data_Train.to_parquet(MAIN_PATH+"Dataset/Processed/{}_Train.parquet".format(OutputPrefix))
    Data_Val.to_parquet(MAIN_PATH+"Dataset/Processed/{}_Val.parquet".format(OutputPrefix))
    Data_Test.to_parquet(MAIN_PATH+"Dataset/Processed/{}_Test.parquet".format(OutputPrefix))
    Data_Val.to_csv(MAIN_PATH+"Dataset/Processed/{}_Val.csv".format(OutputPrefix)) # Activate this line if you want to take a look on a .csv file


    #--------------------------------- END -----------------------------------------
    print("\n-------------------------------------------")
    print('[INFO]: Train set total size: {}'.format(len(Data_Train)))
    print('[INFO]: Val set total size: {}'.format(len(Data_Val)))
    print('[INFO]: Test set total size: {}'.format(len(Data_Test)))

  elif fSeparateAO2Ds:
    # LOADING DATA 

    dataframeSignalFinal = create_pd_df(SignalFile, NRowsSignalRead)
    dataframeBkgFinal = create_pd_df(BackgroundFile, NRowsBkgRead)

    # Signal: MC candidates matched to the L*. Background: data candidates in the mass sidebands.
    dataframeSignalFinal = dataframeSignalFinal[np.abs(dataframeSignalFinal['fPDGCode']) == LStarPDG]
    dataframeBkgFinal = dataframeBkgFinal[(dataframeBkgFinal['fLStarMass'] < MassWindowMin)
                                          | (dataframeBkgFinal['fLStarMass'] > MassWindowMax)]

    # Keep only the columns both samples have (data carries no MC columns)
    CommonColumns = [c for c in dataframeSignalFinal.columns if c in dataframeBkgFinal.columns]
    dataframeSignalFinal = dataframeSignalFinal[CommonColumns]
    dataframeBkgFinal = dataframeBkgFinal[CommonColumns]

    print("\n-------------------------------------------")
    print('[INFO]: The input dataset has a total of {} signal candidates, {} will be selected: '.format(len(dataframeSignalFinal), NSignal))
    print('[INFO]: The input dataset has a total of {} Bkg candidates, {} will be selected'.format(len(dataframeBkgFinal), NBkg))
    
    # Creating training, validation and testing sets:
    SignalCandidates = dataframeSignalFinal.sample(n=min(NSignal, len(dataframeSignalFinal)), random_state=42)
    BkgCandidates = dataframeBkgFinal.sample(n=min(NBkg, len(dataframeBkgFinal)), random_state=42)
    SignalCandidates["fIsLStar"] = True
    BkgCandidates["fIsLStar"] = False

    # Split signal
    Signal_TrainVal, Signal_Test = train_test_split(SignalCandidates, test_size=0.30, random_state=42)
    Signal_Train, Signal_Val = train_test_split(Signal_TrainVal, test_size=0.10, random_state=42)

    # Split background
    Bkg_TrainVal, Bkg_Test = train_test_split(BkgCandidates, test_size=0.30, random_state=42)
    Bkg_Train, Bkg_Val = train_test_split(Bkg_TrainVal, test_size=0.10, random_state=42)

    # Merge splits
    Data_Train = pd.concat([Signal_Train, Bkg_Train], axis=0).sample(frac=1, random_state=42)
    Data_Val = pd.concat([Signal_Val, Bkg_Val], axis=0).sample(frac=1, random_state=42)
    Data_Test = pd.concat([Signal_Test, Bkg_Test], axis=0).sample(frac=1, random_state=42)

    # Save datasets
    Data_Train.to_parquet(MAIN_PATH + f"Dataset/Processed/{OutputPrefix}_Train.parquet")
    Data_Val.to_parquet(MAIN_PATH + f"Dataset/Processed/{OutputPrefix}_Val.parquet")
    Data_Test.to_parquet(MAIN_PATH + f"Dataset/Processed/{OutputPrefix}_Test.parquet")

    # Optional CSV for validation
    Data_Val.to_csv(MAIN_PATH + f"Dataset/Processed/{OutputPrefix}_Val.csv")

    print("\n-------------------------------------------")
    print("[INFO]: Train set class distribution:\n{}".format(Data_Train["fIsLStar"].value_counts()))

    print("\n-------------------------------------------")
    print('[INFO]: Train set total size: {}'.format(len(Data_Train)))
    print('[INFO]: Val set total size: {}'.format(len(Data_Val)))
    print('[INFO]: Test set total size: {}'.format(len(Data_Test)))

  else:

    dataframeFinal = create_pd_df(DatasetName)
    print("\n-------------------------------------------")
    print('[INFO]: The input dataset has a total of {} signal candidates'.format(len(dataframeFinal[dataframeFinal[Class_name]==True])))
    print('[INFO]: The input dataset has a total of {} Bkg candidates'.format(len(dataframeFinal[dataframeFinal[Class_name]==False])))

    dataframeFinal.to_parquet("{}.parquet".format(DatasetName))
    dataframeFinal.to_csv("{}.csv".format(DatasetName))

  t1 = time.time() - t0
  print("\n_________________________________________")
  print("Total time elapsed (min): ", t1/60)
  print("_________________________________________")

if __name__ == "__main__":
  main()
