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
# 08_UploadModelCCDB
# ================
#
# This script uploads ONNX models to CCDB (run it inside O2Physics environment)
#
#    Comments, questions, complaints, suggestions?
#    Please write to:
#    gianni.shigeru.setoue.liveraro@cern.ch
#    david.dobrigkeit.chinellato@cern.ch

#____________________________
# Imports
import ROOT
import pickle
import os

#---------------------------  MAIN CONFIGURATIONS ----------------------------
sor = 1695750422702
eor = 1698617337860

MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/'
RESULTS_PATH = MAIN_PATH + 'Results/'

print('Which ML Run would you like to load?. Available Runs: \n', os.listdir(RESULTS_PATH))
RunNumber = str (input())
RUN_PATH = RESULTS_PATH+'Run {}'.format(RunNumber)
# Loading dict with predictions
with open(RUN_PATH+"/RunConfig.pkl", 'rb') as f:
    RunConfig = pickle.load(f)

onnx_file_name = RUN_PATH +"/LambdaStar_BDTModel"
onnx_file_path = onnx_file_name+".onnx"
description = onnx_file_name+" first test!"

ccdb_path_production = "Users/o/obenchik/MLModels/BDTModelTestLStar/Binary/Run11"
ccdb_url_production = "https://alice-ccdb.cern.ch"

#----------------------------------  SETUP -----------------------------------
ccdb_path = ccdb_path_production
ccdb_url = ccdb_url_production

ccdb = ROOT.o2.ccdb.CcdbApi()
ccdb.init(ccdb_url)

def uploadONNXToCCDB(description, onnx_file_path, start, stop):
    metadata = ROOT.std.map(ROOT.std.string, ROOT.std.string)()

    metadata.clear()
    metadata.insert(ROOT.std.pair(ROOT.std.string, ROOT.std.string)("Description", "%s" % description))
    metadata.insert(ROOT.std.pair(ROOT.std.string, ROOT.std.string)("Author", "Oussama Benchikhi"))

    # Read ONNX file as binary
    with open(onnx_file_path, 'rb') as file:
        onnx_data = file.read()

    # Store ONNX content as TNamed in CCDB
    #ccdb.storeAsTFileAny(onnx_content, ccdb_path, metadata, start, stop)
    ccdb.storeAsBinaryFile(onnx_data, len(onnx_data), onnx_file_path, "ONNX Model | file read as binary string", ccdb_path, metadata, start, stop)
    


#--------------------------------  EXECUTION ----------------------------
uploadONNXToCCDB(description, onnx_file_path, int(int(sor) / 1000 - 2) * 1000, int(int(eor) / 1000 + 10) * 1000)


