Description: Example Run. Data and MC! MC pos/neg daughters swapped
 
------------------------------------------------------- 
 
Run configurations: 
 
{
  "Dataset": {
    "DatasetName": "LambdaStar",
    "Class": "fIsLStar",
    "Features": [
      "fLambdaCosPA",
      "fDCADaughters",
      "fLambdaAlpha",
      "fLambdaDCANegPV",
      "fLambdaDCAPosPV",
      "fLambdaLifeTime",
      "fLambdaQt",
      "fLambdaRadius",
      "fPhotonAlpha",
      "fPhotonCosPA",
      "fPhotonDCADau",
      "fPhotonDCANegPV",
      "fPhotonDCAPosPV",
      "fPhotonQt",
      "fPhotonRadius",
      "fOPAngle"
    ]
  },
  "BDT": {
    "max_depth": 9,
    "learning_rate": 0.15486920024170198,
    "n_estimators": 53,
    "min_child_weight": 6,
    "gamma": 4.1256115626760464e-08,
    "subsample": 0.4853129360119775,
    "colsample_bytree": 0.033918031986580464,
    "reg_alpha": 0.048001278530127596,
    "reg_lambda": 5.59039191313539e-05
  }
}