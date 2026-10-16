Description: Data: 20 , MC: 2.6k, removed lambda cospa, radius and lifetime. Hard pt cut at 10GeV.
 
------------------------------------------------------- 
 
Run configurations: 
 
{
  "Dataset": {
    "DatasetName": "LambdaStar",
    "Class": "fIsLStar",
    "Features": [
      "fDCADaughters",
      "fLambdaAlpha",
      "fLambdaDCANegPV",
      "fLambdaDCAPosPV",
      "fLambdaQt",
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
    "max_depth": 2,
    "learning_rate": 0.08405782261178024,
    "n_estimators": 162,
    "min_child_weight": 9,
    "gamma": 0.00015600859965420575,
    "subsample": 0.8544375004297208,
    "colsample_bytree": 0.34304498386268467,
    "reg_alpha": 0.1753668437618697,
    "reg_lambda": 2.608938209959854e-07
  }
}