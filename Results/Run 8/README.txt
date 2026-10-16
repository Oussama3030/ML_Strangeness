Description: Data: 20 , MC: 2.0k., removed lambda cospa, radius and lifetime. Hard pt cut at 10GeV. rap/eta 08 , photon cr at 70, lambda at 60. ntrial 40
 
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
    "max_depth": 8,
    "learning_rate": 0.0883865135684475,
    "n_estimators": 112,
    "min_child_weight": 7,
    "gamma": 8.79495185426418e-06,
    "subsample": 0.9113701598580547,
    "colsample_bytree": 0.9906083331775493,
    "reg_alpha": 4.33465857940181e-07,
    "reg_lambda": 0.00015224815019979863
  }
}