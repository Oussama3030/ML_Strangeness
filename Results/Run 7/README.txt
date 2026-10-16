Description: Data: 20 , MC: 2.6k, removed lambda cospa, radius and lifetime. Hard pt cut at 10GeV. rap/eta 08 , photon cr at 70, lambda at 60
 
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
    "max_depth": 5,
    "learning_rate": 0.014983207936089864,
    "n_estimators": 190,
    "min_child_weight": 7,
    "gamma": 1.617895326518392e-06,
    "subsample": 0.5442255834556226,
    "colsample_bytree": 0.674212956770909,
    "reg_alpha": 7.109540656725241e-07,
    "reg_lambda": 0.004161188519893381
  }
}