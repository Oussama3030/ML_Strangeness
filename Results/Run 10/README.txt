Description: Data: 20k , MC: 3.7k., removed lambda cospa, radius and lifetime. Hard pt cut at 10GeV. rap/eta 08 , photon cr at 70, lambda at 60. ntrial 40
 
------------------------------------------------------- 
 
Run configurations: 
 
{
  "Dataset": {
    "DatasetName": "LambdaStar",
    "Class": "fIsLStar",
    "Features": [
      "fLambdaDCADau",
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
    "learning_rate": 0.05161548409855723,
    "n_estimators": 357,
    "min_child_weight": 7,
    "gamma": 2.0215134782737774e-07,
    "subsample": 0.30576123477767264,
    "colsample_bytree": 0.939020075821907,
    "reg_alpha": 2.1118520213327846e-08,
    "reg_lambda": 0.15631777126527546
  }
}