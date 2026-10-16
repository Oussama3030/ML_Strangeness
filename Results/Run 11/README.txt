Description: Data: 20k , MC: 3.7k., removed lambda cospa, radius and lifetime. Hard pt cut at 10GeV. rap/eta 08 , photon cr at 70, lambda at 60. ntrial 40/ added arm vars
 
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
      "fOPAngle",
      "fLStarQtArm",
      "fLStarAlpha"
    ]
  },
  "BDT": {
    "max_depth": 6,
    "learning_rate": 0.10439853450431397,
    "n_estimators": 284,
    "min_child_weight": 6,
    "gamma": 8.993592361729879e-07,
    "subsample": 0.9051457828267293,
    "colsample_bytree": 0.432094241789153,
    "reg_alpha": 3.2549166580262374e-07,
    "reg_lambda": 0.002758000888277132
  }
}