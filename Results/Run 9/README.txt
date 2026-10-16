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
    "max_depth": 2,
    "learning_rate": 0.049849789537073634,
    "n_estimators": 298,
    "min_child_weight": 1,
    "gamma": 1.9935185370598936e-06,
    "subsample": 0.6252206748098751,
    "colsample_bytree": 0.7278968659717769,
    "reg_alpha": 1.2586800190563545e-06,
    "reg_lambda": 0.8477851501548531
  }
}