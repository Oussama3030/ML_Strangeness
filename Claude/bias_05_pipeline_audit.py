"""
bias_05 -- bookkeeping problems in the pipeline itself, independent of physics.

These are the ones that inflate the *test* number without anybody touching a
distribution: how the sample is drawn, and what ends up on both sides of the
train/test split.
"""
import os
import numpy as np
import pandas as pd
import uproot

import biaslib as bl

R = bl.Report('bias_05_pipeline_audit')

# ------------------------------------------------------- 1. how much data is read
R.head('1. HOW MUCH OF THE DATA IS ACTUALLY USED')
rf = uproot.open(bl.BKG_FILE)
dirs = [k.split(';')[0] for k in rf.keys(recursive=False) if k.startswith('DF_')]
R('  AO2DMerged_Bkg.root holds {} dataframes.'.format(len(dirs)))
R('')
R('  Codes/01_ProcessData.py sets NRowsBkgRead = 2000000, and create_pd_df')
R('  applies it as a break AFTER each whole directory:')
R('')
R('      if n_rows_max > 0 and sum(len(d) for d in dataframesPerDirectory) >= n_rows_max:')
R('          break')
R('')
cum = 0
stop_at = None
for i, d in enumerate(dirs[:12]):
    n = rf['{}/O2sigma0cores'.format(d)].num_entries
    cum += n
    if stop_at is None and cum >= 2000000:
        stop_at = i + 1
R('  so it stops after {} of {} dataframes ({} candidates).'.format(stop_at, len(dirs), cum))
R('  That is ~{:.1f}% of the file, and they are the FIRST {} timeframes, not a'.format(
    100.0 * stop_at / len(dirs), stop_at))
R('  random draw: whatever run / fill / occupancy condition those timeframes')
R('  belong to is what the background class means.')

# ------------------------------------------------ 2. sample sizes and class balance
R.head('2. SAMPLE SIZES AND CLASS BALANCE')
tr = pd.read_parquet(bl.MAIN_PATH + 'Dataset/Processed/LambdaStar_Train.parquet')
te = pd.read_parquet(bl.MAIN_PATH + 'Dataset/Processed/LambdaStar_Test.parquet')
va = pd.read_parquet(bl.MAIN_PATH + 'Dataset/Processed/LambdaStar_Val.parquet')
for tag, d in [('Train', tr), ('Val', va), ('Test', te)]:
    vc = d[bl.CLASS_NAME].value_counts()
    R('  {:6s} n = {:5d}   signal = {:5d}   background = {:5d}   S/B = 1:{:.0f}'.format(
        tag, len(d), int(vc.get(True, 0)), int(vc.get(False, 0)),
        vc.get(False, 0) / max(vc.get(True, 1), 1)))
mc = bl.load_mc()
n_avail = int((np.abs(mc.fPDGCode) == bl.LSTAR_PDG).sum())
R('')
R('  NSignal = 1000 was requested; {} true L* exist in the MC file.'.format(n_avail))
R('  The whole model rests on {} signal candidates, {} of them in training.'.format(
    1000, int(tr[bl.CLASS_NAME].sum())))
R('  No scale_pos_weight is set, so the 1:10 imbalance also means the confusion')
R('  matrix and any accuracy number at threshold 0.5 are dominated by background.')

# ------------------------------------------- 3. the same physical V0 on both sides
R.head('3. THE SAME PHYSICAL V0 APPEARS IN TRAIN AND TEST')
R('  A candidate is a (Lambda V0, photon V0) pair.  One reconstructed Lambda is')
R('  paired with several photons, so several candidates share the same physical')
R('  Lambda.  train_test_split shuffles candidates, not V0s, so the identical')
R('  Lambda lands on both sides of the split and the test set is not')
R('  independent of the training set.')
R('')
R('  Identity is taken from fLambdaV0ID / fPhotonV0ID.  These are unambiguous')
R('  for the signal (the MC file is a single dataframe); in data they restart')
R('  per dataframe, so the background numbers below are quoted separately.')
R('')
for cls, tag in [(True, 'signal'), (False, 'background')]:
    a, b = tr[tr[bl.CLASS_NAME] == cls], te[te[bl.CLASS_NAME] == cls]
    for col, name in [('fLambdaV0ID', 'Lambda'), ('fPhotonV0ID', 'photon')]:
        frac = b[col].isin(set(a[col])).mean()
        R('  {:10s} {:6s} : {:5d} train / {:5d} test, {:5.1f}% of test candidates reuse '
          'a training {}'.format(tag, name, len(a), len(b), 100 * frac, name))
R('')
R('  The leak is concentrated in the signal class -- the small class whose')
R('  recall drives the ROC curve.  Roughly three out of five "unseen" signal')
R('  candidates in the test set are a photon swapped onto a Lambda the model')
R('  already trained on, sharing all eight fLambda* features.')
R('')
R('  Background sharing is low only because 10000 candidates were drawn from')
R('  {} available: collisions are rare by dilution, not by construction.'.format(
    '4.6 million'))
R('')
R('  MC: {} candidates are built from only {} distinct Lambda V0s'.format(
    len(mc), mc.fLambdaV0ID.nunique()))
sig_raw = mc[np.abs(mc.fPDGCode) == bl.LSTAR_PDG]
R('      the {} true L* come from {} distinct Lambda V0s and {} distinct'.format(
    len(sig_raw), sig_raw.fLambdaV0ID.nunique(), sig_raw.fPhotonV0ID.nunique()))
R('      photon V0s, i.e. only {} distinct (Lambda, photon) pairings.'.format(
    sig_raw.groupby(['fLambdaV0ID', 'fPhotonV0ID']).ngroups))
g = sig_raw.groupby(['fLambdaV0ID', 'fPhotonV0ID']).size()
R('      {} pairings are written more than once ({} extra rows), once under the'.format(
    int((g > 1).sum()), int((g - 1).sum())))
R('      L* and once under the anti-L* hypothesis, with slightly different')
R('      topology values -- so ~{:.0f}% of the signal sample is the same'.format(
    100.0 * (g - 1).sum() / len(sig_raw)))
R('      physical candidate counted twice.')

# ------------------------------------------------------- 4. exact duplicate rows
R.head('4. EXACT DUPLICATE CANDIDATES ACROSS THE SPLIT')
cols = bl.FEATURES
ktr = set(map(tuple, np.round(tr[cols].values, 6)))
kte = list(map(tuple, np.round(te[cols].values, 6)))
dup = sum(1 for k in kte if k in ktr)
R('  test rows identical to a training row in all 16 features: {} / {} ({:.2f}%)'.format(
    dup, len(kte), 100.0 * dup / len(kte)))

# ------------------------------------------------------- 5. what the label means
R.head('5. WHAT THE LABEL ACTUALLY MEANS')
R('  fIsLStar = True  : MC candidate with |fPDGCode| == 3124')
R('  fIsLStar = False : DATA candidate with mass outside 1.45-1.59')
R('')
R('  Among the 1396 MC candidates labelled signal:')
for c in ['fLambdaIsCorrectlyAssoc', 'fPhotonIsCorrectlyAssoc']:
    if c in sig_raw.columns:
        v = sig_raw[c].value_counts()
        R('    {:26s} True {:5d}  False {:5d}   ({:.1f}% correctly associated)'.format(
            c, int(v.get(True, 0)), int(v.get(False, 0)),
            100.0 * v.get(True, 0) / len(sig_raw)))
R('')
R('  The probability plot in Results/Run 3 is labelled "Correctly Associated"')
R('  vs "Wrongly Associated".  That is not what was trained: the two classes')
R('  are "MC true L*" and "data mass sideband".  A candidate that is wrongly')
R('  associated in MC is not in the background class at all -- it is simply')
R('  discarded.  The plot therefore cannot show what its legend claims.')

# ------------------------------------------------------ 6. the model configuration
R.head('6. MODEL CONFIGURATION')
p = bl.run3_params()
nfeat = len(bl.FEATURES)
R('  colsample_bytree = {:.4f}  ->  {:.2f} of {} features per tree'.format(
    p['colsample_bytree'], p['colsample_bytree'] * nfeat, nfeat))
R('  XGBoost clamps that to 1, so every tree in the ensemble sees exactly ONE')
R('  randomly chosen feature.  With max_depth = {} the model is close to an'.format(p['max_depth']))
R('  additive single-variable ensemble -- it cannot use correlations between')
R('  features, and the reported importance is spread almost at random.')
R('')
R('  Optuna ran {} trials, scored on the validation set, and the best'.format(10))
R('  configuration was then kept.  The validation AUC is therefore a selected')
R('  maximum, not an unbiased estimate; only the test set is unbiased, and')
R('  section 3 shows the test set is not independent either.')

R.save()
