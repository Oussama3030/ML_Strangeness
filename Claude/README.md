# Is the Λ(1520) BDT biased?

Investigation of the near-perfect BDT separation seen in `Results/Run 3/ProbabilityPlot.png`.

Everything here is read-only with respect to the analysis: the macros read
`Dataset/Raw/*.root` and `Dataset/Processed/*.parquet` and write only into
`Claude/output/` and `Claude/cache/`. Nothing in `Codes/`, `Dataset/` or
`Results/` is modified.

```
python3 run_all.py          # ~6 min, or run any bias_0N_*.py on its own
```

Results land in `output/` as one `.txt` and one or more `.png` per macro.

---

## Short answer

Yes. The Run 3 plot is not measuring Λ(1520)-vs-combinatorial. Four separate
effects inflate it, and the largest is a bookkeeping bug that makes simulation
and data trivially distinguishable.

The decisive number: train the same BDT, same hyperparameters, same 16
features, on **MC background vs data background** — two samples with *no
physics difference whatsoever*, both pure combinatorics. It reaches

```
AUC = 0.9999
```

A classifier that can separate two samples containing the same physics with
AUC ≈ 1 does not need any Λ(1520) information to reproduce the Run 3 plot.

---

## Finding 1 — MC and data are anchored to opposite L3 polarity

`bias_02` · `output/bias_02_root_cause_swap.png`

Median DCA to the primary vertex, sentinel (999) excluded:

| variable | MC signal | MC background | **data** background |
|---|---|---|---|
| `fPhotonDCAPosPV` | **+3.45** | **+8.67** | **−8.16** |
| `fPhotonDCANegPV` | **−3.33** | **−8.52** | **+8.27** |
| `fLambdaDCAPosPV` | +0.114 | +0.119 | −0.142 |
| `fLambdaDCANegPV` | −0.116 | −0.122 | +0.150 |

Both MC columns carry one sign, both data columns the other, for the Λ *and*
the photon. A single threshold at zero tells simulation from data almost
perfectly.

The daughter DCA to the PV is a **signed** quantity whose sign follows the track
bending, so it flips with the L3 polarity. The columns are labelled correctly in
both files — the magnitudes agree, only the signs are mirrored. Negating the four
DCA values in the MC (sentinel `+999` left untouched) puts all four distributions
back on top of the data:

| | before | after |
|---|---|---|
| `fPhotonDCAPosPV` | 0.894 | 0.521 |
| `fPhotonDCANegPV` | 0.831 | 0.558 |
| `fLambdaDCAPosPV` | 0.585 | 0.502 |
| `fLambdaDCANegPV` | 0.585 | 0.502 |

and the placebo training collapses from **0.9999 → 0.752**.

Note this is a *sign* flip, not a daughter interchange. Swapping the Pos/Neg
getters compensates the DCA numerically (because the two daughters bend
oppositely) but also moves the PID, eta, ITS and track-quality columns, which are
polarity independent and must not move — for a Λ → pπ⁻ the proton is the positive
daughter whatever the field does. Measured: swapping leaves
`fPhotonPosTPCNSigmaEl` at AUC 0.758 → 0.760 and `fPhotonPosITSCls` at
0.5001 → 0.5002, i.e. it buys nothing there and only mislabels.

Fixed in `sigma0builder.cxx` via the `fFlipDauDCAToPVSign` configurable
(see "Suggested order of work").

**Consequence on real data.** `bias_04` applies the as-trained model to
791 011 real data candidates: **0.003 %** get P > 0.5. The model keys on a sign
convention the data does not share, so in production it rejects essentially
everything.

## Finding 2 — the MC signal and the data background live at different momenta

`bias_03` · `output/bias_03_kinematics.png`

| sample | median pT | 76 % above 4 GeV/c? |
|---|---|---|
| MC signal | 7.01 GeV/c | 77 % above 4 |
| data background | 1.67 GeV/c | 3 % above 4 |

Spectrum overlap 0.55. All 1396 true Λ* are generator-produced primaries — a
pT-hardened / signal-enhanced production, trained against minimum-bias data.
Every topological variable scales with momentum, so the BDT gets "is this hard
or soft?" for free. Reweighting the background onto the signal pT spectrum
(with the DCA bug removed so it is not masked) costs **0.967 → 0.853**.

## Finding 3 — the background class is *defined* by a mass cut

`bias_04` · `output/bias_04_mass_sculpting.png`

`01_ProcessData.py` builds the background as `fLStarMass < 1.45 or > 1.59`.
`fOPAngle` is essentially the mass at fixed daughter momenta —
corr(`fOPAngle`, `fLStarMass`) = **+0.72** in a narrow pT slice — and it is the
single most powerful feature in the list (single-feature AUC 0.943).

So the BDT partly learns the sideband definition. The practical danger is
sculpting: with the DCA bug repaired, tightening the cut on pure background
preferentially keeps candidates inside the mass window by a factor

| working point | in/out |
|---|---|
| keep top 10 % | 1.11 |
| keep top 1 % | 1.17 |
| keep top 0.1 % | **1.29** |

i.e. the selection alone builds a bump at 1.52 out of background with no Λ* in
it. This one survives fixing Findings 1 and 2, and it is the one that would
quietly corrupt a yield.

## Finding 4 — the test set is not independent of the training set

`bias_05`

A candidate is a (Λ V0, photon V0) pair, and one reconstructed Λ pairs with
several photons. `train_test_split` shuffles candidates, not V0s:

* **59.3 %** of test signal candidates reuse a Λ V0 from training
* **56.0 %** reuse a photon V0

Three out of five "unseen" signal candidates are a photon swapped onto a Λ the
model already trained on, sharing all eight `fLambda*` features. The leak sits
in the small class whose recall drives the ROC. Background sharing is only
~4.5 %, and only because 10 000 candidates were diluted out of 4.6 million.

Also: the 1396 true Λ* contain only 1114 distinct (Λ, γ) pairings — 230 are
written twice, once under the Λ* and once under the anti-Λ* hypothesis, so
~20 % of the signal sample is the same physical candidate counted twice.

## Smaller things worth knowing

* **Only 1.8 % of the data file is used.** `NRowsBkgRead = 2000000` is applied
  as a break *after* each whole dataframe, so the background is the first
  5 of 280 dataframes — consecutive timeframes, not a random draw.
* **`colsample_bytree = 0.0161` × 16 features = 0.26**, clamped to 1: every
  tree sees exactly one randomly chosen feature. With `max_depth = 2` the model
  is close to an additive single-variable ensemble and cannot use correlations.
  The SHAP ranking is correspondingly arbitrary.
* **The Run 3 plot legend is wrong.** It reads "Correctly Associated" vs
  "Wrongly Associated". The classes are "MC true Λ*" and "data mass sideband".
  Wrongly-associated MC candidates are not in the background class — they are
  discarded. (Of the 1396 signal candidates, 80.5 % have
  `fLambdaIsCorrectlyAssoc == True`; the other 19.5 % are labelled signal
  anyway.)
* `NSignal = 1000` of 1396 available; 630 in training, 1:10 imbalance with no
  `scale_pos_weight`, so the confusion matrix at threshold 0.5 is dominated by
  background.

---

## What the number should be

`bias_06` · `output/bias_06_ladder.png` — one bias removed per rung:

| | configuration | AUC |
|---|---|---|
| 0 | as trained | 0.9999 |
| 1 | + group-aware train/test split | 0.9999 |
| 2 | + pos/neg daughter swap repaired | 0.9539 |
| 3 | + background pT-matched to signal | 0.8796 |
| 4 | + background from MC (no MC/data shift) | 0.9397 |
| 5 | + background not selected on mass | **0.9441** |

The ladder is not monotonic on purpose: rung 4 swaps the data background for an
MC one, and combinatorics inside a signal-enhanced MC production is not the
same mixture as combinatorics in minimum-bias data. Rung 5 is honest about
*origin*, but its background still comes from MC, so **0.94 is an upper bound**,
not a guarantee.

There is real physics separation here — roughly AUC 0.94 rather than 0.9994.
The Λ(1520) is genuinely learnable. It is just nowhere near as learnable as the
Run 3 plot claims.

## Suggested order of work

1. **Fix the polarity sign at the source** — done: `sigma0builder.cxx` now has
   `fFlipDauDCAToPVSign` (default `false`, so data is untouched). Set
   `"fFlipDauDCAToPVSign": "true"` in the `sigma0builder` block of the MC config
   and reproduce the MC AO2D. Fixing it here rather than in Python matters
   because the ONNX model uploaded to CCDB would otherwise carry the bug into
   the O2 task. Confirm the anchor run's L3 polarity really is opposite before
   flipping, and check the full data sample does not itself mix polarities —
   a single global flag cannot cover a mixed sample.
2. **Take the background from MC too**, at least for a control training, so
   "which production is this?" is not available as a feature. Keep the
   data-sideband training only as a cross-check, and always run the placebo
   (MC bkg vs data bkg) next to it — if that is not ≈ 0.5, the main training
   cannot be trusted.
3. **Match the pT spectra**, or train in pT bins (the task already plots
   AUC vs pT), or reweight the MC signal to a realistic spectrum.
4. **Split by V0 id, not by candidate** (`GroupShuffleSplit` on
   `fLambdaV0ID`), and de-duplicate the Λ*/anti-Λ* double entries.
5. **Check sculpting before using any working point**: apply the model to data
   and confirm the selection efficiency is flat in `M(Λγ)`. Consider dropping
   `fOPAngle`, or decorrelating the score from the mass.
6. Raise `NOptunaTrials` and fix `colsample_bytree` to a sane floor so the
   tuner cannot hand back a one-feature-per-tree model.
