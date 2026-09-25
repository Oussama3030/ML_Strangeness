"""
bias_04 -- the background class is DEFINED by a mass cut, so the BDT learns the cut.

Codes/01_ProcessData.py builds the background as

    fLStarMass < 1.45  OR  fLStarMass > 1.59

and the signal as MC L*, which by construction sit inside that excluded window.
So "background" does not mean "combinatorial", it means "reconstructed mass
outside 1.45-1.59".  Any feature that knows about the mass therefore separates
the classes for free -- and fOPAngle knows about it exactly, because

    M^2 = m_L^2 + 2 E_g ( E_L - |p_L| cos(theta) )

is essentially a function of the opening angle at fixed daughter momenta.

Consequences tested here:
  1. how well does the opening angle alone reproduce the class label?
  2. what happens if the background is taken from INSIDE the window instead
     (D_IN: same data, same combinatorics, just not selected on mass)?
  3. does a BDT cut sculpt a peak into a pure-background mass spectrum?

(3) is the one that matters for the physics result: if the selection efficiency
rises inside the mass window, a bump appears at 1.52 with no L* in it at all.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import biaslib as bl

R = bl.Report('bias_04_mass_sculpting')
S = bl.build_samples(n_max=300000)
BROKEN = ['fPhotonDCAPosPV', 'fPhotonDCANegPV']
CLEAN = [f for f in bl.FEATURES if f not in BROKEN]

# ----------------------------------------------------------- 1. opening angle vs mass
R.head('1. THE OPENING ANGLE IS A MASS VARIABLE')
data = bl.load_data()
c = np.corrcoef(data['fOPAngle'].values, data['fLStarMass'].values)[0, 1]
R('  corr(fOPAngle, fLStarMass) in data          : {:+.4f}'.format(c))
sub = data[data.fLStarPt.between(1.5, 2.5)]
c2 = np.corrcoef(sub['fOPAngle'].values, sub['fLStarMass'].values)[0, 1]
R('  same, in a narrow pT slice (1.5-2.5 GeV/c)  : {:+.4f}'.format(c2))
R('')
R('  Median fOPAngle of data candidates, by mass region:')
for lo, hi, tag in [(0.0, 1.45, 'below window'), (1.45, 1.59, 'INSIDE window'),
                    (1.59, 3.0, 'above window')]:
    m = data[(data.fLStarMass >= lo) & (data.fLStarMass < hi)]
    R('    {:14s} [{:.2f}, {:.2f})  n = {:8d}   median fOPAngle = {:.3f}'.format(
        tag, lo, hi, len(m), np.median(m.fOPAngle)))
R('')
R('  fOPAngle single-feature AUC, S_MC vs B_DATA : {:.4f}  (bias_02)'.format(0.9429))
R('  It is the strongest "physics" feature in the list, and it is precisely the')
R('  variable the background selection was made on.')

# ------------------------------------------- 2. background taken from inside the window
R.head('2. SAME DATA, BACKGROUND NOT SELECTED ON MASS')
R('  D_IN is data inside 1.45-1.59.  It is the same combinatorics as B_DATA')
R('  (L* -> Lambda gamma is far too rare to matter here), but it was never')
R('  selected on mass.  If the classifier were learning physics, swapping')
R('  B_DATA -> D_IN would change little.')
R('')
n = min(len(S['S_MC']), 20000)
sig = S['S_MC'].sample(n=n, random_state=1)
R('  {:46s} {:>10s} {:>10s}'.format('', 'all 16', 'DCA-fixed'))
for lab, neg in [('S_MC vs B_DATA  (sideband bkg, the pipeline)', S['B_DATA']),
                 ('S_MC vs D_IN    (in-window bkg, same data)', S['D_IN'])]:
    nn = neg.sample(n=min(len(neg), 20000), random_state=1)
    a16, _, _ = bl.fit_and_score(sig, nn, features=bl.FEATURES)
    a14, _, _ = bl.fit_and_score(sig, nn, features=CLEAN)
    R('  {:46s} {:10.4f} {:10.4f}'.format(lab, a16, a14))
R('')
R('  and with fOPAngle also removed:')
noop = [f for f in CLEAN if f != 'fOPAngle']
for lab, neg in [('S_MC vs B_DATA', S['B_DATA']), ('S_MC vs D_IN  ', S['D_IN'])]:
    nn = neg.sample(n=min(len(neg), 20000), random_state=1)
    a, _, _ = bl.fit_and_score(sig, nn, features=noop)
    R('  {:46s} {:10.4f}'.format(lab + '  (13 features, no fOPAngle)', a))

# ------------------------------------------------------------------ 3. sculpting test
R.head('3. WHAT THE MODEL DOES WHEN IT MEETS REAL DATA')
R('  Train exactly as the pipeline does (S_MC vs sideband data), then apply the')
R('  model to ALL data, including the mass window it never saw.  The data holds')
R('  no meaningful L* signal at this level, so a healthy model should keep a')
R('  sensible fraction of candidates and NOT change the mass shape.')

from xgboost import XGBClassifier

bkg_tr = S['B_DATA'].sample(n=min(len(S['B_DATA']), 60000), random_state=7)
appl = data.sample(n=min(len(data), 800000), random_state=11).copy()
appl = appl[~appl.index.isin(bkg_tr.index)]

mbins = np.linspace(1.25, 1.95, 71)
mctr = 0.5 * (mbins[1:] + mbins[:-1])
h_all, _ = np.histogram(appl.fLStarMass, bins=mbins)
inw = ((appl.fLStarMass >= bl.MASS_WINDOW_MIN) & (appl.fLStarMass <= bl.MASS_WINDOW_MAX)).values

models = {}
for tag, feats in [('all 16 features (as trained)', bl.FEATURES),
                   ('14 features, photon DCAs removed', CLEAN)]:
    Xtr = pd.concat([sig[feats], bkg_tr[feats]], ignore_index=True)
    ytr = np.r_[np.ones(len(sig), dtype=int), np.zeros(len(bkg_tr), dtype=int)]
    m = XGBClassifier(**bl.run3_params())
    m.fit(Xtr.values, ytr)
    p = m.predict_proba(appl[feats].values)[:, 1]
    models[tag] = p

    R('')
    R('  -- {} --'.format(tag))
    R('  fraction of data candidates with P > 0.5 : {:.6f}'.format((p > 0.5).mean()))
    R('  {:>22s} {:>10s} {:>12s} {:>12s} {:>10s}'.format(
        'working point', 'threshold', 'eff outside', 'eff inside', 'in/out'))
    for keep in [1.0, 0.10, 0.01, 0.001]:
        thr = np.quantile(p, 1 - keep) if keep < 1.0 else -1.0
        sel = p > thr
        e_out = (sel & ~inw).sum() / max((~inw).sum(), 1)
        e_in = (sel & inw).sum() / max(inw.sum(), 1)
        ratio = e_in / e_out if e_out > 0 else np.nan
        R('  {:>22s} {:10.4f} {:12.5f} {:12.5f} {:10.3f}'.format(
            'keep top {:g}%'.format(keep * 100), thr, e_out, e_in, ratio))

R('')
R('  "in/out" is the factor by which the BDT preferentially keeps candidates')
R('  whose mass falls in the signal window.  1.00 = no sculpting; > 1 means the')
R('  cut alone builds a bump at 1.52 out of pure background.')

# -------------------------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5))

ax = axes[0]
for tag, p in models.items():
    short = 'all 16' if '16' in tag else 'DCA removed'
    ax.hist(p, bins=np.linspace(0, 1, 60), histtype='step', lw=2, label=short)
ax.set_yscale('log')
ax.set_xlabel('BDT output on real data', fontsize=12)
ax.set_ylabel('candidates', fontsize=12)
ax.set_title('Score of real data under the trained model', fontsize=12)
ax.legend(fontsize=9)
ax.grid(ls='--', alpha=0.4)

ax = axes[1]
p = models['14 features, photon DCAs removed']
ax.step(mctr, h_all / h_all.sum(), where='mid', color='k', lw=2, label='no cut')
for keep in [0.10, 0.01, 0.001]:
    thr = np.quantile(p, 1 - keep)
    h, _ = np.histogram(appl.fLStarMass[p > thr], bins=mbins)
    if h.sum() > 0:
        ax.step(mctr, h / h.sum(), where='mid', label='top {:g}%'.format(keep * 100))
ax.axvspan(bl.MASS_WINDOW_MIN, bl.MASS_WINDOW_MAX, color='salmon', alpha=0.25,
           label='excluded from training')
ax.set_xlabel(r'$M(\Lambda\gamma)$ (GeV/$c^2$)', fontsize=12)
ax.set_ylabel('normalised counts', fontsize=12)
ax.set_title('Pure-background mass shape\nbefore and after the BDT cut', fontsize=12)
ax.legend(fontsize=8)
ax.grid(ls='--', alpha=0.4)

ax = axes[2]
sub = data.sample(n=min(len(data), 300000), random_state=3)
ax.hist2d(sub.fLStarMass, sub.fOPAngle,
          bins=[np.linspace(1.1, 2.5, 120), np.linspace(0, 3.2, 120)],
          norm=matplotlib.colors.LogNorm(), cmap='viridis')
ax.axvline(bl.MASS_WINDOW_MIN, color='r', ls='--', lw=1.5)
ax.axvline(bl.MASS_WINDOW_MAX, color='r', ls='--', lw=1.5)
ax.set_xlabel(r'$M(\Lambda\gamma)$ (GeV/$c^2$)', fontsize=12)
ax.set_ylabel('fOPAngle (rad)', fontsize=12)
ax.set_title('fOPAngle vs mass in data\nred lines = the background definition', fontsize=12)
fig.tight_layout()
bl.savefig(fig, 'bias_04_mass_sculpting.png')

R.save()
