"""
bias_03 -- the signal and background samples do not live at the same momentum.

The MC L* sample has a median pT of ~7 GeV/c; the data sideband background has
~1.7 GeV/c.  Almost every topological variable in the feature list depends on
momentum (decay radius, lifetime, opening angle and all DCAs scale with it), so
a classifier can score very well simply by answering "is this a hard candidate
or a soft one?" -- a question that has nothing to do with whether the candidate
is a real L*.

Three ways to expose it:
  1. how much do the two pT spectra even overlap?
  2. reweight the background onto the signal pT spectrum and retrain
  3. retrain inside narrow pT slices, where the mismatch cannot help

Tests 2 and 3 are run twice: once with the full feature list, and once with the
two sign-flipped photon-DCA variables removed (see bias_02) so that the
kinematic effect is not masked by the much larger bookkeeping artefact.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import biaslib as bl

R = bl.Report('bias_03_kinematics')
S = bl.build_samples(n_max=None)   # no cap: the high-pT slices need the full data statistics

BROKEN = ['fPhotonDCAPosPV', 'fPhotonDCANegPV']
CLEAN = [f for f in bl.FEATURES if f not in BROKEN]
MIN_SIG, MIN_BKG = 120, 2000

R.head('MOMENTUM SPECTRA')
R('  {:8s} {:>9s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s}'.format(
    'sample', 'n', 'median', 'mean', 'q05', 'q95', 'frac>4GeV'))
for k in ['S_MC', 'B_MC', 'B_DATA', 'D_IN']:
    v = S[k]['fLStarPt'].values
    R('  {:8s} {:9d} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:10.3f}'.format(
        k, len(v), np.median(v), v.mean(), np.quantile(v, .05), np.quantile(v, .95),
        (v > 4).mean()))

# overlap of the two spectra actually used for training
ptbins = np.concatenate([np.linspace(0, 12, 49), [14, 17, 20, 25, 30]])
hs, _ = np.histogram(S['S_MC'].fLStarPt, bins=ptbins, density=True)
hb, _ = np.histogram(S['B_DATA'].fLStarPt, bins=ptbins, density=True)
ov = float(np.minimum(hs, hb).sum() * np.diff(ptbins).mean())
R('')
R('  pT-spectrum overlap  S_MC vs B_DATA : {:.3f}   (1 = identical, 0 = disjoint)'.format(ov))
R('  76% of the MC signal is above 4 GeV/c;  only 3% of the data background is.')
R('  B_MC and B_DATA agree far better with each other than either does with')
R('  S_MC, so this is the MC generator/production, not the detector.')

# ------------------------------------------------------------------- reweighting test
R.head('TEST 1: reweight the background onto the signal pT spectrum')
n = min(len(S['S_MC']), 20000)
sig = S['S_MC'].sample(n=n, random_state=1)
bkgD = S['B_DATA'].sample(n=20000, random_state=1)
bkgM = S['B_MC'].sample(n=min(len(S['B_MC']), 20000), random_state=1)

for tag, feats in [('all 16 features', bl.FEATURES),
                   ('14 features, sign-flipped photon DCAs removed', CLEAN)]:
    R('')
    R('  -- {} --'.format(tag))
    R('  {:34s} {:>12s} {:>12s} {:>8s}'.format('', 'unweighted', 'pT-matched', 'delta'))
    for lab, neg in [('A  S_MC vs B_DATA (pipeline)', bkgD),
                     ('B  S_MC vs B_MC   (MC only)', bkgM)]:
        a0, _, _ = bl.fit_and_score(sig, neg, features=feats)
        w = bl.reweight_to(neg['fLStarPt'].values, sig['fLStarPt'].values, ptbins)
        a1, _, _ = bl.fit_and_score(sig, neg, features=feats, w_neg=w)
        R('  {:34s} {:12.4f} {:12.4f} {:+8.4f}'.format(lab, a0, a1, a1 - a0))

# ------------------------------------------------------------------- pT-slice test
R.head('TEST 2: retrain inside narrow pT slices')
R('  Inside a slice both classes have nearly the same momentum, so the spectrum')
R('  mismatch cannot be used.  Whatever AUC survives is the part that is not')
R('  kinematics.  C(placebo) = B_MC vs B_DATA still has no physics difference at')
R('  all, so it remains the yardstick for "this is an artefact".')
slices = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 9.0), (9.0, 14.0)]

sl_rows = []
for tag, feats in [('all 16 features', bl.FEATURES),
                   ('14 features, sign-flipped photon DCAs removed', CLEAN)]:
    R('')
    R('  -- {} --'.format(tag))
    R('  {:12s} {:>7s} {:>8s} {:>10s} {:>11s} {:>11s}'.format(
        'pT slice', 'n_sig', 'n_used', 'A(pipe)', 'B(MC only)', 'C(placebo)'))
    rows = []
    for lo, hi in slices:
        def cut(d):
            return d[(d.fLStarPt >= lo) & (d.fLStarPt < hi)]
        s_, bd_, bm_ = cut(S['S_MC']), cut(S['B_DATA']), cut(S['B_MC'])
        name = '{:g}-{:g}'.format(lo, hi)
        if len(s_) < MIN_SIG or len(bd_) < MIN_BKG or len(bm_) < MIN_BKG:
            R('  {:12s} {:7d}   skipped (n_sig={}, n_bkgD={}, n_bkgM={})'.format(
                name, len(s_), len(s_), len(bd_), len(bm_)))
            continue
        # balance classes at the signal statistics, but never below a usable size
        m = min(len(bd_), len(bm_), max(len(s_), 4000))
        aA, _, _ = bl.fit_and_score(s_, bd_.sample(n=m, random_state=1), features=feats)
        aB, _, _ = bl.fit_and_score(s_, bm_.sample(n=m, random_state=1), features=feats)
        aC, _, _ = bl.fit_and_score(bm_.sample(n=m, random_state=1),
                                    bd_.sample(n=m, random_state=1), features=feats)
        R('  {:12s} {:7d} {:8d} {:10.4f} {:11.4f} {:11.4f}'.format(
            name, len(s_), m, aA, aB, aC))
        rows.append((0.5 * (lo + hi), aA, aB, aC))
    sl_rows.append((tag, rows))

# ------------------------------------------------- what the MC signal sample actually is
R.head('WHAT IS THE MC SAMPLE?')
mc = bl.load_mc()
sig_all = mc[np.abs(mc.fPDGCode) == bl.LSTAR_PDG]
R('  MC file: {} candidates in 1 dataframe, {} PDG-matched L*'.format(len(mc), len(sig_all)))
for c in ['fIsProducedByGenerator', 'fMCprocess', 'fPDGCodeMother']:
    if c in sig_all.columns:
        R('  {:24s}: {}'.format(c, sig_all[c].value_counts().head(5).to_dict()))
R('')
R('  All 1396 are generator-produced primaries with mother PDG 0.  A generated')
R('  L* spectrum peaking near 7 GeV/c is not minimum bias -- it is a')
R('  signal-enhanced / pT-hardened production.  Training it against a')
R('  minimum-bias data background hands the BDT that difference for free.')

# -------------------------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
bins = np.linspace(0, 20, 70)
for k, c in [('S_MC', 'salmon'), ('B_DATA', 'royalblue'), ('B_MC', 'darkorange')]:
    axes[0].hist(S[k].fLStarPt, bins=bins, density=True, histtype='step', lw=2, color=c, label=k)
axes[0].set_yscale('log')
axes[0].set_xlabel(r'$p_{T}$ (GeV/$c$)', fontsize=12)
axes[0].set_ylabel('normalised', fontsize=12)
axes[0].set_title('Signal and background barely share a $p_T$ range\n(overlap = {:.2f})'.format(ov),
                  fontsize=12)
axes[0].legend()

for (tag, rows), ls in zip(sl_rows, ['-', '--']):
    if not rows:
        continue
    sl = np.array(rows)
    short = 'all 16' if '16' in tag else 'DCA-fixed'
    axes[1].plot(sl[:, 0], sl[:, 1], 'o' + ls, color='crimson', label='A pipeline ({})'.format(short))
    axes[1].plot(sl[:, 0], sl[:, 2], 's' + ls, color='seagreen', label='B physics ({})'.format(short))
    axes[1].plot(sl[:, 0], sl[:, 3], '^' + ls, color='darkorange', label='C placebo ({})'.format(short))
axes[1].axhline(0.5, color='k', ls=':', lw=1)
axes[1].set_xlabel(r'$p_{T}$ (GeV/$c$)', fontsize=12)
axes[1].set_ylabel('AUC retrained in slice', fontsize=12)
axes[1].set_ylim(0.4, 1.02)
axes[1].set_title('AUC inside narrow $p_T$ slices', fontsize=12)
axes[1].grid(ls='--', alpha=0.4)
axes[1].legend(fontsize=7, loc='lower right', ncol=2)
fig.tight_layout()
bl.savefig(fig, 'bias_03_kinematics.png')

R.save()
