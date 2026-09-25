"""
bias_08 -- would a maximum-pT cut on the MC fix the spectrum mismatch?

The idea: the MC is flat in [0, 20] GeV/c while the data background dies above
~4, so cut the MC at the data's upper edge and the two samples live in the same
range.  That removes the *range* mismatch.  The question this macro answers is
whether it removes the *shape* mismatch, which is what the BDT actually exploits.

Inside any surviving window the MC is still flat and the data still falls
steeply, so a classifier can keep answering "hard or soft?".  Measured here:

  cut only        truncate both samples at X, train, no reweighting
  cut + reweight  same, plus matching the background pT shape to the signal
  placebo         B_MC vs B_DATA in the same window (no physics difference,
                  so whatever it scores is what "cut only" is riding on)

plus the signal statistics left after each cut, since that is the real cost.

Everything uses the polarity-repaired MC and a group-aware split, so the only
thing varying is the kinematic treatment.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import biaslib as bl

R = bl.Report('bias_08_ptcut_vs_reweight')

mc = bl.load_mc()
data = bl.load_data()
is_lstar = np.abs(mc.fPDGCode) == bl.LSTAR_PDG
S_MC = bl.fix_dca_polarity(mc[is_lstar].copy())
B_MC = bl.fix_dca_polarity(mc[(~is_lstar) & bl.in_sideband(mc)].copy())
B_DATA = data[bl.in_sideband(data)]

ptbins = np.concatenate([np.linspace(0, 12, 49), [14, 17, 20, 25, 30]])
NB = 20000

# ------------------------------------------------------- where does the data end?
R.head('WHERE DOES THE DATA BACKGROUND ACTUALLY END?')
qs = [0.50, 0.90, 0.95, 0.99, 0.999, 0.9999, 1.0]
v = B_DATA.fLStarPt.values
R('  n = {} sideband candidates'.format(len(v)))
for q in qs:
    R('    q{:<8.4g} = {:7.2f} GeV/c   ({} candidates above)'.format(
        q, np.quantile(v, q), int(round(len(v) * (1 - q)))))
R('')
R('  The "data maximum" is a single outlier at {:.1f} GeV/c; matching it would'.format(v.max()))
R('  cut nothing at all.  Any useful cut has to sit near where the data still')
R('  has statistics, i.e. 3-5 GeV/c -- and that is where most of the MC signal')
R('  is NOT.  {:.0f}% of the MC signal sits above 4 GeV/c.'.format(
    100 * (S_MC.fLStarPt > 4).mean()))


def spectra_overlap(a, b, lo, hi):
    e = np.linspace(lo, hi, 40)
    ha, _ = np.histogram(a, bins=e, density=True)
    hb, _ = np.histogram(b, bins=e, density=True)
    return float(np.minimum(ha, hb).sum() * np.diff(e).mean())


# --------------------------------------------------------------- the scan
R.head('SCAN OVER THE MAXIMUM-pT CUT')
R('  {:>8s} {:>7s} {:>8s} {:>9s} {:>9s} {:>10s} {:>12s} {:>10s}'.format(
    'cut', 'n_sig', 'overlap', 'medMC', 'medDATA', 'cut only', 'cut+reweight', 'placebo'))

rows = []
for X in [2.0, 3.0, 4.0, 5.0, 6.0, 20.0]:
    s_ = S_MC[S_MC.fLStarPt < X]
    bd_ = B_DATA[B_DATA.fLStarPt < X]
    bm_ = B_MC[B_MC.fLStarPt < X]
    if len(s_) < 80 or len(bd_) < 2000:
        R('  {:>8s} {:7d}   skipped (too few)'.format('<{:g}'.format(X), len(s_)))
        continue

    bd_s = bd_.sample(n=min(len(bd_), NB), random_state=1)
    bm_s = bm_.sample(n=min(len(bm_), NB), random_state=1)

    ov = spectra_overlap(s_.fLStarPt.values, bd_.fLStarPt.values, 0, X)

    a_cut, _, _ = bl.fit_and_score(s_, bd_s, group_split=True)
    w = bl.reweight_to(bd_s.fLStarPt.values, s_.fLStarPt.values, ptbins)
    a_rw, _, _ = bl.fit_and_score(s_, bd_s, group_split=True, w_neg=w)
    a_pl, _, _ = bl.fit_and_score(bm_s, bd_s, group_split=True)

    R('  {:>8s} {:7d} {:8.3f} {:9.2f} {:9.2f} {:10.4f} {:12.4f} {:10.4f}'.format(
        '<{:g}'.format(X), len(s_), ov, np.median(s_.fLStarPt), np.median(bd_.fLStarPt),
        a_cut, a_rw, a_pl))
    rows.append((X, len(s_), ov, a_cut, a_rw, a_pl))

R('')
R('  (cut = 20 is "no cut": the MC generator range, shown for reference)')

# --------------------------------------------------------------- interpretation
R.head('WHAT THE NUMBERS SAY')
if rows:
    tab = pd.DataFrame(rows, columns=['X', 'n_sig', 'overlap', 'cut', 'rw', 'placebo'])
    best = tab.iloc[(tab.X - 4.0).abs().argmin()]
    nocut = tab[tab.X == 20.0]
    R('  At a 4 GeV/c cut:')
    R('    signal left        : {} of {} candidates ({:.0f}% thrown away)'.format(
        int(best.n_sig), len(S_MC), 100 * (1 - best.n_sig / len(S_MC))))
    R('    spectra overlap    : {:.3f}  (1 = identical shapes)'.format(best.overlap))
    R('    AUC cut only       : {:.4f}'.format(best.cut))
    R('    AUC cut + reweight : {:.4f}   <- the cut alone leaves {:+.3f}'.format(
        best.rw, best.cut - best.rw))
    R('    AUC placebo        : {:.4f}   <- still no physics in it'.format(best.placebo))
    if len(nocut):
        R('')
        R('  For comparison, no cut at all but reweighted: {:.4f}'.format(
            float(nocut.rw.iloc[0])))
R('')
R('  A maximum-pT cut fixes the RANGE but not the SHAPE.  Truncating a flat')
R('  spectrum at 4 GeV/c leaves a flat spectrum in [0, 4]; the data background')
R('  is still falling steeply across that same window, so "hard or soft?"')
R('  remains available and the placebo stays well above 0.5.')
R('')
R('  It also costs most of the signal, and it costs it exactly where the')
R('  analysis is most interesting: after the cut the model has never seen a')
R('  high-pT L*, so it cannot be applied above the cut at all.')
R('')
R('  Reweighting keeps every candidate and matches the shape, so it dominates')
R('  the cut on both axes.  If a cut is wanted anyway, the honest version is a')
R('  LOWER bound as well -- train inside a window where both samples have')
R('  statistics -- or, better, train in pT bins, which the task already')
R('  supports (it plots AUC vs pT).')

# -------------------------------------------------------------------------- plot
if rows:
    tab = pd.DataFrame(rows, columns=['X', 'n_sig', 'overlap', 'cut', 'rw', 'placebo'])
    fin = tab[tab.X < 20]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.plot(fin.X, fin.cut, 'o-', color='crimson', label='cut only')
    ax.plot(fin.X, fin.rw, 's-', color='seagreen', label='cut + pT reweight')
    ax.plot(fin.X, fin.placebo, '^-', color='darkorange', label='placebo (no physics)')
    ax.axhline(0.5, color='k', ls=':', lw=1)
    ax.set_xlabel(r'maximum $p_T$ cut (GeV/$c$)', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_ylim(0.45, 1.02)
    ax.set_title('A cut does not close the gap', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(ls='--', alpha=0.4)

    ax = axes[1]
    ax.plot(fin.X, 100 * fin.n_sig / len(S_MC), 'o-', color='navy')
    ax.set_xlabel(r'maximum $p_T$ cut (GeV/$c$)', fontsize=11)
    ax.set_ylabel('% of MC signal surviving', fontsize=11)
    ax.set_title('...and it costs most of the signal', fontsize=12)
    ax.grid(ls='--', alpha=0.4)

    ax = axes[2]
    e = np.linspace(0, 8, 50)
    ax.hist(S_MC.fLStarPt, bins=e, density=True, histtype='step', lw=2,
            color='salmon', label='MC signal (flat)')
    ax.hist(B_DATA.fLStarPt.values, bins=e, density=True, histtype='step', lw=2,
            color='royalblue', label='data bkg (falling)')
    ax.axvline(4.0, color='k', ls='--', lw=1.5, label='cut at 4 GeV/c')
    ax.set_yscale('log')
    ax.set_xlabel(r'$p_{T}$ (GeV/$c$)', fontsize=11)
    ax.set_ylabel('normalised', fontsize=11)
    ax.set_title('Below the cut the shapes still disagree', fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    bl.savefig(fig, 'bias_08_ptcut_vs_reweight.png')

R.save()
