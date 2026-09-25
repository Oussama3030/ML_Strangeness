"""
bias_02 -- which variables carry the bias, one at a time.

bias_01 showed that MC background and data background are separable at
AUC ~ 1.  This macro attributes that to individual features.

For every training feature we quote three single-feature AUCs:

    A_i   S_MC vs B_DATA   what the pipeline sees
    B_i   S_MC vs B_MC     physics only  (same origin)
    C_i   B_MC vs B_DATA   origin only   (same physics)

A feature with a large C_i is a simulation/data mismatch: the BDT will use it,
and whatever it buys on the test set will not transfer to real data.  A feature
with a large B_i and small C_i is genuinely useful.

Single-feature AUC is computed from the raw variable (no training needed), so
there is nothing to overfit: AUC = P(x_signal > x_background), symmetrised.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

import biaslib as bl

R = bl.Report('bias_02_feature_audit')
S = bl.build_samples()


def single_auc(pos, neg, feat):
    x = np.r_[pos[feat].values, neg[feat].values].astype(float)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    ok = np.isfinite(x)
    if ok.sum() < 10 or len(np.unique(x[ok])) < 2:
        return np.nan
    a = roc_auc_score(y[ok], x[ok])
    return max(a, 1 - a)          # direction-free separation power


def overlap(pos, neg, feat, nbins=80):
    """Histogram overlap coefficient: 1 = identical shapes, 0 = disjoint support."""
    a = pos[feat].values.astype(float); b = neg[feat].values.astype(float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    lo = min(np.percentile(a, 0.1), np.percentile(b, 0.1))
    hi = max(np.percentile(a, 99.9), np.percentile(b, 99.9))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    bins = np.linspace(lo, hi, nbins)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    w = bins[1] - bins[0]
    return float(np.minimum(ha, hb).sum() * w)


R.head('SINGLE-FEATURE SEPARATION POWER')
R('  A = S_MC vs B_DATA (pipeline)   B = S_MC vs B_MC (physics only)   C = B_MC vs B_DATA (origin only)')
R('  ovl = MC/data histogram overlap for background only (1 = perfect agreement)')
R('')
rows = []
for f in bl.FEATURES:
    a = single_auc(S['S_MC'], S['B_DATA'], f)
    b = single_auc(S['S_MC'], S['B_MC'], f)
    c = single_auc(S['B_MC'], S['B_DATA'], f)
    ks = ks_2samp(S['B_MC'][f].dropna(), S['B_DATA'][f].dropna()).statistic
    ov = overlap(S['B_MC'], S['B_DATA'], f)
    rows.append((f, a, b, c, ks, ov))

tab = pd.DataFrame(rows, columns=['feature', 'A_pipeline', 'B_physics', 'C_origin',
                                  'KS_mc_data', 'ovl_mc_data'])
tab = tab.sort_values('C_origin', ascending=False)
R(tab.to_string(index=False, float_format=lambda v: '{:.4f}'.format(v)))

R('')
bad = tab[tab.C_origin > 0.70]
R('  Features whose MC/data mismatch alone gives AUC > 0.70  ({} of {}):'.format(len(bad), len(tab)))
for _, r in bad.iterrows():
    R('    {:20s} C = {:.3f}   overlap = {:.3f}'.format(r.feature, r.C_origin, r.ovl_mc_data))

# ---------------------------------------------------------------- cumulative ablation
R.head('ABLATION: drop the worst MC/data offenders one by one')
R('  Each row retrains A (pipeline) and C (placebo) after removing the listed')
R('  features.  If A stays high only while C stays high, A is riding on C.')
R('')
order = list(tab.feature)
kept = list(bl.FEATURES)
n = min(len(S['S_MC']), 20000)
R('  {:38s} {:>10s} {:>10s}'.format('features used', 'A(pipe)', 'C(placebo)'))
for drop_n in range(0, 7):
    kept = [f for f in bl.FEATURES if f not in order[:drop_n]]
    if len(kept) < 2:
        break
    aucA, _, _ = bl.fit_and_score(S['S_MC'].sample(n=n, random_state=1),
                                  S['B_DATA'].sample(n=n, random_state=1), features=kept)
    aucC, _, _ = bl.fit_and_score(S['B_MC'].sample(n=20000, random_state=1),
                                  S['B_DATA'].sample(n=20000, random_state=1), features=kept)
    lab = 'all {}'.format(len(kept)) if drop_n == 0 else '-{}'.format(', '.join(order[:drop_n]))
    R('  {:38s} {:10.4f} {:10.4f}'.format(lab[:38], aucA, aucC))

# ------------------------------------------------------------------------------ plots
SENTINEL = 900.0     # fPhoton*DCA*PV uses 999 for "not computed"


def finite_range(arrays, lo_q=0.5, hi_q=99.5):
    v = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    v = v[np.isfinite(v) & (np.abs(v) < SENTINEL)]
    return np.nanpercentile(v, lo_q), np.nanpercentile(v, hi_q)


worst = list(tab.feature[:6])
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))
for ax, f in zip(axes.ravel(), worst):
    lo, hi = finite_range([S['B_MC'][f].values, S['B_DATA'][f].values])
    bins = np.linspace(lo, hi, 60)
    for k, col, style in [('B_DATA', 'royalblue', dict(histtype='step', lw=2)),
                          ('B_MC', 'darkorange', dict(histtype='step', lw=2)),
                          ('S_MC', 'salmon', dict(histtype='stepfilled', alpha=0.35))]:
        v = S[k][f].values.astype(float)
        v = v[np.abs(v) < SENTINEL]
        ax.hist(v, bins=bins, density=True, color=col, label=k, **style)
    nsent = ['{}: {:.0f}%'.format(k, 100 * np.mean(np.abs(S[k][f].values) > SENTINEL))
             for k in ['S_MC', 'B_MC', 'B_DATA']]
    extra = ''
    if any(not x.endswith(' 0%') for x in nsent):
        extra = '\nat sentinel 999 -- ' + ', '.join(nsent)
    ax.set_title('{}   C(origin) = {:.3f}{}'.format(
        f, float(tab[tab.feature == f].C_origin.iloc[0]), extra), fontsize=9)
    ax.set_xlabel(f, fontsize=9)
    ax.legend(fontsize=7)
fig.suptitle('Both backgrounds are combinatorial: orange vs blue is pure MC/data mismatch',
             fontsize=13)
fig.tight_layout()
bl.savefig(fig, 'bias_02_worst_features.png')

# ------------------------------------------------------------------ the root cause
# The two worst variables are not merely mis-modelled: they have the opposite
# SIGN in MC and in data, because the positive and negative daughter of the V0
# are interchanged between the two productions.
R.head('ROOT CAUSE: THE POSITIVE AND NEGATIVE DAUGHTER ARE INTERCHANGED')
R('  Median DCA-to-primary-vertex, sentinel (999) excluded:')
R('')
R('  {:20s} {:>12s} {:>12s} {:>12s}'.format('variable', 'S_MC', 'B_MC', 'B_DATA'))
dca = ['fPhotonDCAPosPV', 'fPhotonDCANegPV', 'fLambdaDCAPosPV', 'fLambdaDCANegPV']
for f in dca:
    med = []
    for k in ['S_MC', 'B_MC', 'B_DATA']:
        v = S[k][f].values.astype(float)
        med.append(np.median(v[np.abs(v) < SENTINEL]))
    R('  {:20s} {:+12.3f} {:+12.3f} {:+12.3f}'.format(f, *med))
R('')
R('  Both MC columns carry one sign and both data columns the other.  A single')
R('  threshold at zero therefore tells simulation from data almost perfectly,')
R('  which is exactly the C = 0.9999 of bias_01.')
R('')
R('  Swapping the Pos/Neg columns in the MC removes it:')
R('  {:20s} {:>12s} {:>12s}'.format('variable', 'C before', 'C after'))
# only the MC samples are repaired -- the data is the reference and is left alone
S_fix = {k: (bl.fix_daughter_swap(v) if k in ('S_MC', 'B_MC') else v)
         for k, v in S.items()}
for f in dca:
    R('  {:20s} {:12.4f} {:12.4f}'.format(
        f, single_auc(S['B_MC'], S['B_DATA'], f),
        single_auc(S_fix['B_MC'], S['B_DATA'], f)))

fig3, axes = plt.subplots(2, 4, figsize=(19, 8))
for j, f in enumerate(dca):
    lo, hi = finite_range([S['B_MC'][f].values, S['B_DATA'][f].values], 1, 99)
    lim = max(abs(lo), abs(hi))
    bins = np.linspace(-lim, lim, 70)
    for i, (src, ttl) in enumerate([(S, 'as produced'), (S_fix, 'MC Pos/Neg swapped back (data untouched)')]):
        ax = axes[i, j]
        for k, col, style in [('B_DATA', 'royalblue', dict(histtype='step', lw=2)),
                              ('B_MC', 'darkorange', dict(histtype='step', lw=2)),
                              ('S_MC', 'salmon', dict(histtype='stepfilled', alpha=0.35))]:
            v = src[k][f].values.astype(float)
            v = v[np.abs(v) < SENTINEL]
            ax.hist(v, bins=bins, density=True, color=col, label=k, **style)
        ax.axvline(0, color='k', lw=1, ls=':')
        ax.set_title('{}\n{}  (AUC {:.3f})'.format(
            f, ttl, single_auc(src['B_MC'], S['B_DATA'], f)), fontsize=9)
        ax.set_xlabel(f, fontsize=8)
        if j == 0:
            ax.legend(fontsize=7)
fig3.suptitle('Top: MC and data sit on opposite sides of zero.  '
              'Bottom: one Pos/Neg swap puts them back on top of each other.', fontsize=13)
fig3.tight_layout()
bl.savefig(fig3, 'bias_02_root_cause_swap.png')

fig2, ax = plt.subplots(figsize=(9, 6))
yy = np.arange(len(tab))
ax.barh(yy + 0.2, tab.C_origin, height=0.38, color='darkorange', label='C: origin only (MC vs data bkg)')
ax.barh(yy - 0.2, tab.B_physics, height=0.38, color='seagreen', label='B: physics only (MC sig vs MC bkg)')
ax.axvline(0.5, color='k', ls='--', lw=1)
ax.set_yticks(yy); ax.set_yticklabels(tab.feature, fontsize=9)
ax.set_xlabel('single-feature AUC', fontsize=12)
ax.set_xlim(0.4, 1.0)
ax.set_title('Per-feature: how much is physics, how much is simulation/data mismatch', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', ls='--', alpha=0.4)
bl.savefig(fig2, 'bias_02_physics_vs_origin.png')

R.save()
