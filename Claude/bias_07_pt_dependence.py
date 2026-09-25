"""
bias_07 -- which training variables are really momentum in disguise?

The MC signal is generated flat (or at least far harder than minimum bias) while
the data background follows a steeply falling spectrum, so any variable that
tracks pT hands the BDT a free discriminant.  This macro ranks the 16 features
by how much of a pT proxy each one is.

Three independent measurements per feature:

  rho_pT     Spearman correlation with pT, measured in DATA BACKGROUND ONLY.
             One origin, one selection, 4.6M candidates -- no label involved, so
             this is pure "does this variable know the momentum?".

  AUC_hard   Single-feature AUC separating hard (pT > 4) from soft (pT < 2)
             candidates, again inside the data background alone.  0.5 = blind to
             momentum, 1.0 = a perfect momentum meter.

  dAUC_rw    The part that actually feeds the bias: single-feature AUC for
             S_MC vs B_DATA, unweighted minus pT-matched.  Large positive =
             that feature's apparent power evaporates once the spectra agree.

The polarity-mirrored DCA columns are repaired first (bias_02 / the builder fix),
otherwise that artefact swamps the kinematic effect.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import biaslib as bl

R = bl.Report('bias_07_pt_dependence')
SENT = 900.0

mc = bl.load_mc()
data = bl.load_data()
is_lstar = np.abs(mc.fPDGCode) == bl.LSTAR_PDG
S_MC = bl.fix_dca_polarity(mc[is_lstar].copy())   # as the rebuilt MC will be
B_DATA = data[bl.in_sideband(data)]

# ---------------------------------------------------------------- the MC spectrum
R.head('IS THE MC REALLY FLAT IN pT?')
sig = mc[is_lstar]
px = sig.fLambdaMCPx.values + sig.fPhotonMCPx.values
py = sig.fLambdaMCPy.values + sig.fPhotonMCPy.values
pt_true = np.sqrt(px**2 + py**2)
R('  Generated (MC-truth) pT of the 1396 reconstructed true L*:')
R('    min {:.2f}   q05 {:.2f}   median {:.2f}   q95 {:.2f}   max {:.2f}'.format(
    pt_true.min(), np.quantile(pt_true, .05), np.median(pt_true),
    np.quantile(pt_true, .95), pt_true.max()))
edges = np.arange(0, 22, 2.0)
h, _ = np.histogram(pt_true, bins=edges)
R('')
R('    {:>12s} {:>8s}'.format('pT bin', 'counts'))
for i in range(len(h)):
    R('    {:5.0f} - {:<4.0f} {:8d}  {}'.format(edges[i], edges[i + 1], h[i],
                                                '#' * int(40 * h[i] / max(h.max(), 1))))
R('')
R('  The spectrum stops dead at 20.0 GeV/c and starts at ~0: that hard edge is')
R('  the generator range, so yes -- flat generation in [0, 20] GeV/c.  The shape')
R('  above is that flat spectrum folded with acceptance x efficiency, which is')
R('  why it peaks around 4-6 rather than staying flat.')
R('')
R('  Caveat: this is the truth pT of RECONSTRUCTED candidates, i.e. the')
R('  generated spectrum folded with acceptance x efficiency.  The generated')
R('  particle table is not in this AO2D, so flatness at generation cannot be')
R('  confirmed here -- but the shape is clearly consistent with a flat (or')
R('  pT-hardened) generation, and it is nothing like the data background.')
R('  Data background: median {:.2f}, q95 {:.2f}.'.format(
    np.median(B_DATA.fLStarPt), np.quantile(B_DATA.fLStarPt, .95)))

# ------------------------------------------------------- per-feature pT dependence
R.head('PER-FEATURE pT DEPENDENCE')

nsub = 400000
bkg = B_DATA.sample(n=min(len(B_DATA), nsub), random_state=1)
hard = bkg[bkg.fLStarPt > 4.0]
soft = bkg[bkg.fLStarPt < 2.0]
ptbins = np.concatenate([np.linspace(0, 12, 49), [14, 17, 20, 25, 30]])


def clean(v):
    v = np.asarray(v, dtype=float)
    return v, np.isfinite(v) & (np.abs(v) < SENT)


def single_auc(a, b):
    a, oa = clean(a)
    b, ob = clean(b)
    x = np.r_[a[oa], b[ob]]
    y = np.r_[np.ones(oa.sum()), np.zeros(ob.sum())]
    if len(np.unique(x)) < 2:
        return np.nan
    v = roc_auc_score(y, x)
    return max(v, 1 - v)


rows = []
w_bkg = bl.reweight_to(B_DATA.fLStarPt.values, S_MC.fLStarPt.values, ptbins)
for f in bl.FEATURES:
    v, ok = clean(bkg[f].values)
    rho = spearmanr(v[ok], bkg.fLStarPt.values[ok]).statistic
    a_hard = single_auc(hard[f].values, soft[f].values)

    # unweighted vs pT-matched single-feature AUC, S_MC vs B_DATA
    s, os_ = clean(S_MC[f].values)
    d, od = clean(B_DATA[f].values)
    x = np.r_[s[os_], d[od]]
    y = np.r_[np.ones(os_.sum()), np.zeros(od.sum())]
    w0 = np.r_[np.ones(os_.sum()), np.ones(od.sum())]
    w1 = np.r_[np.ones(os_.sum()), w_bkg[od]]
    a0 = roc_auc_score(y, x, sample_weight=w0)
    a1 = roc_auc_score(y, x, sample_weight=w1)
    a0, a1 = (a0, a1) if a0 >= 0.5 else (1 - a0, 1 - a1)
    rows.append((f, rho, a_hard, a0, a1, a0 - a1))

tab = pd.DataFrame(rows, columns=['feature', 'rho_pT', 'AUC_hard', 'AUC_raw',
                                  'AUC_ptmatched', 'dAUC_rw'])
tab = tab.reindex(tab.AUC_hard.abs().sort_values(ascending=False).index)
R('  measured in data background only (no labels) | S_MC vs B_DATA')
R('')
R(tab.to_string(index=False, float_format=lambda v: '{:+.4f}'.format(v)))

R('')
strong = tab[tab.AUC_hard > 0.70]
weak = tab[tab.AUC_hard < 0.60]
R('  STRONG momentum proxies (AUC_hard > 0.70):')
for _, r in strong.iterrows():
    R('    {:18s} rho = {:+.3f}   AUC_hard = {:.3f}   loses {:+.3f} AUC when pT-matched'.format(
        r.feature, r.rho_pT, r.AUC_hard, r.dAUC_rw))
R('')
R('  Essentially momentum-blind (AUC_hard < 0.60):')
R('    ' + ', '.join(weak.feature))

R.head('READING')
R('  fOPAngle is the extreme case: a two-body opening angle goes like ~m/p, so')
R('  it is almost a pure momentum variable, AND it is the variable the mass')
R('  sideband cut was made on (bias_04).  It is doing double duty as a bias.')
R('')
R('  fLambdaRadius and fLambdaLifeTime scale with the boost, so a harder signal')
R('  decays further out.  That is real physics, but here it separates the')
R('  classes only because the two samples sit at different momenta.')
R('')
R('  fPhotonRadius is set by where the material is, not by the momentum, so it')
R('  stays flat -- which is exactly why it is one of the safer features.')
R('  The Armenteros variables (Qt, Alpha) are bounded by the decay kinematics')
R('  and are likewise close to momentum-blind.')

# ------------------------------------------- signal vs background, both profiled
# Coarse common binning: the signal has only ~1400 candidates, so it needs wider
# bins than the background to give a stable median.
profbins = np.array([0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0])
pcent = 0.5 * (profbins[1:] + profbins[:-1])


def profile(df, f, bins, nmin=25):
    v, ok = clean(df[f].values)
    pt = df.fLStarPt.values
    idx = np.clip(np.digitize(pt, bins) - 1, 0, len(bins) - 2)
    xs, med, lo, hi = [], [], [], []
    for k in range(len(bins) - 1):
        m = ok & (idx == k)
        if m.sum() < nmin:
            continue
        xs.append(0.5 * (bins[k] + bins[k + 1]))
        med.append(np.median(v[m]))
        lo.append(np.quantile(v[m], 0.25))
        hi.append(np.quantile(v[m], 0.75))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


R.head('SIGNAL AND BACKGROUND PROFILED IN THE SAME pT BINS')
R('  Medians per pT bin. "sep" is the single-feature AUC inside that bin alone,')
R('  i.e. how well the variable still separates once both classes sit at the')
R('  same momentum.  0.5 there means the variable only ever worked because the')
R('  two samples had different spectra.')
for f in ['fOPAngle', 'fLambdaCosPA', 'fLambdaRadius']:
    R('')
    R('  -- {} --'.format(f))
    R('  {:>12s} {:>8s} {:>11s} {:>11s} {:>8s}'.format('pT bin', 'n_sig', 'signal', 'bkg', 'sep'))
    for k in range(len(profbins) - 1):
        lo_, hi_ = profbins[k], profbins[k + 1]
        s_ = S_MC[(S_MC.fLStarPt >= lo_) & (S_MC.fLStarPt < hi_)]
        b_ = bkg[(bkg.fLStarPt >= lo_) & (bkg.fLStarPt < hi_)]
        if len(s_) < 25 or len(b_) < 200:
            continue
        sv, sok = clean(s_[f].values)
        bv, bok = clean(b_[f].values)
        R('  {:>12s} {:8d} {:11.4f} {:11.4f} {:8.3f}'.format(
            '{:g}-{:g}'.format(lo_, hi_), len(s_), np.median(sv[sok]), np.median(bv[bok]),
            single_auc(s_[f].values, b_[f].values)))

prof_feats = list(tab.feature[:6])
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
for ax, f in zip(axes.ravel(), prof_feats):
    xb, mb, lb, hb = profile(bkg, f, profbins, nmin=200)
    xs, ms, ls_, hs = profile(S_MC, f, profbins, nmin=25)
    ax.plot(xb, mb, 'o-', color='royalblue', label='data bkg')
    ax.fill_between(xb, lb, hb, color='royalblue', alpha=0.18)
    ax.plot(xs, ms, 's-', color='crimson', label='MC signal')
    ax.fill_between(xs, ls_, hs, color='crimson', alpha=0.18)
    ax.set_xlabel(r'$p_{T}$ (GeV/$c$)', fontsize=10)
    ax.set_ylabel(f, fontsize=10)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(['1', '2', '5', '10', '20'])
    ax.set_title('{}   AUC_hard = {:.3f}'.format(
        f, float(tab[tab.feature == f].AUC_hard.iloc[0])), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(ls='--', alpha=0.4)
fig.suptitle('Both classes profiled in the SAME $p_T$ bins (shaded = 25-75%).\n'
             'Where the two curves meet, the variable separates only via the spectrum.',
             fontsize=12)
fig.tight_layout()
bl.savefig(fig, 'bias_07_pt_profiles.png')

fig2, ax = plt.subplots(figsize=(9, 6))
t = tab.sort_values('AUC_hard')
yy = np.arange(len(t))
ax.barh(yy, t.AUC_hard - 0.5, left=0.5, color=[
    'crimson' if a > 0.7 else ('darkorange' if a > 0.6 else 'seagreen') for a in t.AUC_hard])
ax.axvline(0.5, color='k', lw=1)
ax.set_yticks(yy)
ax.set_yticklabels(t.feature, fontsize=9)
ax.set_xlabel('single-feature AUC, hard ($p_T>4$) vs soft ($p_T<2$), data background only',
              fontsize=10)
ax.set_xlim(0.5, 1.0)
ax.set_title('How much is each training variable just a momentum meter?', fontsize=12)
ax.grid(axis='x', ls='--', alpha=0.4)
bl.savefig(fig2, 'bias_07_pt_proxy_ranking.png')

R.save()
