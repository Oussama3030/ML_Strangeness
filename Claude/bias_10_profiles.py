"""
bias_10 -- one plot per variable: the per-pT-bin value, nothing else.

For every training variable, writes a single-axes figure to
output/Individual/<variable>_<stat>.png with the four samples overlaid:

  S_MC     MC true L*                            crimson
  B_MC     MC combinatorial background           orange
  B_DATA   data, mass outside 1.45-1.59          blue     <- the training background
  D_IN     data, mass inside 1.45-1.59           green

Band is 25-75% for the median, +/-1 sigma for the mean.
Set STATS below to pick which get written.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import biaslib as bl

STATS = ['mean', 'median']          # <- edit to ['mean'] or ['median'] if you want one

# Default binning is the one the analysis already fits in
# (plotMacros/Output/FitSpectrumLstar_pT_1.0_2.0, 2.0_3.0, 3.0_4.0, 4.0_5.2, 5.2_7.0),
# extended upwards to cover the MC's flat range.  With 1396 true L* this gives
# 66-254 signal candidates per bin; the old 12-bin grid had bins with only 12.
PTBINS = np.array([1.0, 2.0, 3.0, 4.0, 5.2, 7.0, 10.0, 14.0, 20.0])

# Set True to use bins of equal SIGNAL count instead (7 bins, ~200 each).
EQUAL_SIGNAL_BINS = False

# A bin is dropped unless it holds at least this many candidates.
NMIN = {'S_MC': 50, 'other': 500}
SENT = 998.0
FLOOR = 1e-7                        # float32 precision near cosPA = 1

COL = {'S_MC': 'crimson', 'B_MC': 'darkorange', 'B_DATA': 'royalblue', 'D_IN': 'seagreen'}
LBL = {'S_MC': 'S_MC  MC true L*',
       'B_MC': 'B_MC  MC combinatorial',
       'B_DATA': 'B_DATA  data, mass outside 1.45-1.59',
       'D_IN': 'D_IN  data, mass inside 1.45-1.59'}

OUT = os.path.join(bl.OUT_DIR, 'Individual')
os.makedirs(OUT, exist_ok=True)
R = bl.Report('bias_10_profiles')

mc = bl.load_mc()
data = bl.load_data()
is_lstar = np.abs(mc.fPDGCode) == bl.LSTAR_PDG
rng = np.random.RandomState(0)
SAMPLES = {
    'S_MC': bl.fix_dca_polarity(mc[is_lstar].copy()),
    'B_MC': bl.fix_dca_polarity(mc[(~is_lstar) & bl.in_sideband(mc)].copy()),
    'B_DATA': data[bl.in_sideband(data)].sample(n=600000, random_state=rng),
    'D_IN': data[~bl.in_sideband(data)].sample(n=600000, random_state=rng),
}


def is_cospa(f):
    return 'CosPA' in f


def values(df, f):
    """Finite, sentinel-free values; cosPA is returned as 1-cosPA so it is readable."""
    v = df[f].values.astype(float)
    ok = np.isfinite(v) & (np.abs(v) < SENT)
    if is_cospa(f):
        v = np.maximum(1.0 - v, FLOOR)
    return v, ok


# Error bars are 1 sigma (68%) for BOTH statistics, so the mean and median plots
# are directly comparable.  Z = 1.0; use 1.96 here if you want 95% instead.
Z = 1.0


def central_and_error(x, stat, inflate=1.0):
    """Central value, its 1-sigma statistical uncertainty, and the spread band.

    mean   : standard error of the mean, sigma/sqrt(n), symmetric.
    median : distribution-free interval from the order statistics.  The count of
             points below the true median is Binomial(n, 1/2), mean n/2 and
             sd sqrt(n)/2, so the 1-sigma interval is the pair of sorted values
             at ranks n/2 +/- Z*sqrt(n)/2.  This is asymmetric by construction
             and stays valid for the badly skewed variables, where sigma/sqrt(n)
             would be meaningless.

    `inflate` accounts for candidates that are not independent (see n_eff below):
    both errors are scaled by sqrt(n / n_eff).

    The shaded band on the plot is NOT this -- it is the width of the
    distribution (25-75% or +/-1 sigma) and does not shrink with statistics.
    """
    n = len(x)
    if stat == 'mean':
        mu = x.mean()
        sd = x.std(ddof=1) if n > 1 else 0.0
        err = inflate * sd / np.sqrt(n)
        return mu, err, err, mu - sd, mu + sd
    xs = np.sort(x)
    med = np.median(xs)
    half = Z * np.sqrt(n) / 2.0
    j = int(max(np.floor(n / 2.0 - half), 0))
    k = int(min(np.ceil(n / 2.0 + half), n - 1))
    return (med, inflate * (med - xs[j]), inflate * (xs[k] - med),
            np.percentile(xs, 25), np.percentile(xs, 75))


def owner_key(df, f):
    """Which V0 a variable belongs to.  Several candidates are built from the same
    reconstructed V0 and then share that V0's variables EXACTLY, so they are not
    independent measurements.  V0 ids restart in every dataframe, hence df_tag."""
    if f.startswith('fLambda'):
        col = 'fLambdaV0ID'
    elif f.startswith('fPhoton'):
        col = 'fPhotonV0ID'
    else:
        return None                      # fOPAngle, fDCADaughters: pair-level, one per candidate
    tag = df['df_tag'].astype(str) if 'df_tag' in df.columns else '?'
    return (tag + '_' + df[col].astype(str)).values


def profile(df, f, stat, nmin, bins):
    v, ok = values(df, f)
    idx = np.clip(np.digitize(df.fLStarPt.values, bins) - 1, 0, len(bins) - 2)
    key = owner_key(df, f)
    xs, mid, elo, ehi, lo, hi, ns = [], [], [], [], [], [], []
    for k in range(len(bins) - 1):
        m = ok & (idx == k)
        if m.sum() < nmin:
            continue
        # effective sample size: distinct V0s, not candidates
        if key is None:
            inflate = 1.0
        else:
            n_eff = max(len(np.unique(key[m])), 1)
            inflate = np.sqrt(m.sum() / n_eff)
        c, el, eh, b1, b2 = central_and_error(v[m], stat, inflate)
        xs.append(np.sqrt(bins[k] * bins[k + 1]) if bins[k] > 0
                  else 0.5 * (bins[k] + bins[k + 1]))
        mid.append(c); elo.append(el); ehi.append(eh)
        lo.append(b1); hi.append(b2); ns.append(int(m.sum()))
    return [np.array(a) for a in (xs, mid, elo, ehi, lo, hi, ns)]


if EQUAL_SIGNAL_BINS:
    q = np.quantile(SAMPLES['S_MC'].fLStarPt.values, np.linspace(0, 1, 8))
    PTBINS = np.round(q, 2)

R.head('BINNING')
R('  pT bins: ' + ', '.join('{:g}'.format(b) for b in PTBINS))
R('')
R('  signal candidates per bin:')
hsig, _ = np.histogram(SAMPLES['S_MC'].fLStarPt.values, bins=PTBINS)
for i, n in enumerate(hsig):
    R('    {:5g} - {:<5g}  {:5d}   (median uncertainty ~ {:.0f}% of the IQR)'.format(
        PTBINS[i], PTBINS[i + 1], n, 100 * 1.253 / np.sqrt(max(n, 1))))
R('')
R('  Bins below NMIN are dropped: {}'.format(NMIN))
R('')
R('  Error bars are 1 sigma for both statistics, inflated by sqrt(n/n_eff) where')
R('  n_eff counts distinct V0s rather than candidates.  In the MC signal the')
R('  1396 candidates come from only {} distinct Lambda V0s and {} distinct'.format(
    SAMPLES['S_MC'].fLambdaV0ID.nunique(), SAMPLES['S_MC'].fPhotonV0ID.nunique()))
R('  photon V0s, so for fLambda*/fPhoton* variables the honest error is about')
R('  {:.2f}x larger than sqrt(n) would suggest.'.format(
    np.sqrt(len(SAMPLES['S_MC']) / SAMPLES['S_MC'].fLambdaV0ID.nunique())))
R('')
for k, v in SAMPLES.items():
    R('  {:8s} n = {:8d}'.format(k, len(v)))
R('')

for f in bl.FEATURES:
    for stat in STATS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for k in ['B_DATA', 'D_IN', 'B_MC', 'S_MC']:
            nmin = NMIN['S_MC'] if k == 'S_MC' else NMIN['other']
            xs, mid, elo, ehi, lo, hi, ns = profile(SAMPLES[k], f, stat, nmin, PTBINS)
            if len(xs) == 0:
                continue
            lab = LBL[k]
            if k == 'S_MC':
                lab += '  (n/bin: {}-{})'.format(ns.min(), ns.max())
            ax.errorbar(xs, mid, yerr=[elo, ehi], fmt='o-', color=COL[k], lw=2, ms=5,
                        capsize=3, elinewidth=1.4, label=lab)
            ax.fill_between(xs, lo, hi, color=COL[k], alpha=0.13)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(['1', '2', '5', '10', '20'])
        if is_cospa(f):
            ax.set_yscale('log')
        ax.set_xlabel(r'$p_{T}$ (GeV/$c$)', fontsize=13)
        ax.set_ylabel(('1 - ' + f) if is_cospa(f) else f, fontsize=13)
        ax.set_title('{}  --  bin {}\nbars = statistical error, band = {}'.format(
            f, stat, '25-75%' if stat == 'median' else r'$\pm1\sigma$ spread'), fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(ls='--', alpha=0.4)
        path = os.path.join(OUT, '{}_{}.png'.format(f, stat))
        fig.savefig(path, bbox_inches='tight', dpi=120)
        plt.close(fig)
    R('  [saved] Individual/{}_{{{}}}.png'.format(f, ','.join(STATS)))

R.save()
