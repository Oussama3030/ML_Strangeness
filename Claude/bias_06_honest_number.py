"""
bias_06 -- take the biases away one at a time and see what is left.

Each rung of the ladder removes exactly one of the problems the previous macros
identified, keeping everything else fixed (same hyperparameters, same feature
list unless stated, same 70/30 fraction).  The last rung is the honest estimate
of what this classifier can do.

  0  as trained                       S_MC vs B_DATA, random split, 16 features
  1  + group-aware split              no Lambda V0 on both sides of the split
  2  + daughter swap repaired         MC pos/neg DCA columns put back
  3  + pT-matched background          background reweighted to the signal spectrum
  4  + background from MC             removes the remaining simulation/data shift
  5  + background not mass-selected   removes the sideband definition leak

Rung 5 is the number to quote.  Rungs 0-3 all compare simulation against data
and therefore all contain some amount of "which production is this?".
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import biaslib as bl

R = bl.Report('bias_06_honest_number')

mc = bl.load_mc()
data = bl.load_data()
is_lstar = np.abs(mc.fPDGCode) == bl.LSTAR_PDG

S_MC = mc[is_lstar].copy()
B_MC_side = mc[(~is_lstar) & bl.in_sideband(mc)].copy()
B_MC_all = mc[~is_lstar].copy()                       # no mass selection at all
B_DATA = data[bl.in_sideband(data)].sample(n=20000, random_state=1).copy()

# repaired copies
S_MC_fix = bl.fix_daughter_swap(S_MC)
B_MC_side_fix = bl.fix_daughter_swap(B_MC_side)
B_MC_all_fix = bl.fix_daughter_swap(B_MC_all)

ptbins = np.concatenate([np.linspace(0, 12, 49), [14, 17, 20, 25, 30]])
NB = 20000


def sub(d, n=NB, seed=1):
    return d.sample(n=min(len(d), n), random_state=seed)


R.head('THE LADDER')
R('  {:52s} {:>8s}'.format('configuration', 'AUC'))
rungs = []

a, _, _ = bl.fit_and_score(S_MC, sub(B_DATA))
rungs.append(('0  as trained (S_MC vs data sideband)', a))

a, _, _ = bl.fit_and_score(S_MC, sub(B_DATA), group_split=True)
rungs.append(('1  + group-aware train/test split', a))

a, _, _ = bl.fit_and_score(S_MC_fix, sub(B_DATA), group_split=True)
rungs.append(('2  + pos/neg daughter swap repaired', a))

w = bl.reweight_to(B_DATA['fLStarPt'].values, S_MC['fLStarPt'].values, ptbins)
bd = B_DATA.copy(); bd['_w'] = w
bd = sub(bd)
a, _, _ = bl.fit_and_score(S_MC_fix, bd, group_split=True, w_neg=bd['_w'].values)
rungs.append(('3  + background pT-matched to the signal', a))

bm = sub(B_MC_side_fix)
w = bl.reweight_to(bm['fLStarPt'].values, S_MC['fLStarPt'].values, ptbins)
a, _, _ = bl.fit_and_score(S_MC_fix, bm, group_split=True, w_neg=w)
rungs.append(('4  + background from MC (no data/MC shift)', a))

bm2 = sub(B_MC_all_fix)
w = bl.reweight_to(bm2['fLStarPt'].values, S_MC['fLStarPt'].values, ptbins)
a, _, _ = bl.fit_and_score(S_MC_fix, bm2, group_split=True, w_neg=w)
rungs.append(('5  + background NOT selected on mass  <-- honest', a))

for lab, a in rungs:
    R('  {:52s} {:8.4f}'.format(lab, a))

R('')
R('  drop from rung 0 to rung 5 : {:+.4f}'.format(rungs[-1][1] - rungs[0][1]))
R('')
R('  The ladder is not monotonic, and that is real: rung 4 swaps the data')
R('  background for an MC one, and combinatorial background inside a')
R('  signal-enhanced MC production is not the same mixture as combinatorial')
R('  background in minimum-bias data.  Rung 5 is honest about ORIGIN, but its')
R('  background still comes from MC, so it is an upper bound on what the')
R('  classifier will do on real data, not a guarantee.')

# ------------------------------------------------------ does the repair really work?
R.head('CHECK: DOES THE REPAIR REMOVE THE SIMULATION/DATA HANDLE?')
R('  The placebo training (MC background vs data background) has no physics')
R('  difference in it, so its AUC is a direct readout of how much bookkeeping')
R('  the classifier can still exploit.')
R('')
for tag, mcb in [('before repair', B_MC_side), ('after repair ', B_MC_side_fix)]:
    a, _, _ = bl.fit_and_score(sub(mcb), sub(B_DATA), group_split=True)
    R('  placebo B_MC vs B_DATA, {} : AUC = {:.4f}'.format(tag, a))
R('')
R('  Per-variable, MC vs data background (1.000 = perfectly separable):')
from sklearn.metrics import roc_auc_score
R('  {:22s} {:>10s} {:>10s}'.format('variable', 'before', 'after'))
for f in ['fPhotonDCAPosPV', 'fPhotonDCANegPV', 'fLambdaDCAPosPV', 'fLambdaDCANegPV']:
    out = []
    for mcb in [B_MC_side, B_MC_side_fix]:
        x = np.r_[mcb[f].values, B_DATA[f].values].astype(float)
        y = np.r_[np.ones(len(mcb)), np.zeros(len(B_DATA))]
        ok = np.isfinite(x)
        v = roc_auc_score(y[ok], x[ok])
        out.append(max(v, 1 - v))
    R('  {:22s} {:10.4f} {:10.4f}'.format(f, out[0], out[1]))

# ---------------------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(10, 5.5))
labs = [l for l, _ in rungs]
vals = [v for _, v in rungs]
cols = ['crimson'] + ['indianred'] * 3 + ['steelblue', 'seagreen']
ax.bar(range(len(vals)), vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, '{:.3f}'.format(v), ha='center', fontsize=10)
ax.set_xticks(range(len(labs)))
ax.set_xticklabels([l.split('  ', 1)[0] for l in labs], fontsize=11)
ax.set_ylim(0.5, 1.03)
ax.axhline(0.5, color='k', ls=':', lw=1)
ax.set_ylabel('held-out AUC', fontsize=12)
ax.set_title('Removing one bias at a time', fontsize=13)
for i, l in enumerate(labs):
    ax.text(i, 0.52, l.split('  ', 1)[1], rotation=90, ha='center', va='bottom',
            fontsize=8, color='white' if i < 4 else 'white')
ax.grid(axis='y', ls='--', alpha=0.4)
bl.savefig(fig, 'bias_06_ladder.png')

R.save()
