"""
bias_01 -- Is the BDT separating signal from background, or simulation from data?

The pipeline labels MC L* candidates "signal" and data sideband candidates
"background".  Those two samples differ in two ways at once, and a BDT will
happily use either one:

    (i)  physics      -- a true L* really does look different from combinatorics
    (ii) origin       -- a simulated candidate really does look different from a
                         reconstructed one (resolution, material, occupancy, ...)

Three trainings separate them.  All three use the Run 3 hyperparameters and the
Run 3 feature list, so the numbers are directly comparable:

    A)  S_MC  vs B_DATA   the pipeline's own training -- reproduces the plot
    B)  S_MC  vs B_MC     same physics question, one single origin (honest)
    C)  B_MC  vs B_DATA   NO physics difference at all: both samples are
                          combinatorial background.  Any AUC above 0.5 here is
                          pure origin, i.e. exactly the handle that makes (A)
                          look better than (B).

If C is close to 1, the headline plot is largely an MC-tagger.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import biaslib as bl

R = bl.Report('bias_01_domain_shift')

R.head('BUILDING SAMPLES')
S = bl.build_samples()
for k, v in S.items():
    R('  {:7s} {:8d} candidates'.format(k, len(v)))
R('')
R('  S_MC   : MC, |fPDGCode| == 3124')
R('  B_MC   : MC, not a true L*, mass in the same sidebands as B_DATA')
R('  B_DATA : data, mass in sidebands (< {} or > {})'.format(bl.MASS_WINDOW_MIN,
                                                             bl.MASS_WINDOW_MAX))
R('  D_IN   : data, mass INSIDE the window (control, never trained on)')

# ------------------------------------------------------------------ the three trainings
tests = [
    ('A  S_MC vs B_DATA  (the pipeline)', 'S_MC', 'B_DATA'),
    ('B  S_MC vs B_MC    (honest, MC only)', 'S_MC', 'B_MC'),
    ('C  B_MC vs B_DATA  (placebo: no physics difference)', 'B_MC', 'B_DATA'),
]

R.head('TRAININGS  (Run 3 hyperparameters, Run 3 features, 70/30 split)')
results = {}
for label, pk, nk in tests:
    pos, neg = S[pk], S[nk]
    # equalise statistics so the three AUCs are not driven by sample size
    n = min(len(pos), len(neg), 20000)
    auc, model, out = bl.fit_and_score(pos.sample(n=n, random_state=1),
                                       neg.sample(n=n, random_state=1))
    results[label[0]] = (label, auc, out, model)
    R('  {:52s} AUC = {:.4f}   (n = {} + {})'.format(label, auc, n, n))

aucA = results['A'][1]
aucB = results['B'][1]
aucC = results['C'][1]

R('')
R('  A - B = {:+.4f}   how much of the pipeline AUC is NOT reproducible'.format(aucA - aucB))
R('             within a single origin')
R('  C     = {:.4f}   AUC obtainable with zero physics difference'.format(aucC))
R('             (0.5 = MC and data indistinguishable, 1.0 = trivially separable)')

# ------------------------------------------------------------------- spectator version
R.head('PLACEBO ON SPECTATOR VARIABLES (never given to the BDT)')
R('  Same B_MC vs B_DATA question, but using only tracking-quality variables')
R('  (TPC crossed rows, ITS clusters/chi2, nsigma, eta).  These carry no L*')
R('  physics whatsoever, so an AUC far from 0.5 means MC and data are simply')
R('  different detectors as far as any classifier is concerned.')
spec = [c for c in bl.SPECTATORS if c in S['B_MC'].columns and c in S['B_DATA'].columns]
n = min(len(S['B_MC']), len(S['B_DATA']), 20000)
auc_spec, _, _ = bl.fit_and_score(S['B_MC'].sample(n=n, random_state=1),
                                  S['B_DATA'].sample(n=n, random_state=1),
                                  features=spec)
R('')
R('  spectator-only  B_MC vs B_DATA   AUC = {:.4f}   ({} variables)'.format(auc_spec, len(spec)))

# ------------------------------------------------------------------------- label shuffle
R.head('NEGATIVE CONTROL: labels shuffled')
auc_shuf, _, _ = bl.fit_and_score(S['S_MC'].sample(n=min(len(S['S_MC']), 20000), random_state=1),
                                  S['B_DATA'].sample(n=min(len(S['S_MC']), 20000), random_state=1),
                                  shuffle_labels=True)
R('  S_MC vs B_DATA with randomised labels   AUC = {:.4f}'.format(auc_shuf))
R('  (must sit at ~0.5; if not, the split itself leaks)')

# -------------------------------------------------------------------------------- plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, key in zip(axes, ['A', 'B', 'C']):
    label, auc, out, _ = results[key]
    bins = np.linspace(0, 1, 60)
    ax.hist(out.p[out.y == 0], bins=bins, color='royalblue', alpha=0.6,
            label='negative class')
    ax.hist(out.p[out.y == 1], bins=bins, color='salmon', alpha=0.7,
            label='positive class')
    ax.set_yscale('log')
    ax.set_xlabel('BDT output (P)')
    ax.set_ylabel('Counts')
    ax.set_title('{}\nAUC = {:.4f}'.format(label, auc), fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle('Same features, same hyperparameters -- only the two samples change',
             fontsize=12)
fig.tight_layout()
bl.savefig(fig, 'bias_01_three_trainings.png')

# ROC overlay
from sklearn.metrics import roc_curve
fig2, ax = plt.subplots(figsize=(6, 5.5))
for key, col in zip(['A', 'B', 'C'], ['crimson', 'seagreen', 'darkorange']):
    label, auc, out, _ = results[key]
    fpr, tpr, _ = roc_curve(out.y, out.p)
    ax.plot(fpr, tpr, color=col, label='{}  AUC={:.3f}'.format(label.split('(')[0].strip(), auc))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='random, AUC=0.5')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('Where does the performance come from?', fontsize=13)
ax.grid(linestyle='--', alpha=0.5)
ax.legend(loc='lower right', fontsize=9)
bl.savefig(fig2, 'bias_01_roc_overlay.png')

R.head('VERDICT')
if aucC > 0.90:
    R('  C = {:.3f}: MC and data backgrounds are almost perfectly separable with'.format(aucC))
    R('  the training features alone.  The classifier does not need any L* physics')
    R('  to reproduce most of the headline separation -- "MC or not" is enough.')
elif aucC > 0.70:
    R('  C = {:.3f}: a large, non-physics handle exists between MC and data.'.format(aucC))
else:
    R('  C = {:.3f}: MC/data domain shift is modest in these features.'.format(aucC))
R('')
R('  Honest in-domain performance (B) = {:.4f}'.format(aucB))
R('  Pipeline performance        (A) = {:.4f}'.format(aucA))
R.save()
