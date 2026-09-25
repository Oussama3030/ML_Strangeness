"""
biaslib -- shared loading / plumbing for the Lambda(1520) BDT bias investigation.

Everything here is READ-ONLY with respect to the analysis: it re-derives the same
quantities as Codes/01_ProcessData.py so that the diagnostics run on exactly the
sample the BDT was trained on, but it never writes into Dataset/ or Results/.

Sample definitions used throughout
----------------------------------
  S_MC    MC candidates with |fPDGCode| == 3124            (the "signal" class)
  B_MC    MC candidates that are NOT a true L*, in the same
          mass sidebands as the data background             (honest MC background)
  B_DATA  data candidates in the mass sidebands             (the "background" class)
  D_IN    data candidates INSIDE the signal mass window     (control; never trained on)

The pipeline trains S_MC vs B_DATA.  Those two differ in *two* ways at once:
they are different physics AND different origin (simulation vs detector).  The
whole point of this suite is to separate those two effects.
"""

import os
import numpy as np
import pandas as pd
import uproot

# ----------------------------------------------------------------------------- paths
MAIN_PATH = '/home/oussama/O2WorkingDirectory/RunAnalysisLambdaStar/ML_Strangeness/'
CLAUDE_DIR = os.path.join(MAIN_PATH, 'Claude')
OUT_DIR = os.path.join(CLAUDE_DIR, 'output')
CACHE_DIR = os.path.join(CLAUDE_DIR, 'cache')

SIGNAL_FILE = MAIN_PATH + 'Dataset/Raw/AO2DMerged_Sig.root'
BKG_FILE = MAIN_PATH + 'Dataset/Raw/AO2DMerged_Bkg.root'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------- analysis config
# mirrored from Codes/01_ProcessData.py and Codes/02_HyperparameterTuning.py
LSTAR_PDG = 3124
MASS_WINDOW_MIN = 1.45
MASS_WINDOW_MAX = 1.59
CLASS_NAME = 'fIsLStar'

FEATURES = ['fLambdaCosPA', 'fDCADaughters', 'fLambdaAlpha', 'fLambdaDCANegPV',
            'fLambdaDCAPosPV', 'fLambdaLifeTime', 'fLambdaQt', 'fLambdaRadius',
            'fPhotonAlpha', 'fPhotonCosPA', 'fPhotonDCADau', 'fPhotonDCANegPV',
            'fPhotonDCAPosPV', 'fPhotonQt', 'fPhotonRadius', 'fOPAngle']

# variables NOT given to the BDT, used as "is this simulation or data?" proxies
SPECTATORS = ['fLambdaPosTPCCrossedRows', 'fLambdaNegTPCCrossedRows',
              'fLambdaPosITSCls', 'fLambdaNegITSCls',
              'fLambdaPosITSChi2PerNcl', 'fLambdaNegITSChi2PerNcl',
              'fPhotonPosTPCCrossedRows', 'fPhotonNegTPCCrossedRows',
              'fPhotonPosITSCls', 'fPhotonNegITSCls',
              'fLambdaPosPrTPCNSigma', 'fLambdaNegPiTPCNSigma',
              'fPhotonPosTPCNSigmaEl', 'fPhotonNegTPCNSigmaEl',
              'fPhotonPsiPair', 'fLambdaPosEta', 'fLambdaNegEta']

CORE_TABLES = ['O2sigma0cores', 'O2sigma0lambda', 'O2sigma0photon']
MC_TABLE = 'O2sigma0mccores'

MASS_LAMBDA = 1.115683


# ------------------------------------------------------------------ derived variables
def add_derived(df):
    """Reproduce exactly the derived columns of Codes/01_ProcessData.py."""
    dot = (df['fPhotonPx'] * df['fLambdaPx'] + df['fPhotonPy'] * df['fLambdaPy']
           + df['fPhotonPz'] * df['fLambdaPz'])
    magPhoton = np.sqrt(df['fPhotonPx']**2 + df['fPhotonPy']**2 + df['fPhotonPz']**2)
    magLambda = np.sqrt(df['fLambdaPx']**2 + df['fLambdaPy']**2 + df['fLambdaPz']**2)
    ptot = magPhoton * magLambda
    arg = np.divide(dot, ptot, out=np.zeros_like(dot, dtype=float), where=ptot > 0)
    df['fOPAngle'] = np.arccos(np.clip(arg, -1.0, 1.0))

    ePhoton = magPhoton                      # massless photon
    eLambda = np.sqrt(magLambda**2 + MASS_LAMBDA**2)
    px = df['fPhotonPx'] + df['fLambdaPx']
    py = df['fPhotonPy'] + df['fLambdaPy']
    pz = df['fPhotonPz'] + df['fLambdaPz']
    m2 = (ePhoton + eLambda)**2 - (px**2 + py**2 + pz**2)
    df['fLStarMass'] = np.sqrt(np.clip(m2, 0.0, None))
    df['fLStarPt'] = np.sqrt(px**2 + py**2)
    df['fLStarP'] = np.sqrt(px**2 + py**2 + pz**2)
    df['fPhotonP'] = magPhoton
    df['fLambdaP'] = magLambda
    return df


def in_sideband(df):
    return (df['fLStarMass'] < MASS_WINDOW_MIN) | (df['fLStarMass'] > MASS_WINDOW_MAX)


# ------------------------------------------------------------------------- MC loading
def load_mc(force=False):
    """Full MC file (a single DF_1).  Returns every candidate, signal and not."""
    cache = os.path.join(CACHE_DIR, 'mc_all.parquet')
    if os.path.exists(cache) and not force:
        return pd.read_parquet(cache)

    rfile = uproot.open(SIGNAL_FILE)
    dirs = sorted({k.split(';')[0].split('/')[0] for k in rfile.keys() if '/' in k})
    frames = []
    for d in dirs:
        tabs = [rfile['{}/{}'.format(d, t)].arrays(library='pd')
                for t in CORE_TABLES + [MC_TABLE]]
        tabs = [t for t in tabs if len(t) > 0]
        n = {len(t) for t in tabs}
        if len(n) > 1:
            raise ValueError('MC tables in {} have lengths {}'.format(d, sorted(n)))
        frames.append(pd.concat(tabs, axis=1))
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = add_derived(df)
    df['origin'] = 'MC'
    df['df_tag'] = 'MC'
    df.to_parquet(cache)
    return df


# ----------------------------------------------------------------------- data loading
def load_data(n_df=12, force=False):
    """First `n_df` dataframes of the data file.

    Codes/01_ProcessData.py reads the data with NRowsBkgRead = 2e6, which its
    per-directory break turns into "the first ~5 DFs of 280", so a handful of
    DFs is already more than the pipeline itself uses.
    """
    cache = os.path.join(CACHE_DIR, 'data_{}df_v2.parquet'.format(n_df))
    if os.path.exists(cache) and not force:
        return pd.read_parquet(cache)

    rfile = uproot.open(BKG_FILE)
    dirs = [k.split(';')[0] for k in rfile.keys(recursive=False) if k.startswith('DF_')]
    frames = []
    for d in dirs[:n_df]:
        tabs = [rfile['{}/{}'.format(d, t)].arrays(library='pd') for t in CORE_TABLES]
        n = {len(t) for t in tabs}
        if len(n) > 1:
            raise ValueError('data tables in {} have lengths {}'.format(d, sorted(n)))
        one = pd.concat(tabs, axis=1)
        one['df_tag'] = d
        frames.append(one)
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = add_derived(df)
    df['origin'] = 'DATA'
    df.to_parquet(cache)
    return df


# ------------------------------------------------------------------ the four samples
def build_samples(n_df=12, seed=42, n_max=40000):
    """Return dict with S_MC, B_MC, B_DATA, D_IN (each already sideband-selected
    where appropriate, and capped at n_max rows for speed; n_max=None = no cap)."""
    mc = load_mc()
    data = load_data(n_df=n_df)

    rng = np.random.RandomState(seed)

    def cap(df):
        if n_max and len(df) > n_max:
            return df.sample(n=n_max, random_state=rng)
        return df

    is_lstar = np.abs(mc['fPDGCode']) == LSTAR_PDG
    s_mc = mc[is_lstar].copy()
    # honest MC background: not a true L*, and passing the same sideband cut that
    # defines the data background, so the two backgrounds are directly comparable
    b_mc = cap(mc[(~is_lstar) & in_sideband(mc)].copy())
    b_data = cap(data[in_sideband(data)].copy())
    d_in = cap(data[~in_sideband(data)].copy())

    return {'S_MC': s_mc, 'B_MC': b_mc, 'B_DATA': b_data, 'D_IN': d_in}


# ----------------------------------------------------------------------- BDT plumbing
def run3_params():
    """Hyperparameters of Results/Run 3 -- the run in the plot under discussion."""
    return dict(max_depth=2, learning_rate=0.13497998411880038, n_estimators=214,
                min_child_weight=8, gamma=7.539764256698049e-07,
                subsample=0.25182301517615435, colsample_bytree=0.016078701672376943,
                reg_alpha=0.19843046524130653, reg_lambda=0.0007602612136568495,
                random_state=42, eval_metric='logloss')


def fit_and_score(pos, neg, features=None, params=None, seed=42, test_size=0.30,
                  w_pos=None, w_neg=None, shuffle_labels=False, group_split=False):
    """Train one BDT on pos-vs-neg and return (auc, model, test frame).

    Split is stratified, so the AUC below is an honest held-out number for
    whatever question `pos` vs `neg` happens to pose.
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    features = FEATURES if features is None else features
    params = run3_params() if params is None else params

    X = pd.concat([pos[features], neg[features]], axis=0, ignore_index=True)
    y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    w = np.r_[np.ones(len(pos)) if w_pos is None else np.asarray(w_pos),
              np.ones(len(neg)) if w_neg is None else np.asarray(w_neg)]

    if shuffle_labels:
        y = np.random.RandomState(seed).permutation(y)

    idx = np.arange(len(y))
    if group_split:
        # keep every candidate built from the same Lambda V0 on one side of the split
        g = np.r_[group_key(pos), group_key(neg)]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        itr, ite = next(gss.split(idx, y, groups=g))
    else:
        itr, ite = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    model = XGBClassifier(**params)
    model.fit(X.iloc[itr].values, y[itr], sample_weight=w[itr])
    p = model.predict_proba(X.iloc[ite].values)[:, 1]
    auc = roc_auc_score(y[ite], p, sample_weight=w[ite])

    out = pd.DataFrame({'y': y[ite], 'p': p, 'w': w[ite]})
    return auc, model, out


def reweight_to(source_vals, target_vals, bins):
    """Per-candidate weights that morph `source_vals` into the shape of `target_vals`."""
    hs, _ = np.histogram(source_vals, bins=bins)
    ht, _ = np.histogram(target_vals, bins=bins)
    hs = hs.astype(float); ht = ht.astype(float)
    hs[hs == 0] = np.inf
    ratio = (ht / ht.sum()) / (hs / np.nansum(hs[np.isfinite(hs)]))
    ratio[~np.isfinite(ratio)] = 0.0
    k = np.clip(np.digitize(source_vals, bins) - 1, 0, len(bins) - 2)
    w = ratio[k]
    return w / w.mean() if w.mean() > 0 else w


# ------------------------------------------------------- the MC/data daughter swap
# The DCA-to-primary-vertex of the positive and negative daughter is interchanged
# (equivalently, sign-inverted) between the MC and the data production, for both
# the Lambda and the photon V0.  bias_02 measures it; swapping the four columns in
# the MC brings every one of them back on top of the data.
SWAP_PAIRS = [('fPhotonDCAPosPV', 'fPhotonDCANegPV'),
              ('fLambdaDCAPosPV', 'fLambdaDCANegPV')]


def fix_daughter_swap(df):
    """Return a copy of an MC frame with the pos/neg DCA columns put back."""
    out = df.copy()
    for a, b in SWAP_PAIRS:
        if a in out.columns and b in out.columns:
            out[a], out[b] = df[b].values, df[a].values
    return out


def fix_dca_polarity(df):
    """Exactly what sigma0builder's signedDCAToPV() now does: negate the signed
    daughter DCA-to-PV, leaving the +999 "not available" marker untouched."""
    out = df.copy()
    for c in ['fPhotonDCAPosPV', 'fPhotonDCANegPV',
              'fLambdaDCAPosPV', 'fLambdaDCANegPV']:
        if c in out.columns:
            v = out[c].values.astype(float).copy()
            keep = np.abs(v) <= 998.0
            v[keep] = -v[keep]
            out[c] = v
    return out


def group_key(df):
    """One label per physical Lambda V0.  V0 ids restart in every dataframe, so
    the dataframe tag has to be part of the key for data."""
    tag = df['df_tag'] if 'df_tag' in df.columns else pd.Series('?', index=df.index)
    return (tag.astype(str) + '_' + df['fLambdaV0ID'].astype(str)).values


# --------------------------------------------------------------------------- reporting
class Report(object):
    """Collects text so each macro leaves a readable .txt next to its plots."""

    def __init__(self, name):
        self.name = name
        self.lines = []

    def __call__(self, *args):
        line = ' '.join(str(a) for a in args)
        print(line)
        self.lines.append(line)

    def head(self, title):
        self('')
        self('=' * 78)
        self(title)
        self('=' * 78)

    def save(self):
        path = os.path.join(OUT_DIR, self.name + '.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines) + '\n')
        print('\n[saved] ' + path)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    print('[saved] ' + path)
