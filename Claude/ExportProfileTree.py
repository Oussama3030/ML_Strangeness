"""
ExportProfileTree -- flatten the four samples into one ROOT file for PlotBiasProfiles.C

The samples cannot be read straight out of the AO2Ds: fLStarPt / fOPAngle are
derived, the MC needs the polarity repair (sigma0builder's signedDCAToPV), and the
photon DCA sentinel has to be dropped.  This does all of that once and writes a
flat TTree the macro can just loop over.

Output: output/BiasProfiles.root
    cand      TTree, one entry per candidate: sample, pt, the 16 variables,
              lambdaKey / photonKey (distinct-V0 ids, for the n_eff correction)
    varNames  TObjArray of TObjString, the variable order
    xlo/xhi   TVectorD, robust plot range per variable
    isCosPA   TVectorD, 1 where the stored value is 1-cosPA rather than cosPA

Sentinels (|v| >= 998) are written as NaN; the macro skips non-finite values.
"""
import numpy as np
import ROOT
from array import array

import biaslib as bl

OUTFILE = bl.OUT_DIR + '/BiasProfiles.root'
SENT = 998.0
FLOOR = 1e-7                      # float32 precision near cosPA = 1
NDATA = 250000                    # per data sample; still ~40k per pT bin

SAMPLE_ORDER = ['S_MC', 'B_MC', 'B_DATA', 'D_IN']


def is_cospa(f):
    return 'CosPA' in f


def main():
    mc = bl.load_mc()
    data = bl.load_data()
    is_lstar = np.abs(mc.fPDGCode) == bl.LSTAR_PDG
    rng = np.random.RandomState(0)

    samples = {
        'S_MC': bl.fix_dca_polarity(mc[is_lstar].copy()),
        'B_MC': bl.fix_dca_polarity(mc[(~is_lstar) & bl.in_sideband(mc)].copy()),
        'B_DATA': data[bl.in_sideband(data)].sample(n=NDATA, random_state=rng),
        'D_IN': data[~bl.in_sideband(data)].sample(n=NDATA, random_state=rng),
    }

    feats = list(bl.FEATURES)
    nvar = len(feats)

    # ---- transform once, so the macro only ever plots what it is given
    cols = {}
    for tag, df in samples.items():
        out = {}
        for f in feats:
            v = df[f].values.astype(np.float64).copy()
            v[~np.isfinite(v) | (np.abs(v) >= SENT)] = np.nan
            if is_cospa(f):
                v = np.maximum(1.0 - v, FLOOR)
            out[f] = v
        out['pt'] = df.fLStarPt.values.astype(np.float64)
        # distinct-V0 ids: V0 ids restart in every dataframe, so include the tag
        for key, col in [('lambdaKey', 'fLambdaV0ID'), ('photonKey', 'fPhotonV0ID')]:
            tagcol = df['df_tag'].astype(str) if 'df_tag' in df.columns else '?'
            out[key] = np.unique(tagcol + '_' + df[col].astype(str), return_inverse=True)[1]
        cols[tag] = out

    # ---- robust plot range per variable, across all samples
    xlo = ROOT.TVectorD(nvar)
    xhi = ROOT.TVectorD(nvar)
    iscp = ROOT.TVectorD(nvar)
    for i, f in enumerate(feats):
        allv = np.concatenate([cols[t][f] for t in SAMPLE_ORDER])
        allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if is_cospa(f):
            lo = max(lo, FLOOR)
        else:
            pad = 0.03 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        xlo[i], xhi[i], iscp[i] = float(lo), float(hi), 1.0 if is_cospa(f) else 0.0

    # ---- write
    fout = ROOT.TFile(OUTFILE, 'RECREATE')
    tree = ROOT.TTree('cand', 'candidates for the bias profiles')

    b_sample = array('i', [0])
    b_pt = array('d', [0.0])
    b_lkey = array('i', [0])
    b_pkey = array('i', [0])
    b_var = {f: array('d', [0.0]) for f in feats}

    tree.Branch('sample', b_sample, 'sample/I')
    tree.Branch('pt', b_pt, 'pt/D')
    tree.Branch('lambdaKey', b_lkey, 'lambdaKey/I')
    tree.Branch('photonKey', b_pkey, 'photonKey/I')
    for f in feats:
        tree.Branch(f, b_var[f], f + '/D')

    ntot = 0
    for isample, tag in enumerate(SAMPLE_ORDER):
        c = cols[tag]
        n = len(c['pt'])
        print('[INFO] writing {:8s} {:8d} candidates'.format(tag, n))
        for j in range(n):
            b_sample[0] = isample
            b_pt[0] = c['pt'][j]
            b_lkey[0] = int(c['lambdaKey'][j])
            b_pkey[0] = int(c['photonKey'][j])
            for f in feats:
                b_var[f][0] = c[f][j]
            tree.Fill()
        ntot += n

    names = ROOT.TObjArray()
    names.SetOwner(True)
    for f in feats:
        names.Add(ROOT.TObjString(f))

    tree.Write()
    names.Write('varNames', ROOT.TObject.kSingleKey)
    xlo.Write('xlo')
    xhi.Write('xhi')
    iscp.Write('isCosPA')
    fout.Close()

    print('[INFO] wrote {} entries to {}'.format(ntot, OUTFILE))
    for tag in SAMPLE_ORDER:
        print('       sample {} = {}'.format(SAMPLE_ORDER.index(tag), tag))


if __name__ == '__main__':
    main()
