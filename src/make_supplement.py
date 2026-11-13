"""Build the Supplementary Material (LaTeX), the Fig. 2 example panels and the clipping count
from the outputs of BSPC_revision_rerun.ipynb. Only reads the results folder, the image zips,
the metadata table and (optionally) render_log.csv; nothing is re-computed except table layout."""
import os, re, zipfile, io, json
import numpy as np
import pandas as pd
from PIL import Image


def run(OUT_DIR, IMAGE_ZIPS, METADATA_CSV, RENDER_LOG=None, META_ID='ECG_recording_number',
        META_PART='patient_id', META_DRUG='EXTRT', META_TPT='TPT', META_DIST='QT_nomogram_distance_ms'):
    R = lambda name: pd.read_csv(os.path.join(OUT_DIR, name))
    SUP = os.path.join(OUT_DIR, 'supplementary')
    FIG = os.path.join(SUP, 'figures')
    os.makedirs(FIG, exist_ok=True)
    notes = {}

    # ---------- helpers ----------
    def esc(x):
        s = str(x)
        for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'),
                     ('$', r'\$'), ('{', r'\{'), ('}', r'\}'), ('~', r'\~{}'), ('^', r'\^{}')]:
            s = s.replace(a, b)
        return s.replace('>=', r'$\geq$')

    def fmt(v, d=3):
        if isinstance(v, (int, np.integer)):
            return f'{v:d}'
        try:
            f = float(v)
        except (TypeError, ValueError):
            return esc(v)
        if np.isnan(f):
            return '--'
        if f.is_integer() and abs(f) >= 1 and d == 0:
            return f'{int(f):d}'
        s = f'{f:.{d}f}'
        return s.replace('-', '$-$') if s.startswith('-') else s

    CFG = {'image_only': 'Image only', 'img+colour': 'Image + colour (C+$\\bar{M}$)',
           'nested_selected': 'Nested selection', 'logistic': 'QT and heart rate (logistic)'}
    for L in ['No Label', 'L1', 'L2', 'L3']:
        CFG[f'img+morph[{L}]'] = f'Image + {L} ($\\bar{{C}}$+M)'
        CFG[f'img+both[{L}]'] = f'Image + {L} (C+M)'
        CFG[f'prompt:morph[{L}]'] = f'Prompt only: {L} ($\\bar{{C}}$+M)'
        CFG[f'prompt:both[{L}]'] = f'Prompt only: {L} (C+M)'
    CFG['prompt:colour'] = 'Prompt only: colour (C+$\\bar{M}$)'
    REND = {'colour': 'Pseudo-coloured', 'colourless': 'Colourless', 'colour_greyscale': 'Greyscale of coloured',
            'QT+HR': 'QT and heart rate'}
    SCH = {'participant_grouped': 'participant-grouped', 'image_stratified': 'image-level stratified',
           'timepoint_grouped': 'triplicate-grouped'}
    STRAT = {'morphology only': 'Morphology only ($\\bar{C}$+M)', 'colour + morphology': 'Colour and morphology (C+M)',
             'colour only': 'Colour only (C+$\\bar{M}$)', 'no prompt': 'No prompt ($\\bar{C}$+$\\bar{M}$)'}
    cfg = lambda c: CFG.get(c, esc(c))
    strat = lambda s: next((v for k, v in STRAT.items() if str(s).startswith(k)), esc(s))
    order_cfg = (['image_only', 'img+colour'] + [f'img+morph[{L}]' for L in ['No Label', 'L1', 'L2', 'L3']]
                 + [f'img+both[{L}]' for L in ['No Label', 'L1', 'L2', 'L3']] + ['nested_selected', 'prompt:colour']
                 + [f'prompt:morph[{L}]' for L in ['No Label', 'L1', 'L2', 'L3']]
                 + [f'prompt:both[{L}]' for L in ['No Label', 'L1', 'L2', 'L3']] + ['logistic'])
    okey = lambda c: order_cfg.index(c) if c in order_cfg else 999

    tables = []

    def table(caption, label, header, rows, colspec, size=r'\footnotesize', long=False):
        body = '\n'.join(' & '.join(r) + r' \\' if r != ['MIDRULE'] else r'\midrule' for r in rows)
        if long:
            t = (f'{size}\n\\setlength{{\\tabcolsep}}{{3.5pt}}\n\\begin{{longtable}}{{{colspec}}}\n'
                 f'\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{" & ".join(header)} \\\\\n\\midrule\n\\endfirsthead\n'
                 f'\\toprule\n{" & ".join(header)} \\\\\n\\midrule\n\\endhead\n{body}\n\\bottomrule\n\\end{{longtable}}\n\\normalsize\n')
        else:
            t = (f'\\begin{{table}}[H]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n{size}\n'
                 f'\\setlength{{\\tabcolsep}}{{3.5pt}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{" & ".join(header)} \\\\\n'
                 f'\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n')
        tables.append(t)

    # ---------- S1 data ----------
    ex = R('data_exclusions_by_rendering.csv')
    table('Recordings and images: exclusions by rendering and class.', 'tab:s-excl',
          ['Reason', 'At risk', 'Low risk', 'No label'],
          [[esc(r.reason), fmt(int(r.high_risk), 0), fmt(int(r.low_risk), 0), fmt(int(getattr(r, 'unknown_label', 0)), 0)]
           for r in ex.itertuples()], 'lrrr')
    if os.path.exists(os.path.join(OUT_DIR, 'label_agreement.csv')):
        la = pd.read_csv(os.path.join(OUT_DIR, 'label_agreement.csv'))
        la = la.fillna('missing')
        table('Agreement between the two label sources in the metadata (rows: QT--nomogram distance $\\geq 0$; '
              'columns: recorded outcome = at risk). Missing: no QT value.', 'tab:s-labels',
              [esc(c) for c in la.columns], [[esc(v) for v in r] for r in la.astype(str).values], 'l' + 'r' * (la.shape[1] - 1))
    hp = R('hierarchy_per_participant.csv').sort_values('participant')
    table('Images and at-risk images per participant.', 'tab:s-part',
          ['Participant', 'Images', 'At risk', 'Drugs', 'Time-point clusters'],
          [[esc(r.participant), fmt(int(r.images), 0), fmt(int(r.high_risk), 0), fmt(int(r.drugs), 0),
            fmt(int(r.timepoint_clusters), 0)] for r in hp.itertuples()], 'lrrrr')
    hd = R('hierarchy_by_drug.csv')
    table('Images and at-risk images per treatment.', 'tab:s-drug', ['Treatment', 'Images', 'At risk'],
          [[esc(r.drug), fmt(int(r.images), 0), fmt(int(r.high_risk), 0)] for r in hd.itertuples()], 'lrr')
    tr = R('triplicate_similarity.csv').iloc[0]
    co = R('cohort_characteristics.csv')
    table('Cohort characteristics (one row per participant) and similarity of replicate ECGs.', 'tab:s-cohort',
          ['Variable', 'Summary'],
          [[esc(r.variable), esc(r.summary)] for r in co.itertuples()] + [['MIDRULE'],
           ['Median QT range within a triplicate (ms)', fmt(tr.median_QT_range_within_cluster_ms, 1)],
           ['Median QT difference, ECGs from different time points (ms)', fmt(tr.median_abs_QT_difference_random_pairs_ms, 1)],
           ['Triplicates with mixed labels', fmt(int(tr.clusters_with_mixed_labels), 0)]], 'lp{0.55\\linewidth}')

    # ---------- S2 folds ----------
    fs = R('fold_summary.csv')
    table('Composition of the cross-validation folds.', 'tab:s-folds',
          ['Split', 'Fold', 'Images', 'At risk', 'Participants'],
          [[SCH.get(r.scheme, esc(r.scheme)), fmt(int(r.fold), 0), fmt(int(r.images), 0), fmt(int(r.high_risk), 0),
            fmt(int(getattr(r, 'participants', np.nan)), 0) if not pd.isna(getattr(r, 'participants', np.nan)) else '--']
           for r in fs.itertuples()], 'llrrr')

    # ---------- S3 zero-shot ----------
    zs = R('zero_shot_matched.csv')
    rows = []
    for rend in ['colour', 'colourless']:
        for r in zs[zs.rendering == rend].itertuples():
            rows.append([REND[rend], esc(r.label_set), strat(r.strategy), fmt(int(r.TP), 0), fmt(int(r.FN), 0),
                         fmt(int(r.TN), 0), fmt(int(r.FP), 0), fmt(r.Sens), fmt(r.Spec), fmt(r.BA), fmt(r.PPV),
                         fmt(r.MCC), fmt(r.ROC_AUC), fmt(r.AUPRC)])
        rows.append(['MIDRULE'])
    table('Zero-shot inference: confusion matrices and metrics on all paired ECGs. AUC is computed from the summed '
          'probability of the at-risk prompts. The colour-only and no-prompt strategies do not depend on the label set.',
          'tab:s-zs', ['Rendering', 'Set', 'Strategy', 'TP', 'FN', 'TN', 'FP', 'Sens.', 'Spec.', 'BA', 'PPV', 'MCC', 'AUC', 'AUPRC'],
          rows[:-1], 'lll' + 'r' * 11, size=r'\scriptsize', long=True)

    # ---------- S4/S5 trained, pooled ----------
    po = R('probe_pooled_out_of_fold.csv').rename(columns={'Spec_at_Sens0.90': 'SpecAt90', 'Sens_at_Spec0.90': 'SensAt90'})
    po['k'] = po.config.map(okey)
    rend_order = {'colour': 0, 'colourless': 1, 'colour_greyscale': 2, 'QT+HR': 3}
    for sch in ['participant_grouped', 'image_stratified', 'timepoint_grouped']:
        sub = po[(po.scheme == sch) & po.threshold.str.startswith('0.5')].copy()
        if sub.empty:
            continue
        sub['r'] = sub.rendering.map(rend_order)
        sub = sub.sort_values(['r', 'k'])
        rows, prev = [], None
        for r in sub.itertuples():
            if prev is not None and r.rendering != prev:
                rows.append(['MIDRULE'])
            prev = r.rendering
            rows.append([REND.get(r.rendering, esc(r.rendering)), cfg(r.config), fmt(int(r.TP), 0), fmt(int(r.FN), 0),
                         fmt(int(r.TN), 0), fmt(int(r.FP), 0), fmt(r.Sens), fmt(r.Spec), fmt(r.PPV), fmt(r.MCC),
                         fmt(r.ROC_AUC), fmt(r.AUPRC), fmt(r.SpecAt90)])
        table(f'Trained probes, {SCH[sch]} cross-validation: pooled out-of-fold confusion matrices at threshold 0.5. '
              'Spec@Sens0.90: specificity at a sensitivity of 0.90 from the pooled ROC curve.', f'tab:s-pooled-{sch}',
              ['Input', 'Probe features', 'TP', 'FN', 'TN', 'FP', 'Sens.', 'Spec.', 'PPV', 'MCC', 'AUC', 'AUPRC', 'Spec@Sens0.90'],
              rows, 'll' + 'r' * 11, size=r'\scriptsize', long=True)

    # ---------- S6 inner thresholds ----------
    sub = po[(po.scheme == 'participant_grouped') & ~po.threshold.str.startswith('0.5')].copy()
    if not sub.empty:
        sub['r'] = sub.rendering.map(rend_order)
        sub = sub.sort_values(['r', 'k', 'threshold'])
        rows = [[REND.get(r.rendering, esc(r.rendering)), cfg(r.config), esc(r.threshold), fmt(int(r.TP), 0), fmt(int(r.FN), 0),
                 fmt(int(r.TN), 0), fmt(int(r.FP), 0), fmt(r.Sens), fmt(r.Spec), fmt(r.PPV), fmt(r.MCC)] for r in sub.itertuples()]
        table('Thresholds chosen by inner participant-grouped cross-validation on the training data of each outer fold '
              '(participant-grouped split), pooled out-of-fold results.', 'tab:s-inner',
              ['Input', 'Probe features', 'Threshold rule', 'TP', 'FN', 'TN', 'FP', 'Sens.', 'Spec.', 'PPV', 'MCC'],
              rows, 'lll' + 'r' * 8, size=r'\scriptsize', long=True)

    # ---------- S7 per fold (participant-grouped) ----------
    pf = R('probe_per_fold.csv')
    sub = pf[(pf.scheme == 'participant_grouped') & pf.config.isin(['image_only', 'logistic', 'nested_selected'])].copy()
    sub['r'] = sub.rendering.map(rend_order)
    sub = sub.sort_values(['r', 'config', 'fold'])
    rows = [[REND.get(r.rendering, esc(r.rendering)), cfg(r.config), fmt(int(r.fold), 0), fmt(int(r.TP), 0), fmt(int(r.FN), 0),
             fmt(int(r.TN), 0), fmt(int(r.FP), 0), fmt(r.Sens), fmt(r.Spec), fmt(r.PPV), fmt(r.MCC), fmt(r.ROC_AUC)]
            for r in sub.itertuples()]
    table('Per-fold results of the image-only probes, nested selection and the QT--heart-rate model '
          '(participant-grouped split, threshold 0.5). Per-fold results for every configuration are in the code repository.',
          'tab:s-perfold', ['Input', 'Probe features', 'Fold', 'TP', 'FN', 'TN', 'FP', 'Sens.', 'Spec.', 'PPV', 'MCC', 'AUC'],
          rows, 'll' + 'r' * 10, size=r'\scriptsize', long=True)
    fm = R('probe_fold_mean_sd.csv')
    sub = fm[fm.scheme == 'participant_grouped'].copy()
    sub['r'] = sub.rendering.map(rend_order); sub['k'] = sub.config.map(okey)
    sub = sub.sort_values(['r', 'k'])
    pm = lambda m, s: f'{fmt(m)} ({fmt(s)})'
    rows = [[REND.get(r.rendering, esc(r.rendering)), cfg(r.config), pm(r.MCC_mean, r.MCC_std), pm(r.BA_mean, r.BA_std),
             pm(r.ROC_AUC_mean, r.ROC_AUC_std), pm(r.AUPRC_mean, r.AUPRC_std)] for r in sub.itertuples()]
    table('Mean (SD) over the five participant-grouped folds.', 'tab:s-foldmean',
          ['Input', 'Probe features', 'MCC', 'BA', 'AUC', 'AUPRC'], rows, 'llrrrr', size=r'\scriptsize', long=True)

    # ---------- S8 nested selection ----------
    ns = R('nested_selection.csv')
    table('Configuration chosen in each outer fold by nested selection (inner cross-validation MCC on training data).',
          'tab:s-nested', ['Rendering', 'Fold', 'Selected configuration', 'Inner MCC'],
          [[REND.get(r.rendering, esc(r.rendering)), fmt(int(r.fold), 0), cfg(r.selected), fmt(r.inner_MCC)] for r in ns.itertuples()],
          'lrlr')

    # ---------- S9 bootstrap ----------
    bt = R('bootstrap_paired_differences.csv')
    rows = []
    for (fam, metric), g in bt.groupby(['family', 'metric'], sort=False):
        for r in g.itertuples():
            rows.append([esc(fam), esc(r.comparison).replace('img+', 'image + ').replace('image\\_only', 'image only'), esc(metric),
                         fmt(r.diff_A_minus_B), f'[{fmt(r.CI95_low)}, {fmt(r.CI95_high)}]', fmt(r.p_boot), fmt(r.p_holm)])
    table('All paired comparisons (participant-grouped split, image probes at threshold 0.5): difference A $-$ B, 95\\% '
          'participant-cluster bootstrap interval, two-sided bootstrap $p$ and Holm-adjusted $p$ within each family and metric.',
          'tab:s-boot', ['Family', 'Comparison (A vs B)', 'Metric', 'Diff.', '95\\% CI', '$p$', '$p_{\\mathrm{Holm}}$'],
          rows, 'p{0.2\\linewidth}p{0.3\\linewidth}lrlrr', size=r'\scriptsize', long=True)

    # ---------- S10 third condition ----------
    if os.path.exists(os.path.join(OUT_DIR, 'third_condition_metrics.csv')):
        tc = R('third_condition_metrics.csv')
        table('Stretched (original) versus fixed-axis renderings on the ECGs available in all four renderings '
              '(image-only probe, participant-grouped split, threshold 0.5).', 'tab:s-geom',
              ['Rendering', 'n', 'At risk', 'TP', 'FN', 'TN', 'FP', 'Sens.', 'Spec.', 'PPV', 'MCC', 'AUC', 'AUPRC'],
              [[esc(r.condition), fmt(int(r.n), 0), fmt(int(r.high_risk), 0), fmt(int(r.TP), 0), fmt(int(r.FN), 0),
                fmt(int(r.TN), 0), fmt(int(r.FP), 0), fmt(r.Sens), fmt(r.Spec), fmt(r.PPV), fmt(r.MCC), fmt(r.ROC_AUC),
                fmt(r.AUPRC)] for r in tc.itertuples()], 'l' + 'r' * 12, size=r'\scriptsize')
        tb = R('third_condition_bootstrap.csv')
        table('Paired differences between renderings on the same ECGs (participant-cluster bootstrap).', 'tab:s-geomboot',
              ['A', 'B', 'Metric', 'Diff.', '95\\% CI', '$p$'],
              [[esc(r.A), esc(r.B), esc(r.metric), fmt(r.diff), f'[{fmt(r.CI95_low)}, {fmt(r.CI95_high)}]', fmt(r.p_boot)]
               for r in tb.itertuples()], 'p{0.24\\linewidth}p{0.24\\linewidth}lrlr', size=r'\scriptsize')

    # ---------- S11 reproduction ----------
    if os.path.exists(os.path.join(OUT_DIR, 'reproduction_check_old_images.csv')):
        rp = R('reproduction_check_old_images.csv')
        rp['check'] = rp.check.astype(str).str.replace('C+M̄', 'colour only', regex=False).str.replace('C̄+M', 'morphology only', regex=False)
        table('Reproduction check: the original protocol (image-level stratified split, unscaled features, original '
              'probe settings) re-run on the originally submitted images.', 'tab:s-repro', ['Check', 'Published', 'Re-run'],
              [[esc(r.check), esc(r.published), esc(r.rerun)] for r in rp.itertuples()], 'p{0.55\\linewidth}ll')

    # ---------- render log ----------
    if RENDER_LOG and os.path.exists(RENDER_LOG):
        rl = pd.read_csv(RENDER_LOG)
        rr = pd.to_numeric(rl['RR_beat1_s'] if 'RR_beat1_s' in rl else rl['RR_s'], errors='coerce')
        rr_ms = np.where(rr.median() < 10, rr * 1000, rr)
        notes['render_log_rows'] = int(len(rl))
        notes['RR_over_1520ms'] = int((rr_ms > 1520).sum())
        notes['not_rendered'] = int((~rl['rendered'].astype(bool)).sum()) if 'rendered' in rl else None
        notes['RR_max_ms'] = float(np.nanmax(rr_ms))

    # ---------- Fig. 2 examples ----------
    meta = pd.read_csv(METADATA_CSV).drop_duplicates(META_ID)
    bm = R('beat_metadata.csv')
    dist = meta.set_index(META_ID)[META_DIST]
    bm['dist'] = bm.rec.map(dist)
    pos = bm[bm.y == 1].dropna(subset=['dist'])
    target = pos.dist.quantile(0.75)
    hi = pos.iloc[(pos.dist - target).abs().argsort().iloc[0]]
    same = bm[(bm.y == 0) & (bm.participant.astype(str) == str(hi.participant)) & (bm.timepoint.astype(str) == str(hi.timepoint))
              & (bm.drug.astype(str).str.lower().str.contains('placebo'))]
    if same.empty:  # placebo, same participant, any time point
        same = bm[(bm.y == 0) & (bm.participant.astype(str) == str(hi.participant))
                  & (bm.drug.astype(str).str.lower().str.contains('placebo'))]
    if same.empty:
        same = bm[(bm.y == 0) & (bm.participant.astype(str) == str(hi.participant))]
    lo = same.iloc[0]
    picks = {'highrisk': hi, 'norisk': lo}
    for rend, zpath in IMAGE_ZIPS.items():
        with zipfile.ZipFile(zpath) as z:
            names = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith('.png') and '__MACOSX' not in n}
            for tag, row in picks.items():
                im = Image.open(io.BytesIO(z.read(names[str(row.file_name)]))).convert('RGB')
                im.save(os.path.join(FIG, f"ecg_{'colour' if rend == 'colour' else 'grey'}_{tag}.png"))
    notes['fig2_highrisk'] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in
                              hi[['rec', 'participant', 'drug', 'timepoint', 'QT', 'HR', 'dist']].to_dict().items()}
    notes['fig2_lowrisk'] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in
                             lo[['rec', 'participant', 'drug', 'timepoint', 'QT', 'HR', 'dist']].to_dict().items()}

    # ---------- write document ----------
    doc = (r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=2cm,landscape]{geometry}
\usepackage{booktabs,longtable,array,float}
\usepackage[T1]{fontenc}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary Material\\[4pt]\large Vision--language detection of drug-induced QT prolongation at risk of Torsades de Pointes from pseudo-coloured ECG images}
\author{D. Labrador Alonso and A. Alahmadi}
\date{}
\begin{document}
\maketitle
\noindent All values are computed from the analysis outputs in the public code repository. TP, FN, TN, FP: true
positives, false negatives, true negatives, false positives (positive class: at risk). Sens.\ sensitivity; Spec.\
specificity; BA balanced accuracy; PPV positive predictive value; MCC Matthews correlation coefficient; AUC area under
the ROC curve; AUPRC area under the precision--recall curve. C: colour named in the prompt; M: morphology named.

""" + '\n'.join(tables) + r"""
\end{document}
""")
    path = os.path.join(SUP, 'BSPC-D-26-17774_supplementary.tex')
    open(path, 'w', encoding='utf-8').write(doc)
    with open(os.path.join(SUP, 'supplement_notes.json'), 'w') as f:
        json.dump(notes, f, indent=2, default=str)
    print('Wrote', path, f'({len(tables)} tables)')
    print(json.dumps(notes, indent=2, default=str))
    return path, notes
