"""
Build a single self-contained HTML report (report.html) for the Bonn seizure
study. All figures are base64-embedded so the file is fully portable.
Academic styling: serif typography, booktabs-style tables, numbered
figures/tables, aspect-ratio-aware figure layout.
"""

import os
import csv
import base64

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results_seizure')


def img(path):
    """Return a base64 data URI for a PNG, or '' if missing."""
    full = os.path.join(RES, path)
    if not os.path.exists(full):
        return ''
    with open(full, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/png;base64,{b64}'


def load_rows():
    with open(os.path.join(RES, 'results_summary.csv')) as f:
        return list(csv.DictReader(f))


def fmt(x, n=4):
    try:
        return f'{float(x):.{n}f}'
    except (TypeError, ValueError):
        return str(x)


rows = load_rows()


def summary_table():
    head = ('<tr><th>Task</th><th>Model</th><th>Acc.</th><th>&kappa;</th>'
            '<th>AUC</th><th>Sz. Sens.</th><th>Sz. Spec.</th>'
            '<th>Params</th><th>Time</th></tr>')
    body = ''
    prev = None
    for r in rows:
        sep = ' class="grouptop"' if (prev and prev != r['scheme']) else ''
        prev = r['scheme']
        body += (f'<tr{sep}><td>{r["scheme"]}</td><td>{r["model"]}</td>'
                 f'<td>{fmt(r["accuracy"])}</td><td>{fmt(r["kappa"])}</td>'
                 f'<td>{fmt(r["macro_auc"])}</td>'
                 f'<td>{fmt(r["seizure_sens"])}</td>'
                 f'<td>{fmt(r["seizure_spec"])}</td>'
                 f'<td>{int(r["params"]):,}</td><td>{fmt(r["train_min"],1)}&thinsp;m</td></tr>')
    return f'<table class="booktabs">{head}{body}</table>'


FIGNO = {'n': 0}


def model_figure(scheme, model):
    FIGNO['n'] += 1
    n = FIGNO['n']
    task = 'binary' if scheme == 'binary' else '3-class'
    return f'''
    <figure class="modelfig">
      <img class="wide" src="{img(f"{scheme}/{model}_learning.png")}" alt="learning curves">
      <img class="wide" src="{img(f"{scheme}/{model}_confusion.png")}" alt="confusion matrix">
      <img class="square" src="{img(f"{scheme}/{model}_roc.png")}" alt="roc curves">
      <figcaption><span class="fnum">Figure {n}.</span> <strong>{model}</strong> on the
      {task} task. <em>(a)</em> training and validation accuracy/loss across epochs;
      <em>(b)</em> confusion matrix in raw counts and row-normalized form;
      <em>(c)</em> one-vs-rest ROC curves with per-class AUC.</figcaption>
    </figure>'''


figures_html = ''
for scheme in ['binary', '3class']:
    pretty = 'Binary task (seizure vs. non-seizure)' if scheme == 'binary' \
        else 'Three-class task (healthy / interictal / ictal)'
    figures_html += f'<h3>5.{1 if scheme=="binary" else 2}&ensp;{pretty}</h3>'
    for model in ['ATCNet', 'EEGNet']:
        figures_html += model_figure(scheme, model)


HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EEG-Based Seizure Detection on the Bonn Dataset</title>
<style>
  :root {{
    --ink:#16181d; --soft:#454b57; --muted:#6b7280; --rule:#1a1a1a;
    --hair:#d8dce3; --bg:#ffffff; --page:#eef0f3; --accent:#3a4a66;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  }}
  * {{ box-sizing:border-box; }}
  html {{ -webkit-text-size-adjust:100%; }}
  body {{
    margin:0; background:var(--page); color:var(--ink);
    font-family:var(--serif); font-size:17px; line-height:1.62;
  }}
  .page {{
    max-width:780px; margin:32px auto; background:var(--bg);
    padding:64px 72px 72px; box-shadow:0 1px 6px rgba(0,0,0,.08);
  }}

  /* ---- Title block ---- */
  .titleblock {{ text-align:center; border-bottom:2.5px solid var(--rule); padding-bottom:26px; margin-bottom:8px; }}
  .titleblock .kicker {{ font-variant:small-caps; letter-spacing:.18em; font-size:.82rem;
    color:var(--muted); margin-bottom:14px; }}
  .titleblock h1 {{ font-size:1.96rem; line-height:1.22; margin:0 0 10px; font-weight:600; }}
  .titleblock .subtitle {{ font-size:1.12rem; color:var(--soft); font-style:italic; margin:0; }}
  .titleblock .meta {{ font-size:.9rem; color:var(--muted); margin-top:16px; }}

  /* ---- Headings ---- */
  h2 {{ font-size:1.28rem; font-weight:600; margin:38px 0 10px; }}
  h2 .num {{ color:var(--accent); margin-right:.5em; }}
  h3 {{ font-size:1.08rem; font-weight:600; margin:26px 0 8px; color:#222; }}
  h4 {{ font-size:1rem; font-weight:600; margin:18px 0 6px; color:var(--accent); }}

  /* ---- Body text ---- */
  p {{ margin:10px 0; text-align:justify; hyphens:auto; }}
  .abstract {{ font-size:.97rem; color:var(--soft); background:#f7f8fa;
    border:1px solid var(--hair); border-radius:4px; padding:18px 22px; margin:22px 0 8px; }}
  .abstract h4 {{ margin:0 0 6px; color:var(--ink); font-variant:small-caps; letter-spacing:.05em; }}
  ol, ul {{ padding-left:1.4em; }}
  ol li, ul li {{ margin:5px 0; text-align:justify; }}
  code {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:.82em;
    background:#f0f2f5; padding:1px 5px; border-radius:3px; }}

  /* ---- Booktabs tables ---- */
  .tabwrap {{ margin:18px 0 22px; }}
  .tabwrap .cap {{ font-size:.88rem; color:var(--soft); margin-bottom:6px; }}
  .tabwrap .cap b {{ color:var(--ink); }}
  table.booktabs {{ width:100%; border-collapse:collapse; font-size:.9rem;
    border-top:2px solid var(--rule); border-bottom:2px solid var(--rule); }}
  table.booktabs th {{ font-weight:600; padding:7px 8px; text-align:center;
    border-bottom:1px solid var(--rule); }}
  table.booktabs td {{ padding:6px 8px; text-align:center; }}
  table.booktabs td:nth-child(2) {{ font-style:italic; }}
  table.booktabs tr.grouptop td {{ border-top:1px solid var(--hair); }}
  table.data {{ width:100%; border-collapse:collapse; font-size:.88rem;
    border-top:2px solid var(--rule); border-bottom:2px solid var(--rule); }}
  table.data th {{ text-align:left; font-weight:600; padding:7px 10px; border-bottom:1px solid var(--rule); }}
  table.data td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--hair); }}
  table.data tr:last-child td {{ border-bottom:none; }}

  /* ---- Callouts (restrained, academic) ---- */
  .note {{ border-left:3px solid var(--accent); background:#f6f7f9;
    padding:12px 18px; margin:16px 0; font-size:.95rem; }}
  .note .lbl {{ font-variant:small-caps; letter-spacing:.06em; font-weight:600; color:var(--accent); }}
  .caution {{ border-left:3px solid #8a6d3b; background:#faf7f0;
    padding:12px 18px; margin:16px 0; font-size:.95rem; }}
  .caution .lbl {{ font-variant:small-caps; letter-spacing:.06em; font-weight:600; color:#8a6d3b; }}

  /* ---- Figures (aspect-ratio aware) ---- */
  figure.modelfig {{ margin:24px 0 30px; }}
  figure.modelfig img {{ display:block; border:1px solid var(--hair); background:#fff; }}
  figure.modelfig img.wide {{ width:100%; margin:0 auto 10px; }}
  figure.modelfig img.square {{ width:62%; max-width:430px; margin:0 auto 10px; }}
  figcaption {{ font-size:.86rem; color:var(--soft); line-height:1.5;
    text-align:justify; padding-top:2px; border-top:1px solid var(--hair); margin-top:4px; }}
  .fnum {{ font-weight:600; color:var(--ink); }}

  footer {{ margin-top:44px; padding-top:18px; border-top:1px solid var(--hair);
    font-size:.82rem; color:var(--muted); text-align:center; line-height:1.5; }}

  @media (max-width:840px) {{
    .page {{ padding:40px 26px; margin:0; }}
    figure.modelfig img.square {{ width:90%; }}
    body {{ font-size:16px; }}
  }}
  @media print {{
    body {{ background:#fff; }}
    .page {{ box-shadow:none; margin:0; max-width:none; padding:0; }}
    figure.modelfig {{ break-inside:avoid; }}
    h2 {{ break-after:avoid; }}
  }}
</style>
</head>
<body>
<div class="page">

  <div class="titleblock">
    <div class="kicker">Technical Report</div>
    <h1>EEG-Based Epileptic Seizure Detection on the Bonn Dataset</h1>
    <p class="subtitle">A Leakage-Safe Comparison of ATCNet and EEGNet</p>
    <p class="meta">Single-channel intracranial &amp; scalp EEG &middot; TensorFlow / Keras implementation</p>
  </div>

  <div class="abstract">
    <h4>Abstract</h4>
    We adapt the EEG-ATCNet framework&mdash;originally designed for multi-channel motor-imagery
    decoding&mdash;to the task of epileptic seizure detection on the single-channel Bonn University
    EEG dataset. We emphasise a methodologically sound evaluation: data are partitioned at the
    level of original recordings <em>before</em> windowing, eliminating the segment-leakage that
    inflates many published results on this dataset. Two architectures are compared&mdash;the
    114k-parameter attention model ATCNet and a compact 2k-parameter EEGNet baseline&mdash;across a
    canonical binary task and a harder three-class task. Both models exceed 0.99 macro&nbsp;AUC and
    reach 0.95 seizure sensitivity on the binary task. Crucially, the small EEGNet matches ATCNet on
    every aggregate metric, indicating that ATCNet&rsquo;s spatial-attention machinery confers little
    advantage on single-channel input. ATCNet&rsquo;s sole consistent edge lies in seizure
    sensitivity, the clinically decisive direction.
  </div>

  <h2><span class="num">1</span>Introduction &amp; Objective</h2>
  <p>Automated detection of epileptic seizures from electroencephalography (EEG) is a long-standing
  problem in clinical signal processing. This report repurposes the EEG-ATCNet codebase, built for
  motor-imagery brain&ndash;computer interfaces, to classify seizure activity. The objective is twofold:
  to establish a <em>trustworthy</em> baseline on the widely used Bonn dataset, and to test whether a
  large attention-based architecture is warranted for single-channel seizure data, or whether a
  lightweight convolutional model suffices.</p>

  <h2><span class="num">2</span>Dataset</h2>
  <p>The Bonn dataset (Andrzejak et&nbsp;al., 2001) comprises five sets of 100 single-channel EEG
  segments each. Every segment contains 4,097 samples (~23.6&nbsp;s) recorded at 173.61&nbsp;Hz. The
  release is randomised with respect to patient and recording contact, so subject identities are not
  recoverable.</p>
  <div class="tabwrap">
    <div class="cap"><b>Table 1.</b> Composition of the five Bonn sets and our class assignment.</div>
    <table class="data">
      <tr><th>Set</th><th>Subjects</th><th>Condition</th><th>Recording site</th><th>Assigned class</th></tr>
      <tr><td>A&nbsp;(Z)</td><td>Healthy</td><td>Eyes open</td><td>Scalp (surface)</td><td>Healthy</td></tr>
      <tr><td>B&nbsp;(O)</td><td>Healthy</td><td>Eyes closed</td><td>Scalp (surface)</td><td>Healthy</td></tr>
      <tr><td>C&nbsp;(N)</td><td>Epilepsy</td><td>Interictal</td><td>Intracranial, opposite hemisphere</td><td>Interictal</td></tr>
      <tr><td>D&nbsp;(F)</td><td>Epilepsy</td><td>Interictal</td><td>Intracranial, epileptogenic zone</td><td>Interictal</td></tr>
      <tr><td>E&nbsp;(S)</td><td>Epilepsy</td><td><b>Ictal (seizure)</b></td><td>Intracranial, epileptogenic zone</td><td>Ictal / Seizure</td></tr>
    </table>
  </div>
  <p>We study two labelling schemes: a <strong>binary</strong> task (set&nbsp;E vs. the rest&mdash;the
  canonical Bonn benchmark) and a more demanding <strong>three-class</strong> task distinguishing
  healthy, interictal, and ictal states.</p>

  <h2><span class="num">3</span>Methodology</h2>

  <h3>3.1&ensp;Leakage-safe evaluation</h3>
  <div class="caution"><span class="lbl">Why many Bonn results are inflated.&ensp;</span>
  The popular tabular version of this dataset divides each 23.6&nbsp;s recording into roughly 23 short
  fragments. When those fragments are split into train/test at random, pieces of the <em>same physical
  recording</em> appear on both sides; the network memorises the recording rather than the underlying
  pathology, and reported accuracy approaches 99&ndash;100% spuriously.</div>
  <p>To avoid this, we split at the level of the 500 <em>original</em> segments&mdash;stratified
  into training, validation, and test (320&nbsp;/&nbsp;80&nbsp;/&nbsp;100 segments, i.e. 64&nbsp;/&nbsp;16&nbsp;/&nbsp;20%)&mdash;and only then window each segment. No window from a given
  recording ever crosses a split boundary. This is the central design decision of the study and the
  reason the reported figures are conservative and trustworthy.</p>

  <h3>3.2&ensp;Preprocessing</h3>
  <ol>
    <li>Load all 500 segments as one-dimensional signals of 4,097 samples.</li>
    <li>Stratified segment-level split into train / validation / test.</li>
    <li>Window each segment into 1,024-sample windows (~5.9&nbsp;s) with a 512-sample stride
        (7 windows per segment).</li>
    <li>Z-score standardisation using statistics computed on the training split only.</li>
    <li>Reshape to the model input contract <code>(N, 1, channels=1, 1024)</code>.</li>
  </ol>
  <p>This yields <strong>2,240</strong> training, <strong>560</strong> validation, and
  <strong>700</strong> test windows, preserving the 2&nbsp;:&nbsp;2&nbsp;:&nbsp;1 class ratio across
  every split.</p>

  <h3>3.3&ensp;Models and training protocol</h3>
  <p><strong>ATCNet</strong> (114k parameters) combines a convolutional block, multi-head
  self-attention, and a temporal convolutional network with sliding-window encoding. Its depthwise
  <em>spatial</em> convolution is intended for multi-channel montages; with a single channel that
  dimension is degenerate. <strong>EEGNet</strong> (2k parameters) is a compact, widely adopted EEG
  CNN used here as an honest reference point.</p>
  <p>Both models are trained identically: Adam optimiser (learning rate 10<sup>&minus;3</sup>),
  categorical cross-entropy loss, batch size 64, up to 150 epochs with early stopping (patience 30 on
  validation accuracy) and learning-rate reduction on plateau. Training is CPU-only, as TensorFlow on
  native Windows provides no GPU support.</p>

  <h3>3.4&ensp;Evaluation metrics</h3>
  <p>Because seizure detection is an imbalanced, safety-critical problem, we report
  <strong>seizure sensitivity</strong> (recall of the ictal class) and <strong>specificity</strong>
  explicitly, alongside overall accuracy, Cohen&rsquo;s&nbsp;&kappa;, and macro one-vs-rest AUC. Bare
  accuracy is an inadequate objective for a detector.</p>

  <h2><span class="num">4</span>Results</h2>
  <div class="tabwrap">
    <div class="cap"><b>Table 2.</b> Held-out test performance for both models on both tasks.
    Sz.&nbsp;Sens./Spec. denote seizure sensitivity and specificity.</div>
    {summary_table()}
  </div>
  <div class="note"><span class="lbl">Key observations.&ensp;</span>
  (i)&nbsp;On the binary benchmark both models reach 0.95 seizure sensitivity and ~0.99 AUC.
  (ii)&nbsp;On the three-class task EEGNet slightly leads on accuracy and &kappa; in under half
  the training time (5.1 vs. 12.0&nbsp;min). (iii)&nbsp;ATCNet retains a small but consistent advantage in seizure sensitivity
  (three-class: 0.907 vs. 0.886): it detects more seizures at the cost of marginally more false alarms.</div>

  <h2><span class="num">5</span>Figures</h2>
  {figures_html}

  <h2><span class="num">6</span>Discussion</h2>
  <p><strong>Model capacity should match the task.</strong> ATCNet earns its parameters on
  multi-channel motor imagery, where attention integrates information across electrodes. Bonn is
  single-channel, so that machinery is idle and a 2k-parameter network suffices. This is a property of
  the <em>dataset</em>, not a shortcoming of ATCNet.</p>
  <p><strong>Sensitivity is not accuracy.</strong> For a seizure detector the most consequential
  quantity is how many seizures are caught. ATCNet&rsquo;s higher sensitivity, despite occasionally
  lower headline accuracy, is arguably the preferable operating point: a missed seizure is costlier
  than a false alarm.</p>
  <p><strong>The hard boundary is interictal vs. healthy.</strong> The three-class confusion matrices
  show ictal segments are almost perfectly separated; nearly all residual error lies between the
  healthy and interictal classes&mdash;the genuinely subtle distinction. A leaky split would have masked
  this structure behind a near-perfect aggregate score.</p>

  <h2><span class="num">7</span>Limitations &amp; Future Work</h2>
  <ul>
    <li><strong>Saturated benchmark.</strong> Bonn is small (500 segments) and single-channel; AUCs
        near 0.99 leave little headroom. The results are a clean proof-of-concept, not a clinical claim.</li>
    <li><strong>No patient-level split.</strong> Because the release is randomised over patients and
        contacts, genuine cross-patient generalisation cannot be assessed on this dataset.</li>
    <li><strong>Next step.</strong> Port this pipeline&mdash;preserving the leakage-safe split and
        sensitivity-first metrics&mdash;to a multi-channel, imbalanced, cross-patient corpus such as
        CHB-MIT or the TUH EEG Seizure Corpus, where ATCNet&rsquo;s attention is expected to add value.</li>
  </ul>

  <h2><span class="num">8</span>Reproducibility</h2>
  <div class="tabwrap">
    <div class="cap"><b>Table 3.</b> Artifacts produced by the pipeline.</div>
    <table class="data">
      <tr><th>Artifact</th><th>File</th></tr>
      <tr><td>Leakage-safe data loader</td><td><code>preprocess_bonn.py</code></td></tr>
      <tr><td>Training &amp; evaluation</td><td><code>main_seizure.py</code></td></tr>
      <tr><td>Figure helpers</td><td><code>report_utils.py</code></td></tr>
      <tr><td>Results table</td><td><code>results_seizure/results_summary.csv</code></td></tr>
      <tr><td>Run log</td><td><code>results_seizure_run.log</code></td></tr>
    </table>
  </div>
  <p>All experiments use a fixed random seed (42). Reproduce with
  <code>python main_seizure.py</code> followed by <code>python generate_report.py</code>.</p>

  <footer>
    Dataset: R.&nbsp;G.&nbsp;Andrzejak et&nbsp;al., <em>Physical Review E</em> <b>64</b>, 061907 (2001).
    Models: EEG-ATCNet (Altaheri et&nbsp;al., <em>IEEE TII</em>, 2023).<br>
    This report was generated automatically from the experiment artifacts.
  </footer>

</div>
</body>
</html>'''

out = os.path.join(ROOT, 'report.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Wrote {out} ({len(HTML)/1024:.0f} KB of HTML; images embedded as base64)')
