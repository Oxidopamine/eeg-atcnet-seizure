"""
Seizure detection on the Bonn University EEG dataset using the EEG-ATCNet models.

Why this is a separate script from main_TrainValTest.py
-------------------------------------------------------
main_TrainValTest.py is built around multi-subject, multi-channel motor-imagery
trials (a per-subject loop + LOSO). The Bonn data has neither subject IDs nor
multiple channels, so that scaffolding does not apply. Here we:
  * split at the SEGMENT level (the finest leakage-free unit) -- see preprocess_bonn
  * train ATCNet (with Chans=1) AND EEGNet as an honest single-channel baseline
  * report clinical metrics (seizure sensitivity/specificity, macro-AUC, kappa),
    not bare accuracy, since the goal is detection of a rare/critical class.

Run:  .venv/Scripts/python.exe main_seizure.py
"""

import os
import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (confusion_matrix, accuracy_score, cohen_kappa_score,
                             roc_auc_score, classification_report)

from main_TrainValTest import getModel
from preprocess_bonn import get_bonn_data, CLASS_LABELS
from report_utils import plot_learning_curves, plot_confusion, plot_roc

#%% Configuration
SCHEMES     = ['binary', '3class']   # run both: canonical benchmark + harder task
WIN_LEN     = 1024             # samples per window (~5.9 s @ 173.61 Hz)
STRIDE      = 512
MODELS      = ['ATCNet', 'EEGNet']
EPOCHS      = 150
BATCH_SIZE  = 64
PATIENCE    = 30
LR          = 1e-3
SEED        = 42

DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bonn_data')
RESULTS     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_seizure')


def seizure_sens_spec(y_true, y_pred, seizure_class):
    """Binary sensitivity/specificity for the seizure (ictal) class vs the rest."""
    t = (y_true == seizure_class).astype(int)
    p = (y_pred == seizure_class).astype(int)
    tp = np.sum((t == 1) & (p == 1)); fn = np.sum((t == 1) & (p == 0))
    tn = np.sum((t == 0) & (p == 0)); fp = np.sum((t == 0) & (p == 1))
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return sens, spec


def run_scheme(scheme, summary_rows):
    """Train + evaluate every model under one labelling scheme; save figures."""
    out_dir = os.path.join(RESULTS, scheme)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"#"*72}\n# SCHEME: {scheme}\n{"#"*72}')
    print('Loading Bonn data (segment-level split, then windowing)...')
    Xtr, ytr, Xva, yva, Xte, yte = get_bonn_data(
        DATA_PATH, scheme=scheme, win_len=WIN_LEN, stride=STRIDE, seed=SEED)
    labels = CLASS_LABELS[scheme]
    n_classes = len(labels)
    seizure_class = labels.index('Ictal') if 'Ictal' in labels else \
                    (labels.index('Seizure') if 'Seizure' in labels else n_classes - 1)

    print(f'  train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}')
    print(f'  classes: {labels} | seizure class index: {seizure_class}')

    dataset_conf = {'n_classes': n_classes, 'n_channels': 1, 'in_samples': WIN_LEN}

    for model_name in MODELS:
        tag = f'{scheme}/{model_name}'
        print(f'\n{"="*60}\nTraining {tag}\n{"="*60}')
        model = getModel(model_name, dataset_conf)
        model.compile(optimizer=Adam(learning_rate=LR),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        ckpt = os.path.join(out_dir, f'{model_name}_best.weights.h5')
        callbacks = [
            ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True,
                            save_weights_only=True, mode='max'),
            EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                          mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5),
        ]

        t0 = time.time()
        history = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                            batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
        train_min = (time.time() - t0) / 60

        # ---- Evaluation on the held-out test set ----
        model.load_weights(ckpt)
        proba = model.predict(Xte, verbose=0)
        y_pred = proba.argmax(axis=1)
        y_true = yte.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            macro_auc = roc_auc_score(yte, proba, multi_class='ovr', average='macro')
        except ValueError:
            macro_auc = float('nan')
        sens, spec = seizure_sens_spec(y_true, y_pred, seizure_class)
        cm = confusion_matrix(y_true, y_pred)

        print(f'\n--- {tag} test results ---')
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))
        print('Confusion matrix (rows=true, cols=pred):'); print(cm)
        print(f'Accuracy={acc:.4f}  kappa={kappa:.4f}  macroAUC={macro_auc:.4f}')
        print(f'Seizure sensitivity={sens:.4f}  specificity={spec:.4f}  '
              f'(trained in {train_min:.1f} min)')

        # ---- Report figures ----
        title = f'{model_name} ({scheme})'
        plot_learning_curves(history.history,
                             os.path.join(out_dir, f'{model_name}_learning.png'), title)
        plot_confusion(cm, labels,
                       os.path.join(out_dir, f'{model_name}_confusion.png'), title)
        plot_roc(yte, proba, labels,
                 os.path.join(out_dir, f'{model_name}_roc.png'), title)

        summary_rows.append({
            'scheme': scheme, 'model': model_name, 'accuracy': acc, 'kappa': kappa,
            'macro_auc': macro_auc, 'seizure_sens': sens, 'seizure_spec': spec,
            'params': model.count_params(), 'train_min': train_min,
        })


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tf.random.set_seed(SEED); np.random.seed(SEED)

    summary_rows = []
    for scheme in SCHEMES:
        run_scheme(scheme, summary_rows)

    # ---- Console summary ----
    print(f'\n{"="*82}\nOVERALL SUMMARY\n{"="*82}')
    print(f'{"Scheme":<9}{"Model":<10}{"Acc":>8}{"Kappa":>8}{"AUC":>8}'
          f'{"Sz-Sens":>9}{"Sz-Spec":>9}{"Params":>9}{"Train(m)":>10}')
    for r in summary_rows:
        print(f'{r["scheme"]:<9}{r["model"]:<10}{r["accuracy"]:>8.4f}{r["kappa"]:>8.4f}'
              f'{r["macro_auc"]:>8.4f}{r["seizure_sens"]:>9.4f}{r["seizure_spec"]:>9.4f}'
              f'{r["params"]:>9d}{r["train_min"]:>10.1f}')

    # ---- CSV for the report ----
    csv_path = os.path.join(RESULTS, 'results_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)
    print(f'\nSaved results table -> {csv_path}')
    print(f'Saved figures -> {RESULTS}/<scheme>/<model>_{{learning,confusion,roc}}.png')


if __name__ == '__main__':
    main()
