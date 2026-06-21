"""
Reporting helpers for the Bonn seizure experiments: learning curves,
confusion matrices, and ROC curves saved as publication-quality figures.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: write files, never block on a window
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay


def plot_learning_curves(history, out_path, title):
    """Train/val accuracy and loss across epochs -> single 2-panel figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history['accuracy'], label='train')
    ax1.plot(history['val_accuracy'], label='val')
    ax1.set_title(f'{title} — accuracy'); ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['loss'], label='train')
    ax2.plot(history['val_loss'], label='val')
    ax2.set_title(f'{title} — loss'); ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss'); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_confusion(cm, labels, out_path, title):
    """Count + row-normalized confusion matrices side by side."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax1, cmap='Blues', colorbar=False, values_format='d')
    ax1.set_title(f'{title} — counts')
    ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
        ax=ax2, cmap='Blues', colorbar=False, values_format='.2f')
    ax2.set_title(f'{title} — normalized')
    for ax in (ax1, ax2):
        ax.set_xticklabels(labels, rotation=20, ha='right')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_roc(y_onehot, proba, labels, out_path, title):
    """One-vs-rest ROC curve per class with AUC in the legend."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    n_classes = y_onehot.shape[1]
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], proba[:, c])
        ax.plot(fpr, tpr, label=f'{labels[c]} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
    ax.set_title(f'{title} — ROC (one-vs-rest)'); ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
