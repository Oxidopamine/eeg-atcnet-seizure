"""
Loader for the Bonn University EEG dataset (Andrzejak et al., 2001).

Sets (each = 100 single-channel TXT files, 4096 ASCII samples @ 173.61 Hz):
    Z (A) : healthy volunteer, eyes open,  scalp
    O (B) : healthy volunteer, eyes closed, scalp
    N (C) : epilepsy patient, interictal, hippocampal formation (opposite hemisphere)
    F (D) : epilepsy patient, interictal, epileptogenic zone
    S (E) : epilepsy patient, ICTAL (during seizure), epileptogenic zone

The dataset is randomized w.r.t. recording contact / patient, so patient-level
splitting is impossible. The finest leakage-free unit is the original 4096-sample
SEGMENT. We therefore split on segment IDs *before* windowing.

Output contract matches what models.py expects:
    X : (n_trials, 1, n_channels, in_samples)   with n_channels = 1
    y : one-hot (n_trials, n_classes)
"""

import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Map each set folder to a class index under the chosen scheme.
# 3-class (default): healthy / interictal / ictal
LABEL_SCHEMES = {
    '3class': {'Z': 0, 'O': 0, 'N': 1, 'F': 1, 'S': 2},
    # binary: seizure (ictal) vs non-seizure
    'binary': {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1},
    # 5-class: each set is its own class
    '5class': {'Z': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4},
}

CLASS_LABELS = {
    '3class': ['Healthy', 'Interictal', 'Ictal'],
    'binary': ['Non-seizure', 'Seizure'],
    '5class': ['A(Z)', 'B(O)', 'C(N)', 'D(F)', 'E(S)'],
}


def _load_set(data_path, set_name):
    """Load all 100 segments of one set as an array (100, 4096).

    Handles the mixed-case extensions in the official release (set N ships
    as *.TXT, the others as *.txt).
    """
    folder = os.path.join(data_path, set_name)
    files = sorted(glob.glob(os.path.join(folder, '*.txt')) +
                   glob.glob(os.path.join(folder, '*.TXT')))
    files = sorted(set(files))  # de-dup on case-insensitive filesystems
    if not files:
        raise FileNotFoundError(
            f"No TXT files found in {folder}. Expected the extracted set "
            f"'{set_name}' (e.g. {set_name}001.txt).")
    segments = [np.loadtxt(f, dtype=np.float32) for f in files]
    return np.stack(segments, axis=0)  # (n_segments, 4096)


def _window(segment, win_len, stride):
    """Slice a 1-D segment into overlapping windows -> (n_win, win_len)."""
    starts = range(0, len(segment) - win_len + 1, stride)
    return np.stack([segment[s:s + win_len] for s in starts], axis=0)


def get_bonn_data(data_path, scheme='3class', win_len=1024, stride=512,
                  test_size=0.2, val_size=0.2, isStandard=True, seed=42):
    """Load Bonn EEG, split at the SEGMENT level, then window each split.

        Parameters
        ----------
        data_path : str   folder containing extracted sets Z/ O/ N/ F/ S/
        scheme    : str   '3class' (default), 'binary', or '5class'
        win_len   : int   samples per window (1024 ~= 5.9 s)
        stride    : int   hop between windows within a segment
        test_size : float fraction of segments held out for test
        val_size  : float fraction of the *remaining* segments used for val
        isStandard: bool  z-score using statistics fit on the training split only

        Returns
        -------
        (X_train, y_train, X_val, y_val, X_test, y_test) where
        X_* : (n_trials, 1, 1, win_len) float32 ; y_* : one-hot
    """
    mapping = LABEL_SCHEMES[scheme]

    # 1) Load every segment with a stable per-segment label.
    seg_signals, seg_labels = [], []
    for set_name, cls in mapping.items():
        arr = _load_set(data_path, set_name)          # (100, 4096)
        seg_signals.append(arr)
        seg_labels.append(np.full(arr.shape[0], cls, dtype=int))
    seg_signals = np.concatenate(seg_signals, axis=0)  # (500, 4096)
    seg_labels = np.concatenate(seg_labels, axis=0)    # (500,)

    # 2) Split on SEGMENT indices (stratified) BEFORE windowing -> no leakage.
    idx = np.arange(len(seg_labels))
    idx_tr, idx_test = train_test_split(
        idx, test_size=test_size, stratify=seg_labels, random_state=seed)
    idx_tr, idx_val = train_test_split(
        idx_tr, test_size=val_size, stratify=seg_labels[idx_tr],
        random_state=seed)

    # 3) Window within each split.
    def build(indices):
        Xs, ys = [], []
        for i in indices:
            w = _window(seg_signals[i], win_len, stride)   # (n_win, win_len)
            Xs.append(w)
            ys.append(np.full(w.shape[0], seg_labels[i], dtype=int))
        X = np.concatenate(Xs, axis=0)                     # (N, win_len)
        y = np.concatenate(ys, axis=0)
        return X, y

    X_train, y_train = build(idx_tr)
    X_val,   y_val   = build(idx_val)
    X_test,  y_test  = build(idx_test)

    # 4) Standardize using TRAIN statistics only (single channel -> one scaler).
    if isStandard:
        scaler = StandardScaler().fit(X_train.reshape(-1, 1))
        X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_val   = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_test  = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # 5) Reshape to model contract (N, 1, n_channels=1, win_len) and one-hot.
    n_classes = len(CLASS_LABELS[scheme])
    reshape = lambda X: X[:, np.newaxis, np.newaxis, :].astype(np.float32)
    onehot = lambda y: to_categorical(y, num_classes=n_classes)

    return (reshape(X_train), onehot(y_train),
            reshape(X_val),   onehot(y_val),
            reshape(X_test),  onehot(y_test))


if __name__ == '__main__':
    # Quick smoke test / shape report.
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bonn_data')
    Xtr, ytr, Xva, yva, Xte, yte = get_bonn_data(here)
    print('Train:', Xtr.shape, ytr.shape, '| class counts:', ytr.sum(0))
    print('Val:  ', Xva.shape, yva.shape, '| class counts:', yva.sum(0))
    print('Test: ', Xte.shape, yte.shape, '| class counts:', yte.sum(0))
