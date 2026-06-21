# Seizure Detection on the Bonn EEG Dataset

Detecting epileptic seizure phases from EEG, built on top of the ATCNet architecture.
This project adapts ATCNet (originally designed for multi-channel motor-imagery decoding) to the
single-channel **Bonn University EEG dataset**, and benchmarks it against an EEGNet baseline under a
**leakage-safe** evaluation.

> **Built on prior work.** The model architectures (ATCNet, EEGNet, and others) come from the
> open-source [**Altaheri/EEG-ATCNet**](https://github.com/Altaheri/EEG-ATCNet) repository
> (Apache-2.0), by Hamdi Altaheri et al. All credit for those models belongs to the original
> authors — see [Credits & License](#credits--license). The seizure-detection work described
> below is our contribution.

## Our contribution

| File | What it does | Author |
|------|--------------|--------|
| [`preprocess_bonn.py`](preprocess_bonn.py) | Loads the 5 Bonn sets and **splits at the original-segment level *before* windowing** to prevent data leakage; supports `binary` / `3class` / `5class` schemes. | This project |
| [`main_seizure.py`](main_seizure.py) | Trains ATCNet (single-channel) and EEGNet; reports accuracy, Cohen's κ, macro-AUC, and seizure sensitivity/specificity; saves figures and a results CSV. | This project |
| [`report_utils.py`](report_utils.py) | Learning-curve, confusion-matrix, and ROC plotting helpers. | This project |
| [`generate_report.py`](generate_report.py) | Builds a self-contained HTML results report. | This project |
| `models.py`, `attention_models.py`, `preprocess.py`, `main_TrainValTest.py`, ... | ATCNet/EEGNet model definitions and the original motor-imagery pipeline. | Altaheri et al. (upstream) |

## Setup & run

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/macOS: source .venv/bin/activate)
pip install tensorflow numpy matplotlib scikit-learn scipy mne python-dateutil

# Download the Bonn dataset (sets Z, O, N, F, S) and extract each into bonn_data/<SET>/
#   Andrzejak et al. (2001):  http://www.upf.edu/web/ntsa/downloads

python main_seizure.py        # trains both models on the binary + 3-class tasks
python generate_report.py     # builds report.html from the saved results
```

Trained weights for both tasks are included under `results_seizure/<task>/`, so the results can be
reproduced without retraining.

## Method

The Bonn dataset is single-channel EEG, organised as 5 sets of 100 segments (~23.6 s each at
173.61 Hz). We map them to a **binary** task (seizure vs. non-seizure, the canonical benchmark) and
a harder **3-class** task (healthy / interictal / ictal).

The key methodological choice is the **leakage-safe split**: we partition the 500 original segments
into train/validation/test *before* slicing them into windows, so no fragment of a recording appears
in more than one split. Splitting after windowing (as many public examples do) leaks recordings
across splits and inflates accuracy toward ~99%.

## Key results (held-out test set)

| Task | Model | Accuracy | κ | Macro-AUC | Seizure Sens. | Seizure Spec. |
|------|-------|----------|---|-----------|---------------|---------------|
| binary  | ATCNet | 0.970 | 0.908 | 0.995 | 0.950 | 0.975 |
| binary  | EEGNet | 0.967 | 0.900 | 0.993 | 0.950 | 0.971 |
| 3-class | ATCNet | 0.953 | 0.926 | 0.993 | 0.907 | 0.986 |
| 3-class | EEGNet | 0.957 | 0.933 | 0.993 | 0.886 | 0.993 |

The compact EEGNet (~2k params) matches the much larger ATCNet (~114k params) on this single-channel
data, since ATCNet's spatial-attention advantage relies on multiple EEG channels. The intended next
step is the multi-channel TUH EEG corpus.

## Credits & License

The model implementations and the original motor-imagery pipeline in this repository are the work of
**Hamdi Altaheri, Ghulam Muhammad, and Mansour Alsulaiman**, released under the **Apache-2.0** license
(see [`LICENSE`](LICENSE)). Original repository: https://github.com/Altaheri/EEG-ATCNet

Please cite the original ATCNet papers when using the model code:

```bibtex
@article{9852687,
  title   = {Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification},
  author  = {Altaheri, Hamdi and Muhammad, Ghulam and Alsulaiman, Mansour},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2023}, volume = {19}, number = {2}, pages = {2249--2258},
  doi     = {10.1109/TII.2022.3197419}
}
```

The EEGNet baseline is from Lawhern et al. (2018), [arXiv:1611.08024](https://arxiv.org/abs/1611.08024).
The Bonn dataset is from Andrzejak et al., *Physical Review E* 64, 061907 (2001).

Our seizure-detection additions are released under the same Apache-2.0 license. Portions of this work
(pipeline, evaluation code, and the report) were developed with the assistance of the AI tool Claude.
