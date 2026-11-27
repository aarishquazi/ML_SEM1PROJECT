# ECG Classification on PTB-XL and PTB-XL+

**A comparative study of waveform, image, and feature-based multi-label ECG classification**

This repository contains our first-semester Machine Learning project. We investigate how different representations of a 12-lead electrocardiogram (ECG) affect classification performance by comparing conventional machine learning and deep learning models on **PTB-XL** and **PTB-XL+**.

Our focus is predictive performance, dimensionality reduction, computational efficiency, and reproducibility. This project is a research benchmark, not a clinically validated diagnostic system.

## Project team

| Name | Roll number | Role |
| --- | --- | --- |
| **Aarish Quazi** | M26MER008 | Team Leader |
| Ritesh Kumar | M26CSE008 | Team Member |
| Kartik Bharti | M26CSE016 | Team Member |

**Repository:** [aarishquazi/ML_SEM1PROJECT](https://github.com/aarishquazi/ML_SEM1PROJECT)

## Project overview

| Experiment | Input representation | Prediction targets |
| --- | --- | --- |
| PTB-XL+ image classification | Rendered 12SL median beats, HOG features | Top 5, top 10 and 53 automated ECG statements |
| PTB-XL+ numerical classification | 12-lead median beats, 500 Hz, 600 samples per lead | Same 5, 10 and 53 statements |
| PTB-XL+ feature classification | 782 engineered 12SL measurements, with and without PCA | Same 5, 10 and 53 statements |
| Original PTB-XL classification | Full 10-second, 12-lead ECG waveforms at 100 Hz | NORM, MI, STTC, CD, HYP diagnostic superclasses |

The PTB-XL+ statements are machine-generated interpretation labels, whereas original PTB-XL uses diagnostic superclasses derived from annotated SCP-ECG statements. Both problems are multi-label, but their target spaces are not interchangeable.

## Repository structure

The repository contains nine experiment notebooks and an environment requirements file. The two preprocessing scripts are part of the project workflow and should be added before end-to-end reproduction.

```text
ML_SEM1PROJECT/
├── ECG_Classification_2DCNN_5.ipynb
├── ECG_2DCNN_Classification_top10_targets.ipynb
├── ECG_Classification_2DCNN_53.ipynb
├── NumericalECG_5.ipynb
├── NumericalECG_10.ipynb
├── NumericalECG_53.ipynb
├── csv_ECG_Classification_12SL_Features.ipynb
├── csv_ECG_Classification_12SL_Features_PCA.ipynb
├── PTBXL_main10sec/
│   └── PTBXL_100Hz_5Class_CNN.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

### Notebook descriptions

| Notebook | Purpose |
| --- | --- |
| [ECG_Classification_2DCNN_5](ECG_Classification_2DCNN_5.ipynb) | Image classification for five 12SL statements |
| [ECG_2DCNN_Classification_top10_targets](ECG_2DCNN_Classification_top10_targets.ipynb) | Image classification for ten statements |
| [ECG_Classification_2DCNN_53](ECG_Classification_2DCNN_53.ipynb) | Shared fold index, HOG baselines and 2D CNN for 53 statements |
| [NumericalECG_5](NumericalECG_5.ipynb) | Numerical median-beat classification for five statements |
| [NumericalECG_10](NumericalECG_10.ipynb) | Numerical median-beat classification for ten statements |
| [NumericalECG_53](NumericalECG_53.ipynb) | Numerical median-beat classification for 53 statements |
| [csv_ECG_Classification_12SL_Features](csv_ECG_Classification_12SL_Features.ipynb) | Classical ML and MLP on original 12SL features |
| [csv_ECG_Classification_12SL_Features_PCA](csv_ECG_Classification_12SL_Features_PCA.ipynb) | Classification after PCA |
| [PTBXL_100Hz_5Class_CNN](PTBXL_main10sec/PTBXL_100Hz_5Class_CNN.ipynb) | Full-waveform benchmark and held-out test evaluation |

## Datasets and protocol

The datasets are publicly available through PhysioNet and are **not distributed in this repository**.

- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) — ECG recordings, SCP-ECG statements and patient-wise folds.
- [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/) — engineered measurements, median beats and automated interpretation statements.

PTB-XL+ uses 20,914 matched ECGs from 18,202 patients. Our original PTB-XL notebook uses metadata consistent with [version 1.0.1](https://physionet.org/content/ptb-xl/1.0.1/), containing 21,837 recordings before filtering; 21,430 remain after excluding records without diagnostic superclasses. The PTB-XL+ metadata contains 21,799 records, consistent with a later release. Match the corresponding releases and ECG IDs when reproducing the historical runs.

| Partition | Official folds | PTB-XL+ | Original PTB-XL |
| --- | --- | ---: | ---: |
| Training | 1–8 | 16,608 | 17,111 |
| Validation | 9 | 2,148 | 2,156 |
| Test | 10 | 2,158 | 2,163 |
| **Total** | | **20,914** | **21,430** |

Preprocessing and label selection use training data; model selection and the original PTB-XL threshold optimization use validation data. Fold 10 is reserved for held-out evaluation.

## Methodology

1. Validate WFDB records, decode ECG signals, standardize lead order and attach labels by ECG ID.
2. Render median beats and extract HOG descriptors for conventional image models; train a 2D CNN on the images.
3. Train MLP and 1D CNN architectures on numerical median beats, alongside PCA-based conventional baselines.
4. Impute and standardize the clinical measurements; compare original 12SL features against PCA-transformed features.
5. Train a compact 1D CNN and PCA-based baselines on original 100 Hz full-length ECGs.
6. Evaluate macro AUROC, macro average precision (AP), macro F1 and micro F1. Binary accuracy and exact-match accuracy are reported where recorded.

Models evaluated include Logistic Regression, Linear SVM, Random Forest, XGBoost, AdaBoost, MLP, 1D CNN and 2D CNN. The neural classifiers use independent sigmoid outputs and binary cross-entropy, AdamW, learning-rate reduction, and early stopping based on validation macro AP. The main notebooks use random seed 42.

### PCA experiments

| Representation | Original dimensions | PCA dimensions | Training variance retained |
| --- | ---: | ---: | ---: |
| 12SL measurements | 782 | 197 | 95.02% |
| Numerical median beats | 7,200 | 256 | 99.72% |
| Full 100 Hz ECGs | 12,000 | 256 | 56.19% |

## Experimental results

### PTB-XL+: comparison across representations

Each cell reports **macro AUROC / macro AP** on validation fold 9. The team observed comparable performance in separate test runs; the historical values below are recorded validation results and represent estimates of held-out performance, not exact fold-10 measurements.

| Model / representation | 5 labels | 10 labels | 53 labels |
| --- | ---: | ---: | ---: |
| 2D CNN — ECG images | 0.9114 / 0.7958 | 0.9331 / 0.7681 | 0.9231 / 0.4896 |
| 1D CNN — numerical median beats | 0.9297 / 0.8370 | 0.9512 / 0.8201 | 0.9357 / 0.5294 |
| Random Forest — original 12SL | 0.9371 / 0.8634 | 0.9550 / **0.8710** | 0.9356 / 0.6358 |
| XGBoost — original 12SL | **0.9423 / 0.8682** | **0.9577** / 0.8700 | **0.9441 / 0.6883** |
| MLP — 12SL PCA | 0.9236 / 0.8309 | 0.9461 / 0.8128 | 0.9240 / 0.5245 |

XGBoost on original measurements achieves the highest macro AUROC across all three label sets. Random Forest has slightly higher AP for the ten-label task. Numerical 1D CNN consistently outperforms the rendered-image 2D CNN.

### Effect of PCA on the 53-label feature experiment

| Model | Original 782 features | PCA (197 components) |
| --- | ---: | ---: |
| Logistic Regression | 0.8832 | 0.8897 |
| Linear SVM | 0.8721 | 0.8832 |
| Random Forest | 0.9356 | 0.8699 |
| XGBoost | **0.9441** | 0.9023 |
| AdaBoost | 0.9245 | 0.8826 |
| MLP | 0.9269 | 0.9240 |

Values are validation macro AUROC. PCA reduces dimensionality substantially but does not consistently improve classification, especially for tree-based methods. Estimator settings also differ across some runs, so this is an exploratory ablation.

### Original PTB-XL: held-out test performance

These are actual **fold-10 test results**. The F1 values use thresholds chosen on fold 9.

| Model | Macro AUROC | Macro AP | Macro F1 | Micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (PCA) | 0.5050 | 0.2794 | 0.4014 | 0.4112 |
| Linear SVM (PCA) | 0.5051 | 0.2799 | 0.3952 | 0.4397 |
| Random Forest (PCA) | 0.6718 | 0.4207 | 0.4650 | 0.4896 |
| **1D CNN** | **0.9228** | **0.8159** | **0.7465** | **0.7764** |

Our compact 1D CNN contains **743,749 trainable parameters** and the recorded 14-epoch training run took **129.86 seconds** on an NVIDIA GeForce RTX 3050 Laptop GPU. Its validation AUROC of 0.9252 is close to its test AUROC of 0.9228.

AUROC is a discrimination metric, not the percentage of correctly classified labels. Binary accuracy and exact-match accuracy are separate metrics in a multi-label task.

## Installation

Python 3.11 is a recommended starting point.

```bash
git clone https://github.com/aarishquazi/ML_SEM1PROJECT.git
cd ML_SEM1PROJECT

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, use the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to select a build compatible with your CUDA setup. The requirements file lists project dependencies; exact historical package versions were not captured uniformly.

## Reproduction instructions

### 1. Prepare the local datasets

Download and extract the PhysioNet datasets as follows. Do not upload the raw data or generated per-ECG caches to GitHub.

```text
ML_SEM1PROJECT/
├── ptbxl_database.csv              # PTB-XL+ metadata (21,799 rows)
├── data/
│   ├── ptb-xl-plus/
│   │   ├── median_beats/
│   │   │   └── 12sl/              # extracted WFDB files
│   │   ├── features/
│   │   │   └── 12sl_features.csv
│   │   └── labels/
│   │       └── 12sl_statements.csv
│   └── processed/                # generated locally
└── PTBXL_main10sec/
    └── data/
        ├── ptbxl_database.csv     # original metadata (21,837 rows)
        ├── scp_statements.csv
        ├── records100/            # retain original WFDB subdirectories
        └── processed/            # generated locally
```

### 2. Generate median-beat images and attach labels

After placing the project's preprocessing scripts at the repository root, run:

```bash
python convert_12sl_to_images.py --all --features-csv data/ptb-xl-plus/features/12sl_features.csv
python attach_12sl.py
```

These scripts generate the images, manifest and training index under `data/processed/12sl_median_images_final/`.

### 3. Run PTB-XL+ notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Recommended execution order:

1. `ECG_Classification_2DCNN_53.ipynb` — creates the shared fold index and 53-label order.
2. `ECG_Classification_2DCNN_5.ipynb` — reuses the image feature cache.
3. `NumericalECG_53.ipynb`, `NumericalECG_10.ipynb` and `NumericalECG_5.ipynb`.
4. `csv_ECG_Classification_12SL_Features.ipynb`.
5. `csv_ECG_Classification_12SL_Features_PCA.ipynb`.

Run the ten-label image notebook in a separate working copy or configure distinct output paths before execution. Its original checkpoint and cache filenames overlap those of the 53-label notebook.

The two feature notebooks load earlier preprocessing and fitted Logistic Regression/Random Forest artifacts from `data/processed/12sl_feature_models/top_53/`. Restore or regenerate the label and feature order, fitted imputer and scaler, fold IDs, target arrays and baseline models first. Cross-run summary cells may require outputs from other notebooks.

### 4. Run original PTB-XL

From the repository root:

```bash
cd PTBXL_main10sec
jupyter lab
```

Open and execute `PTBXL_100Hz_5Class_CNN.ipynb` with the notebook working directory set to `PTBXL_main10sec/`. It constructs the five diagnostic labels, loads 100 Hz waveforms, trains the PCA baselines and 1D CNN, selects thresholds on fold 9, and evaluates fold 10.

### 5. Preserve reproducibility artifacts

Save aggregate metrics, split IDs, label order, model configurations, training histories, and the installed environment. Keep large generated data, arrays and model checkpoints out of Git.

```bash
python -m pip freeze > environment-lock.txt
```

> **Storage note:** The current .gitignore permits the processed-data directories. Check the staged files before committing and exclude large generated arrays, waveform/image caches and model checkpoints.

## Research report and reproducibility statement

The project is documented in an eight-page CVPR-format LaTeX report titled *A Comparative Study of Waveform, Image, and Feature-Based Multi-Label ECG Classification on PTB-XL and PTB-XL+*.

The reported results are derived from executed project notebooks. Data splits, model configurations, preprocessing, label selection and random seeds are documented in the notebooks. Original PTB-XL performance is measured on the held-out test partition; PTB-XL+ validation scores are used as representative estimates of test performance.

A clean-machine end-to-end rerun remains to be verified. The missing preprocessing and baseline artifacts must be added or regenerated. The report and running code should be versioned together and the exact installed environment recorded when finalizing the repository.

## References

1. Wagner et al. (2020). [PTB-XL, a large publicly available electrocardiography dataset](https://doi.org/10.1038/s41597-020-0495-6). *Scientific Data*.
2. Strodthoff et al. (2021). [Deep learning for ECG analysis: Benchmarks and insights from PTB-XL](https://doi.org/10.1109/JBHI.2020.3022989). *IEEE Journal of Biomedical and Health Informatics*.
3. Strodthoff et al. (2023). [PTB-XL+, a comprehensive electrocardiographic feature dataset](https://doi.org/10.1038/s41597-023-02153-8). *Scientific Data*.
4. Sangha et al. (2022). [Automated multilabel diagnosis on electrocardiographic images and signals](https://doi.org/10.1038/s41467-022-29153-3). *Nature Communications*.
5. Zhou et al. (2026). [Interpretable detection of left ventricular hypertrophy using commercial ECG features and machine learning](https://doi.org/10.3389/fcvm.2026.1825829). *Frontiers in Cardiovascular Medicine*.
6. Metre et al. (2026). [XAI-ECGNet: an explainable hybrid CNN-BiLSTM framework for multi-label cardiac diagnosis](https://doi.org/10.1038/s41598-026-76799-w). *Scientific Reports*.

We acknowledge the developers of PTB-XL, PTB-XL+, PhysioNet, WFDB, scikit-learn, XGBoost, and PyTorch for making this research possible.
