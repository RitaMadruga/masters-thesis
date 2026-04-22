# Thesis Rita: Multi-Omics Integration and Tumor Subtype Classification in TCGA-PAAD

## Introduction

The current version of the project supports reproduction of:

- preprocessing of the multi-omics data
- training of MOFA models for data integration and dimensionality reduction
- tumor subtype classification using the MOFA latent factors

## Repository Structure

```text
.
├── notebooks/
│   ├── downstream/
│   ├── exploratory_analysis/
│   └── preprocessing/
├── src/
│   ├── mofa_train.py
│   ├── subtype_classification_common.py
│   ├── subtype_classification_nested_cv.py
│   └── subtype_classification_train_test_split.py
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

## Data

The `data/` directory is not included in the GitHub repository and must be downloaded separately from the following link:

[Google Drive data folder](https://drive.google.com/drive/folders/1RheZhuHbIdOnLUdtS3k-CcvwFAmkRjxx?usp=sharing)

After downloading it, place the `data/` folder in the project root as follows:

```text
thesis_rita/
├── data/
├── src/
├── notebooks/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment and Dependencies

This project uses:
- `pyproject.toml` to define the project dependencies and metadata
- `uv.lock` to fix the exact versions of the dependencies used in the project

This helps make the environment reproducible across machines.

If `uv` is installed, the environment can be recreated with:

```bash
uv sync
```

If a local virtual environment is preferred, make sure the required dependencies are installed according to `pyproject.toml` and `uv.lock`.

## Notebooks

The notebooks included in this repository were mainly used for:

- data preprocessing
- exploratory data analysis
- testing classification on a single MOFA model
- preliminary results interpretation

The full reproducible runs across the 4 MOFA models were performed using the scripts in `src/`.

## Running the Pipeline

### 1. Train the MOFA models

```bash
.\.venv\Scripts\python.exe src/mofa_train.py
```

### 2. Run the classification experiments

The commands below can be used either to reproduce the full set of classification results or to run smaller subsets of the experiments.

#### 2.1 Full runs

These commands reproduce the complete classification experiments across the 4 MOFA models and generate the results stored in `data/classification_results/`.

Train/test split classification across the 4 MOFA models:

```bash
.\.venv\Scripts\python.exe src\subtype_classification_train_test_split.py --n-seeds 30
```

Nested cross-validation classification across the 4 MOFA models:

```bash
.\.venv\Scripts\python.exe src\subtype_classification_nested_cv.py --n-seeds 30
```

#### 2.2 Custom runs

To reduce runtime, it is also possible to run only a subset of the experiments, for example by selecting:

- a specific classifier
- a specific MOFA model
- a smaller number of seeds

Example 1: run only the Linear SVM classifier:

```bash
.\.venv\Scripts\python.exe src\subtype_classification_train_test_split.py --n-seeds 30 --classifiers linear_svm
```

Example 2: run nested cross-validation only for one MOFA model:

```bash
.\.venv\Scripts\python.exe src\subtype_classification_nested_cv.py --n-seeds 30 --models mofa_trained_lg2
```

## Important Note

- If you re-run the notebooks or scripts, the processed data, trained models, and result files may be overwritten.
