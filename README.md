# The_4_Outliers_VinDatathon

<p align="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/d/de/HCMUT_official_logo.png" width="100" alt="HCMUT Logo">
  &nbsp;&nbsp;
  <img src="https://github.com/hanestuamante/The_4_Outliers_VinDatathon/blob/1e3b62cd035638f6267d83f039ec04b066218572/FTU_logo_2020%20(1).png" width="100" alt="FTU Logo">
</p>

## Overview
This repository contains the solution code for the VinDatathon 2026 Round 1 by team **The_4_Outliers**. Our solution utilizes a 7-Layer Hierarchical Ensemble Forecast combining CatBoost and XGBoost models to predict `Revenue` and `COGS`.

## Prerequisites
Ensure you have the required Python packages installed (e.g., `pandas`, `numpy`, `xgboost`, `catboost`, `jupyter`).

Please make sure the raw dataset is placed in the `datathon-2026-round-1` folder as follows:
- `datathon-2026-round-1/sales.csv`

## Reproduction Steps

To fully reproduce our final submission, please execute the following steps in order:

### 1. Data Preprocessing Phase 1
Run the first preprocessing notebook to clean the data and perform initial feature engineering.
```bash
jupyter nbconvert --to notebook --execute data_preprocessing_phase_1.ipynb
```
*(Alternatively, run it interactively via Jupyter Notebook/Lab)*

### 2. Data Preprocessing Phase 2
Run the second preprocessing notebook to generate the final feature sets. This process will output the required `.parquet` files (e.g., `shared_calendar.parquet`, `shared_daily.parquet`) into the `data/features/` directory.
```bash
jupyter nbconvert --to notebook --execute data_preprocessing_phase_2.ipynb
```
*(Alternatively, run it interactively via Jupyter Notebook/Lab)*

### 3. Model Training & Inference
Run the master pipeline script to train the ensemble models and generate the final predictions.
```bash
cd model
python model.py
```

This will run our 7-layer hierarchical ensemble pipeline internally and generate the final `submission.csv` inside the `model/` directory. No external intermediate files or previous base models are required.

## Repository Structure
- `data_preprocessing_phase_1.ipynb`: Initial data cleaning and preprocessing.
- `data_preprocessing_phase_2.ipynb`: Feature engineering and generation of parquet files.
- `model/model.py`: The main training and inference pipeline.
- `model/submission.csv`: Final prediction output.
- `datathon-2026-round-1/`: Directory containing raw input data.
- `data/`: Directory where generated features and parquet files are stored.
