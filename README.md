# Mushroom Classification

Binary classification pipeline — edible vs poisonous — with EDA, feature engineering, four ML algorithms, confusion matrix, ROC & PR curves, and feature importance.

## Overview

This project trains and evaluates four classification models on the UCI Mushroom dataset (8,124 samples, 23 categorical features). All preprocessing and training run on the backend; the frontend displays metrics and diagnostic plots without a page reload.

## Models

| Model | Notes |
|-------|-------|
| Random Forest | 100 estimators |
| Gradient Boosting | 100 estimators |
| Logistic Regression | max_iter=1000 |
| Decision Tree | default depth |

## Features

- **EDA plots** — class balance, unique values per feature, stacked feature distributions, bivariate breakdown, Cramér's V correlation heatmap
- **Metrics** — accuracy, precision, recall, F1, AUC
- **Diagnostic plots** — confusion matrix, ROC curve, Precision-Recall curve, feature importance ranking
- **Model comparison** — side-by-side accuracy and F1 across all four models
- **Caching** — preprocessed data and trained models are cached in memory so re-running a model is instant

## Dataset

UCI Mushroom dataset: 8,124 mushroom samples with 22 categorical features (cap shape, gill color, habitat, etc.) and a binary target (edible / poisonous). No missing values after imputation.

## Tech Stack

- **Backend:** Flask, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy
- **Frontend:** Chart.js 3.9.1, Vanilla JavaScript

## Project Structure

```
mushroom_classification/
├── __init__.py                          # Flask blueprint, training pipeline, API routes
├── templates/
│   └── mushroom_classification/
│       └── index.html                   # Model selector, metrics display, plot panels
└── static/
    └── script.js                        # Fetch calls, Chart.js model comparison bar chart
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/mushroom-classification/` | Main page |
| GET | `/mushroom-classification/run?model=<key>&force=<bool>` | Train a model, return metrics + plots |
| GET | `/mushroom-classification/compare` | Train all models, return accuracy/F1 JSON |
| GET | `/mushroom-classification/plots` | Generate 5 EDA plots as base64 PNG |

**Model keys:** `random_forest`, `gradient_boosting`, `logistic_regression`, `decision_tree`

## Preprocessing

1. Mode imputation for missing values
2. Label encoding for all categorical features
3. 80/20 train/test split (stratified)
4. StandardScaler applied to encoded features
