# Audio Signal Prediction using Regression

This project models and predicts audio time-series data sampled at 8.82 kHz using linear regression and regularization techniques.

## Overview
- Built a Python pipeline to predict future audio samples from past values
- Evaluated single-step and multi-window predictors
- Applied Ridge and LASSO regression to control overfitting and perform feature selection

## Methods
- Linear Regression
- Ridge Regression (L2 regularization)
- LASSO Regression (L1 regularization)
- Train/test split and R² evaluation

## Results
- Achieved **95.7% test R²** using a 4-sample window predictor
- Ridge regularization improved generalization performance
- LASSO reduced a 9-sample model to **3 dominant coefficients**

## Files
- `predict.py` – main analysis script
- `figure1–4.pdf` – statistical and performance plots

## Tools
Python, NumPy, SciPy, scikit-learn, Matplotlib

## How to Run
```bash
pip install -r requirements.txt
python predict.py bach_8820hz.wav
