#!/usr/bin/env python3
"""
predict.py
ECE 2720 — Programming Exercise 3
Instructor: Prof. Jayadev Acharya
Boubakar Diallo bd453

Usage:
    ./predict.py path/to/bach_8820hz.wav
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# helpers

def load_audio(path):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)
    return sr, data

def make_bins_from_data(data, bin_size=10):
    dmin = np.min(data)
    dmax = np.max(data)
    bins = np.arange(dmin, dmax + bin_size, bin_size)
    return bins

def split_train_test(x):
    half = len(x) // 2
    return x[:half], x[half:]


# 1- figure1.pdf

def make_figure1(x):
    subsampled = x[::30]
    bins = make_bins_from_data(subsampled, bin_size=10)
    plt.figure()
    plt.hist(subsampled, bins=bins, edgecolor='black')
    plt.title("Histogram of every 30th sample")
    plt.xlabel("Sample value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figure1.pdf")
    plt.close()

# 2- Gaussian fit
# figure2.pdf

def make_figure2(x):
    subsampled = x[::30]
    mu = np.mean(subsampled)
    var = np.var(subsampled)  
    sigma = np.sqrt(var)

    bins = make_bins_from_data(subsampled, bin_size=10)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # histogram 
    counts, _ = np.histogram(subsampled, bins=bins)
    bin_width = bins[1] - bins[0]

    # Gaussian 
    gauss = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
    gauss_scaled = gauss * len(subsampled) * bin_width

    plt.figure()
    plt.bar(centers, counts, width=bin_width, alpha=0.6, label="Data histogram")
    plt.plot(centers, gauss_scaled, linewidth=2, label="Gaussian N(μ̂, σ̂²)")
    plt.title(f"Gaussian fit: mu={mu:.2f}, var={var:.2f}")
    plt.xlabel("Sample value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure2.pdf")
    plt.close()
    return mu, var


# 3- step regression: 

def single_step_regression(train, test):
    # train
    Xtrain = train[:-1].reshape(-1, 1)
    Ytrain = train[1:]
    reg = LinearRegression().fit(Xtrain, Ytrain)
    a_star = reg.coef_[0]
    b_star = reg.intercept_
    R2_train = reg.score(Xtrain, Ytrain)

    # test
    Xtest = test[:-1].reshape(-1, 1)
    Ytest = test[1:]
    R2_test = reg.score(Xtest, Ytest)

    return a_star, b_star, R2_train, R2_test

#regression with window size d

def make_window_dataset(x, d):
    N = len(x)
    X = np.zeros((N - d, d))
    Y = np.zeros(N - d)
    for i in range(N - d):
        X[i, :] = x[i:i + d]
        Y[i] = x[i + d]
    return X, Y

# 7. figure3.pdf

def make_figure3(train, test, dmin=2, dmax=20):
    d_values = list(range(dmin, dmax + 1))
    R2_train_list = []
    R2_test_list = []

    for d in d_values:
        Xtrain, Ytrain = make_window_dataset(train, d)
        Xtest, Ytest = make_window_dataset(test, d)
        reg = LinearRegression().fit(Xtrain, Ytrain)
        R2_train_list.append(reg.score(Xtrain, Ytrain))
        R2_test_list.append(reg.score(Xtest, Ytest))

    plt.figure()
    plt.plot(d_values, R2_train_list, marker='o', label="R² train")
    plt.plot(d_values, R2_test_list, marker='s', label="R² test")
    plt.xlabel("Window size d")
    plt.ylabel("R²")
    plt.title("R² vs window size (2–20)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure3.pdf")
    plt.close()

    best_d = d_values[int(np.argmax(R2_test_list))]
    best_R2_test = np.max(R2_test_list)
    return best_d, best_R2_test, d_values, R2_train_list, R2_test_list


# 9. figure4.pdf

def make_figure4(train, test, d=20):
    Xtrain, Ytrain = make_window_dataset(train, d)
    Xtest, Ytest = make_window_dataset(test, d)
    lambdas = np.logspace(7, 20, num=14, base=10.0)
    R2_train = []
    R2_test = []
    for lam in lambdas:
        reg = Ridge(alpha=lam, fit_intercept=True)
        reg.fit(Xtrain, Ytrain)
        R2_train.append(reg.score(Xtrain, Ytrain))
        R2_test.append(reg.score(Xtest, Ytest))

    plt.figure()
    plt.semilogx(lambdas, R2_train, marker='o', label="R² train")
    plt.semilogx(lambdas, R2_test, marker='s', label="R² test")
    plt.xlabel("λ (log scale)")
    plt.ylabel("R²")
    plt.title("Ridge regression, d=20")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure4.pdf")
    plt.close()

    no_overfit_lambda = None
    for lam, rtr, rte in zip(lambdas, R2_train, R2_test):
        if rtr - rte < 0.001: 
            no_overfit_lambda = lam
            break

    # lambda
    idx_best = int(np.argmax(R2_test))
    best_lambda = lambdas[idx_best]

    return lambdas, R2_train, R2_test, no_overfit_lambda, best_lambda


def run_lasso(train, test, d=9, lam=5e11):
    Xtrain, Ytrain = make_window_dataset(train, d)
    alpha = lam / (2.0 * len(train))
    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    lasso.fit(Xtrain, Ytrain)

    coeffs = lasso.coef_ 
    intercept = lasso.intercept_

    zero_mask = np.isclose(coeffs, 0.0, atol=1e-8)

    nonzero_indices = [i for i, z in enumerate(zero_mask) if not z]
    return coeffs, intercept, zero_mask, nonzero_indices


# main

def main():
    if len(sys.argv) < 2:
        print("Usage: ./predict.py path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        sys.exit(1)

    sr, data = load_audio(wav_path)
    make_figure1(data)
    mu, var = make_figure2(data)
    train, test = split_train_test(data)
    a_star, b_star, R2tr_1, R2te_1 = single_step_regression(train, test)
    best_d, best_R2_test, d_vals, R2tr_list, R2te_list = make_figure3(train, test)
    lambdas, R2tr_ridge, R2te_ridge, no_overfit_lambda, best_lambda = make_figure4(train, test, d=20)
    coeffs, intercept, zero_mask, nonzero_indices = run_lasso(train, test, d=9, lam=5e11)

    # print all the answers
    print("=== ECE 2720 PE3 Outputs ===")
    print(f"Sampling rate: {sr} Hz")
    print(f"Gaussian estimates: mu_hat={mu:.4f}, var_hat={var:.4f}")
    print()
    print("Single-step regression (predict x[n+1] from x[n]):")
    print(f"  a* = {a_star}")
    print(f"  b* = {b_star}")
    print(f"  R²_train = {R2tr_1}")
    print(f"  R²_test  = {R2te_1}")
    print()
    print("Multi-window regression (2..20):")
    print(f"  Best test R² = {best_R2_test} at window size d = {best_d}")
    print()
    print("Ridge (d=20):")
    print("  lambdas used:", lambdas)
    print("  R²_train:", R2tr_ridge)
    print("  R²_test :", R2te_ridge)
    print(f"  Smallest lambda where overfitting seems controlled (heuristic): {no_overfit_lambda}")
    print(f"  Lambda that maximizes test R²: {best_lambda}")
    print()
    print("LASSO (d=9, λ=5e11):")
    print("  Coefficients:", coeffs)
    print("  Intercept:", intercept)
    print("  Zero mask (True means coeff ~ 0):", zero_mask)
    print(f"  Nonzero coefficient indices (0-based, most recent sample is index 0): {nonzero_indices}")
    print()
    print("PDFs written: figure1.pdf, figure2.pdf, figure3.pdf, figure4.pdf")

if __name__ == "__main__":
    main()
