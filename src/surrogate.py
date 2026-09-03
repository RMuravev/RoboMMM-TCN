# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Module 4: Surrogate Modeling.
HillTransformer — S-shaped Hill saturation: y = x^alpha / (x^alpha + K^alpha).
MaternHillKernel — custom Matern 3/2 kernel based on Hill-space distances.
GPRegressor — Gaussian Process regressor with analytical marginal log-likelihood.
"""

import math
import numpy as np


class HillTransformer:
    """ S-shaped Hill Saturation Transformation: y = x^alpha / (x^alpha + K^alpha) """
    def __init__(self, alpha=2.5, k_val=0.5):
        self.alpha = alpha
        self.k_val = k_val

    def transform(self, x):
        eps = 1e-8
        x_safe = np.clip(x, eps, None)
        num = np.power(x_safe, self.alpha)
        den = num + np.power(self.k_val, self.alpha)
        return num / den


class MaternHillKernel:
    """ Custom Matern 3/2 Gaussian Process Kernel based on Hill space metrics """
    def __init__(self, sigma_f=1.0, length_scale=0.5, alpha=2.5, k_val=0.5):
        self.sigma_f = sigma_f
        self.length_scale = length_scale
        self.hill = HillTransformer(alpha, k_val)

    def compute_distance(self, x1, x2):
        h1 = self.hill.transform(x1)
        h2 = self.hill.transform(x2)
        return np.linalg.norm(h1 - h2)

    def compute_covariance(self, x1, x2):
        d = self.compute_distance(x1, x2)
        const = math.sqrt(3.0) * d / self.length_scale
        return (self.sigma_f ** 2) * (1.0 + const) * math.exp(-const)


class GPRegressor:
    """ Gaussian Process Regressor with analytical Marginal Log-Likelihood (MLL) """
    def __init__(self, kernel, noise_var=1e-4):
        self.kernel = kernel
        self.noise_var = noise_var
        self.x_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        n = len(x_train)

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel.compute_covariance(self.x_train[i], self.x_train[j])
        K += self.noise_var * np.eye(n)
        self.K_inv = np.linalg.inv(K)

    def predict(self, x_new):
        n = len(self.x_train)
        k_star = np.zeros(n)
        for i in range(n):
            k_star[i] = self.kernel.compute_covariance(x_new, self.x_train[i])

        mu = np.dot(k_star.T, np.dot(self.K_inv, self.y_train))
        var = self.kernel.compute_covariance(x_new, x_new) - np.dot(k_star.T, np.dot(self.K_inv, k_star))
        return mu, max(1e-9, var)
