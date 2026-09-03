# RoboMMM-TCN v1.0 — Copyright (c) 2026 Roman Muravyev
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# For commercial use without AGPLv3 obligations, a commercial
# license is required. Contact: roman.muravev@gmail.com

# -*- coding: utf-8 -*-

"""
Module 7: High-Level Pipeline Execution (Simulation Engine).
CLI entry point: python run_pipeline.py --brand CardioRx --limit 100000000 --roi_barrier 1.20
"""

import argparse
import unittest

import numpy as np
import torch
import torch.optim as optim

from src.losses import StableTweedieLoss, AdaptiveMonotonicLoss
from src.models import TemporalConvolutionalNetwork
from src.interpretability import CausalDAG, TemporalAdstockSHAPExplainer
from src.surrogate import MaternHillKernel, GPRegressor
from src.optimization import BayesianOptimizer


def execute_robommm_simulation(brand="CardioRx", limit=100_000_000, roi_barrier=1.20):
    """ Simulates full production cycle of RoboMMM-TCN pipeline for double-checking """
    print(f"\n--- STARTING «RoboMMM-TCN» PIPELINE SIMULATION (brand={brand}) ---")

    # 1. Generate Synthetic Hierarchical Data
    np.random.seed(42)
    seq_len = 12
    num_features = 2
    x_synth = np.random.uniform(0.1, 1.0, (100, seq_len, num_features))
    y_synth = np.random.uniform(10.0, 100.0, (100, 1))

    inputs = torch.tensor(x_synth, dtype=torch.float32)
    targets = torch.tensor(y_synth, dtype=torch.float32)

    # 2. Build TCN and Monotonic loss with StableTweedie
    tcn_model = TemporalConvolutionalNetwork(num_inputs=num_features, num_channels=[4, 8])
    stable_tweedie = StableTweedieLoss(p=1.5, log_link=True)
    lagrangian_loss = AdaptiveMonotonicLoss(stable_tweedie, lmbda=20.0, mono_indices=[0])

    optimizer = optim.Adam(tcn_model.parameters(), lr=0.01)

    print("Training Monotonic TCN with Tweedie loss...")
    for epoch in range(5):
        optimizer.zero_grad()
        loss, base, penalty = lagrangian_loss(tcn_model, inputs, targets)
        loss.backward()
        optimizer.step()
        lagrangian_loss.step_annealing()
        print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f} "
              f"(Base: {base.item():.4f}, Penalty: {penalty.item():.4f}, "
              f"Lambda: {lagrangian_loss.lmbda:.2f})")

    # 3. Apply Temporal-Adstock SHAP
    print("Applying Causal-constrained Temporal-Adstock SHAP...")
    dag = CausalDAG(num_nodes=num_features)
    dag.add_edge(0, 1)

    shap_explainer = TemporalAdstockSHAPExplainer(
        tcn_model, background_data=x_synth[:10, -1, :], causal_dag=dag
    )
    test_instance = x_synth[0, -1, :]
    shap_vals = shap_explainer.compute_shap_values(test_instance)
    print(f"  Computed SHAP values: Feature 0: {shap_vals[0]:.4f}, Feature 1: {shap_vals[1]:.4f}")

    # 4. Bayesian Global Optimization with Matern-Hill Kernel
    print("Fitting GP Surrogate and running Expected Improvement Optimizer...")
    kernel = MaternHillKernel(sigma_f=10.0, length_scale=0.3, alpha=2.0, k_val=0.4)
    gp = GPRegressor(kernel=kernel, noise_var=1e-3)

    gp.fit([0.1, 0.3, 0.6, 0.8], [15.0, 45.0, 78.0, 82.0])
    bo_opt = BayesianOptimizer(gp, bounds=(0.0, 1.0))
    next_alloc = bo_opt.optimize_next_point(f_best=82.0)
    print(f"  Optimal allocation candidate: {next_alloc:.4f} o.e.")

    print(f"  Budget limit: {limit:,} RUB | ROI barrier: {roi_barrier}")
    print("--- SIMULATION COMPLETED SUCCESSFULLY ---\n")


def run_tests():
    """ Run all unit tests before simulation """
    from tests.test_robommm import TestRoboMMMSuite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRoboMMMSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoboMMM-TCN Pipeline Runner')
    parser.add_argument('--brand', type=str, default='CardioRx', help='Brand name')
    parser.add_argument('--limit', type=float, default=100_000_000, help='Total budget limit (RUB)')
    parser.add_argument('--roi_barrier', type=float, default=1.20, help='ROI barrier')
    parser.add_argument('--skip-tests', action='store_true', help='Skip unit tests')
    args = parser.parse_args()

    if not args.skip_tests:
        test_res = run_tests()
        if not test_res.wasSuccessful():
            print("Unit tests failed. Aborting pipeline.")
            exit(1)

    execute_robommm_simulation(
        brand=args.brand, limit=args.limit, roi_barrier=args.roi_barrier
    )
