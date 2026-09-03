# RoboMMM-TCN v1.0

**Software Complex for Robust Optimization and Factor Analysis of Hierarchical Marketing Budgets Based on Deep Learning with Monotonic Constraints**

Industrial-grade Python/PyTorch library for causal inference, monotonic deep learning, Temporal-Adstock SHAP explanations, and Bayesian surrogate budget optimization under ZBB (Zero-Based Budgeting) constraints.

## Features

- **StableTweedieLoss** — numerically stable Tweedie deviance in log-space for compound Poisson-Gamma distributions
- **AdaptiveMonotonicLoss** — Lagrangian penalty enforcing monotonicity in Sobolev spaces with exponential annealing
- **TemporalConvolutionalNetwork** — causal dilated TCN with weight normalization, GELU, and spatial dropout
- **TemporalAdstockSHAPExplainer** — causal-constrained Shapley values with Markov blanket DAG restrictions
- **MaternHillKernel** — custom Matern 3/2 Gaussian Process kernel with Hill-saturation distance metrics
- **ZBBBudgetCoordinator** — subgradient dual decomposition for hierarchical budget allocation under ROI barriers

## Quick Start

### Requirements

- Python 3.12+
- PyTorch 2.2+
- CUDA 12.1+ (for GPU acceleration, optional)
- MS SQL Server 2019+ (for production data source, optional)

### Installation

```bash
git clone https://github.com/yourusername/RoboMMM-TCN.git
cd RoboMMM-TCN
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt


## License

**RoboMMM-TCN** is distributed under a dual licensing model:

### GNU AGPLv3 (free, open-source)

- ✅ Free use, modification, and redistribution
- ✅ Use in SaaS — but with the obligation to disclose source code of modifications
- ✅ Suitable for research, education, and open-source projects

> ⚠️ **Important:** AGPLv3 requires that any modified code made available
> over a network must be published under the same license.

### Commercial License (paid)

- ✅ Use in proprietary products without disclosing source code
- ✅ Integration into closed SaaS services and internal systems
- ✅ SaaS without the obligation to publish modifications
- ✅ Priority technical support

To obtain a commercial license, contact the author:

- **E-mail:** roman.muravev@gmail.com
- **Phone:** +7 (926) 661-30-29

Commercial license agreements are executed in accordance with
Article 1286 of the Civil Code of the Russian Federation, with possible
state registration in Rospatent.

Full text of AGPLv3: <https://www.gnu.org/licenses/agpl-3.0.txt>
