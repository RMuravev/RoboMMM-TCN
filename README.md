# RoboMMM-TCN v1.0

**Software Complex for Robust Optimization and Factor Analysis of Hierarchical Marketing Budgets Based on Deep Learning with Monotonic Constraints**

## Overview

Industrial-grade Python/PyTorch library for causal inference, monotonic deep learning, Temporal-Adstock SHAP explanations, and Bayesian surrogate budget optimization under ZBB (Zero-Based Budgeting) constraints.

## Installation

### Prerequisites

* **Python 3.12+**
* **CUDA 12.1+** (optional for GPU acceleration)
* **MS SQL Server 2019+** (optional for production data source)

### Installation Steps

```bash
git clone https://github.com/yourusername/RoboMMM-TCN.git
cd RoboMMM-TCN
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start
### Running the Pipeline
```bash
python run_pipeline.py --brand CardioRx --limit 100000000 --roi_barrier 1.20
```
### Running Tests
```bash
python -m pytest tests/test_robommm.py -v
```

## Project Structure
```text
RoboMMM-TCN/
├── src/
│   ├── __init__.py
│   ├── losses.py
│   ├── models.py
│   ├── interpretability.py
│   ├── surrogate.py
│   └── optimization.py
├── tests/
│   ├── __init__.py
│   └── test_robommm.py
├── run_pipeline.py
├── config.yaml
├── requirements.txt
├── .gitignore
├── README.md
├── LICENSE
├── LICENSE-AGPLv3.txt
└── COMMERCIAL_LICENSE.md
```

## Key Features
### Core Functionalities
* **StableTweedieLoss** — numerically stable Tweedie deviance in log-space
* **AdaptiveMonotonicLoss** — Lagrangian penalty for monotonicity
* **Temporal Convolutional Network (TCN)** — causal dilated convolutions
* **Temporal-Adstock SHAP** — causal-constrained explanations
* **Bayesian Optimization** — with Matern-Hill kernel
* **ZBB Budget Coordinator** — hierarchical budget allocation

## Usage Examples
### Basic Usage
```python
from src.models import TemporalConvolutionalNetwork
from src.losses import StableTweedieLoss

# Initialize model
model = TemporalConvolutionalNetwork(num_inputs=2, num_channels=[4, 8])

# Initialize loss
loss_fn = StableTweedieLoss(p=1.5, log_link=True)

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model(inputs)
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

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
