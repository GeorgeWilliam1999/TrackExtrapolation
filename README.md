# TrackExtrapolation

Physics-Informed Neural Networks for Runge-Kutta Integration applied to particle track extrapolation and dynamical systems.

## Overview

This repository implements **Neural Runge-Kutta (NeuralRK)** models that learn to predict RK integration stages, enabling fast and accurate trajectory propagation. The project supports two main applications:

1. **Van der Pol (VDP) Oscillator** – A classic nonlinear dynamical system used for benchmarking
2. **LHCb Particle Tracking** – Track extrapolation through the LHCb detector magnetic field

## Features

- **Physics-Informed Training**: Loss functions that incorporate ODE constraints alongside data-fitting
- **Multiple RK Schemes**: Support for explicit and implicit Butcher tableaux (default: classical RK4)
- **Modular Architecture**: Separable components for magnetic fields, particle states, and simulators
- **Evaluation Suite**: Tools for comparing NeuralRK against classical RK4 integration

## Project Structure

```
TrackExtrapolation/
├── Classes/                        # Core physics/simulation modules
│   ├── magnetic_field.py           # Magnetic field implementations (LHCb, Quadratic)
│   ├── particle.py                 # Particle state representation
│   └── Simulators.py               # Classical RK4 simulators (dz propagation)
│
├── RK_PINNs/                        # Neural RK experiments
│   ├── RK_PINN.ipynb               # Main experimentation notebook
│   ├── RK_PINN_clean.ipynb         # Clean version of notebook
│   ├── VDP_RKPINN_single_step.py   # Single-step training script
│   ├── VDP_RKPINN_multi_step.py    # Multi-step training script
│   │
│   ├── modularisation/             # Utility modules for NeuralRK
│   │   ├── model_utils.py          # NeuralRK class, training loops, rollout functions
│   │   ├── vdp_utils.py            # Van der Pol system definition
│   │   ├── eval_utils.py           # Evaluation and plotting utilities
│   │   └── lhcb_utils.py           # LHCb magnetic field derivatives
│   │
│   ├── Data/                       # Training data storage
│   │   ├── VDP/                    # Van der Pol datasets
│   │   └── LHCb/                   # LHCb tracking datasets
│   │
│   └── Results/                    # Trained models and outputs
│       ├── VDP/Models/             # Saved VDP models (.pt files)
│       └── LHCb/                   # LHCb results
│
├── Results/                         # Top-level results directory
│   └── VDP/Models/                  # Alternative model save location
│
├── early_testing/                   # Sandbox and exploratory scripts
│   ├── TrackX_sandbox.ipynb        # Jupyter sandbox
│   ├── field_view.py               # Field visualization
│   └── Recorded_trajectories/      # JSON trajectory recordings
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)

### Environment Setup

Using Conda (recommended):

```bash
conda create -n TE python=3.10
conda activate TE
pip install -r requirements.txt
```

Or using pip directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Van der Pol Oscillator Example

```python
import torch
from RK_PINNs.modularisation.model_utils import NeuralRK, generate_training_data, train_model_single_step
from RK_PINNs.modularisation.vdp_utils import vdp

# Define RK4 Butcher tableau
butcher = {
    "A": [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]],
    "b": [1/6, 1/3, 1/3, 1/6],
    "c": [0, 0.5, 0.5, 1]
}

# Generate training data
y0 = torch.tensor([1.0, 0.0])
X, K = generate_training_data(vdp, y0, t0=0, t_end=30, dt=0.1, butcher=butcher, mu=1.0)

# Build and train model
model = NeuralRK(input_dim=2, hidden_dim=64, num_layers=3, output_dim=2, dt=0.1, butcher=butcher)
model = model.to("cuda")
X, K = X.to("cuda"), K.to("cuda")

train_model_single_step(model, X[:-1], K, min_epochs=500, patience=50)

# Save model
model.save_model("VDP", "RK4")
```

### 2. Rollout Comparison

```python
from RK_PINNs.modularisation.model_utils import rollout_neural_model, rollout_rk4

x0 = torch.tensor([1.0, 0.0], device="cuda")

# Neural RK trajectory
traj_nn = rollout_neural_model(model, x0, steps=300, dt=0.1)

# Classical RK4 trajectory
traj_rk = rollout_rk4(x0, steps=300, dt=0.1, m=10, butcher=butcher, f=lambda y: vdp(y, mu=1.0))

# Compare
import matplotlib.pyplot as plt
plt.plot(traj_rk[:, 0].cpu(), traj_rk[:, 1].cpu(), label="RK4")
plt.plot(traj_nn[:, 0].cpu(), traj_nn[:, 1].cpu(), "--", label="NeuralRK")
plt.legend()
plt.show()
```

### 3. LHCb Tracking Example

```python
from Classes.magnetic_field import LHCb_Field
from Classes.particle import particle_state
from Classes.Simulators import RK4_sim_dz

# Load magnetic field data
field = LHCb_Field("path/to/Bfield.rtf")

# Create particle
particle = particle_state(
    Ptype="muon",
    position=[0, 0, 0],
    tx=0.1,
    ty=0.05,
    momentum=[0, 0, 10e9],  # 10 GeV
    charge=-1
)

# Run simulation
sim = RK4_sim_dz([particle], field, dz=10, z=0, num_steps=100)
sim.run()
sim.plot_trajectory_with_lorentz_force()
```

## Key Components

### NeuralRK Model

The `NeuralRK` class (in `model_utils.py`) is an MLP that predicts RK stage derivatives:

```
Input: state x ∈ ℝ^d
Output: stages K ∈ ℝ^(s×d) where s = number of RK stages
```

The next state is computed as:
```
x_{n+1} = x_n + dt * Σ_i b_i * K_i
```

### Physics-Informed Loss

Training can use physics-aware losses that enforce:
- **Stage consistency**: K_i ≈ f(x + dt * Σ_j A_{ij} * K_j)
- **Step consistency**: x_{n+1} matches numerical integration
- **Jacobian regularization**: Smooth predictions across nearby states

### Magnetic Field Classes

- `LHCb_Field`: Interpolates real LHCb magnetic field data
- `Quadratic_Field`: Analytical test field with quadratic profile
- `MagneticField` (ABC): Base class for custom field implementations

## Training Tips

1. **Start with small networks**: `hidden_dim=32, num_layers=2` often suffices
2. **Use GPU**: Training is significantly faster on CUDA
3. **Monitor convergence**: Loss should decrease smoothly; spikes indicate learning rate issues
4. **Physics loss weight**: Start with `physics_weight=0.1` and tune based on task

## Model Naming Convention

Saved models follow this pattern:
```
NeuralRK_{SYSTEM}_hd{HIDDEN}_layers{LAYERS}_dt{DT}_{SCHEME}.pt
```

Example: `NeuralRK_VDP_hd64_layers3_dt0.1_RK4_PHYS.pt`

Schemes:
- `RK4`: Naive data-fitting loss
- `RK4_PHYS`: Physics-informed loss

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see LICENSE file for details.

## References

- Chen, R.T.Q. et al. "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Raissi, M. et al. "Physics-Informed Neural Networks" (JCP 2019)
- LHCb Collaboration - Track reconstruction methods
