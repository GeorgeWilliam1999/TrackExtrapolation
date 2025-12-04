# Contributing to TrackExtrapolation

Thank you for your interest in contributing to TrackExtrapolation! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-capable GPU (recommended for development)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/TrackExtrapolation.git
   cd TrackExtrapolation
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n TE python=3.10
   conda activate TE
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
TrackExtrapolation/
├── Classes/                    # Core physics modules (canonical source)
│   ├── magnetic_field.py       # Magnetic field implementations
│   ├── particle.py             # Particle state class
│   └── Simulators.py           # Classical RK4 simulators
│
├── RK_PINNs/                   # Neural RK experiments
│   ├── modularisation/         # Utility modules
│   └── ...
```

### Important Notes

- **`Classes/`** is the canonical location for core physics modules
- **`RK_PINNs/modularisation/`** contains re-exports from `Classes/` for backward compatibility
- Do not duplicate code between these directories

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Maximum line length: 100 characters

### Docstrings

Use NumPy-style docstrings for all public functions and classes:

```python
def my_function(param1, param2):
    """
    Brief description.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    return_type
        Description of return value.

    Examples
    --------
    >>> my_function(1, 2)
    3
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
def compute_loss(
    model: NeuralRK,
    x: torch.Tensor,
    k_true: torch.Tensor
) -> torch.Tensor:
    ...
```

## Git Workflow

### Branches

- `main`: Stable release branch
- `dev`: Development branch
- Feature branches: `feature/description`
- Bug fixes: `fix/description`

### Commits

Write clear, descriptive commit messages:

```
[module] Brief description

- Detailed change 1
- Detailed change 2
```

Examples:
```
[model_utils] Add physics-informed loss function

- Implement stage consistency loss
- Add Jacobian regularization option
- Update training loop to use new loss
```

### Pull Requests

1. Create a feature branch from `dev`
2. Make your changes with clear commits
3. Update documentation if needed
4. Submit PR to `dev` branch
5. Request review from maintainers

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model_utils.py

# Run with coverage
pytest --cov=Classes --cov=RK_PINNs
```

### Writing Tests

Place tests in the `tests/` directory with filenames matching `test_*.py`.

```python
import pytest
import torch
from Classes.magnetic_field import Quadratic_Field

def test_quadratic_field_shape():
    field = Quadratic_Field(B0=1.0)
    B = field.magnetic_field(0, 0, 100)
    assert B.shape == (3,)

def test_quadratic_field_values():
    field = Quadratic_Field(B0=1.0)
    B = field.magnetic_field(0, 0, 0)
    assert B[2] == 0  # Bz should be 0 at z=0
```

## Model Conventions

### Naming

Saved models follow this pattern:
```
NeuralRK_{SYSTEM}_hd{HIDDEN}_layers{LAYERS}_dt{DT}_{SCHEME}.pt
```

- `SYSTEM`: `VDP` or `LHCb`
- `HIDDEN`: Hidden dimension (e.g., 64)
- `LAYERS`: Number of layers (e.g., 3)
- `DT`: Time step (e.g., 0.1)
- `SCHEME`: `RK4` (naive) or `RK4_PHYS` (physics-informed)

### Checkpoints

Model checkpoints should contain:
```python
{
    "model_state_dict": model.state_dict(),
    "model_info": {
        "input_dim": int,
        "hidden_dim": int,
        "output_dim": int,
        "num_layers": int,
        "dt": float,
        "s": int  # number of RK stages
    }
}
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- PyTorch version
- CUDA version (if applicable)
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (if any)

## Questions?

Open a discussion on GitHub or contact the maintainers.

---

Thank you for contributing!
