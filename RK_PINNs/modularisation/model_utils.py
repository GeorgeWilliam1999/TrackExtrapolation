import os
import time
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# ----------------------------
# Module: NeuralRK Integrator
# ----------------------------
# This module provides utilities for generating training data via Runge-Kutta integration,
# defines a neural network to predict RK stages, and supports rollouts using both classical
# and learned integrators.


def generate_training_data(
    func: Callable[..., torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    t_end: float,
    dt: float,
    butcher: Dict[str, List[List[float]]],
    mu: float = 1.0,
    M: int = 10,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training pairs (states X and RK stage outputs K) by integrating a dynamical
    system from t0 to t_end with step dt using a provided Butcher tableau.

    Args:
        func: Right-hand side function of the ODE. Signature should accept a state tensor
              and return its time derivative.
        y0: Initial state tensor of shape (d,).
        t0: Initial time.
        t_end: Final integration time.
        dt: Time step for recording states.
        butcher: Butcher tableau dict with keys 'A', 'b', 'c'.
        mu: System parameter passed to func.
        M: Number of substeps per dt.
        device: Torch device for computation.

    Returns:
        X: Tensor of recorded states of shape (N, d).
        K: Tensor of RK stages of shape (N, s, d).
    """
    if butcher is None:
        raise ValueError("Butcher tableau must be provided.")

    N = int((t_end - t0) / dt) + 1
    x = y0.to(device)
    d = x.numel()
    s = len(butcher['b'])

    X = torch.zeros((N, d), dtype=torch.float32, device=device)
    K = torch.zeros((N, s, d), dtype=torch.float32, device=device)

    for n in range(N):
        X[n] = x
        k, x_next = rk_apply(butcher, x, dt / M, lambda y: func(y, mu=mu))
        for _ in range(M - 1):
            k, x_next = rk_apply(butcher, x_next, dt / M, lambda y: func(y, mu=mu))
        K[n] = k
        x = x_next

    return X, K


def generate_training_data_all_ics(
    func: Callable[..., torch.Tensor],
    y0s: List[torch.Tensor],
    t0: float,
    t_end: float,
    dt: float,
    butcher: Dict[str, List[List[float]]],
    mu: float = 1.0,
    M: int = 10,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for multiple initial conditions by concatenating outputs.

    Args:
        func: ODE right-hand side function.
        y0s: List of initial state tensors.
        t0: Initial time.
        t_end: Final integration time.
        dt: Time step for recording.
        butcher: Butcher tableau.
        mu: System parameter.
        M: Number of substeps per dt.
        device: Torch device.

    Returns:
        X_all: Concatenated state tensor of shape (N_total, d).
        K_all: Concatenated stage tensor of shape (N_total, s, d).
    """
    data = []
    for y0 in y0s:
        X, K = generate_training_data(func, y0, t0, t_end, dt, butcher, mu=mu, M=M, device=device)
        data.append((X, K))

    X_all = torch.cat([pair[0] for pair in data], dim=0)
    K_all = torch.cat([pair[1] for pair in data], dim=0)
    return X_all, K_all

class NeuralRK(nn.Module):
    """
    Neural network to predict Runge-Kutta stage derivatives for integration.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 2,
        dt: float = 0.01,
        butcher: Optional[Dict[str, List[List[float]]]] = None
    ) -> None:
        """
        Initialize the NeuralRK model.

        Args:
            input_dim: Dimension of input state.
            hidden_dim: Number of hidden units per layer.
            num_layers: Number of hidden layers.
            output_dim: Dimension of predicted derivative for each stage.
            dt: Integration time step.
            butcher: Butcher tableau dict; defaults to classical RK4 if None.
        """
        super().__init__()
        if butcher is None:
            butcher = {
                'A': [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ],
                'b': [1/6, 1/3, 1/3, 1/6],
                'c': [0.0, 0.5, 0.5, 1.0]
            }
        self.butcher = butcher
        self.s = len(butcher['b'])
        self.dt = dt
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build MLP
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, self.s * output_dim)]
        self.net = nn.Sequential(*layers)

    def name_model(self, system_name: str, scheme: str) -> str:
        """
        Construct a standardized filename for saving the model.

        Args:
            system_name: Identifier for the dynamical system ('LHCb' or 'VDP').
            scheme: Integration scheme label.

        Returns:
            Full path to the target file (without saving).
        """
        if system_name == "LHCb":
            save_dir = os.path.join("Results", "LHCb", "Models")
        elif system_name == "VDP":
            save_dir = os.path.join("Results", "VDP", "Models")
        else:
            raise ValueError(f"Unknown system_name: {system_name}")

        os.makedirs(save_dir, exist_ok=True)
        filename = f"NeuralRK_{system_name}_hd{self.hidden_dim}_layers{self.num_layers}_dt{self.dt}_{scheme}.pt"
        return os.path.join(save_dir, filename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map input states to RK stage predictions.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, s, output_dim).
        """
        batch_size = x.shape[0]
        out = self.net(x)
        return out.view(batch_size, self.s, self.output_dim)

    def loss_fn(self, x: torch.Tensor, k_true: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-squared error loss between predicted and true RK stages.

        Args:
            x: Input state tensor.
            k_true: True stage derivatives.

        Returns:
            Scalar loss.
        """
        k_pred = self.forward(x)
        return F.mse_loss(k_pred, k_true)

    def save_model(self, system_name: str, scheme: str) -> None:
        """
        Save model parameters and configuration to disk.

        Args:
            system_name: System identifier.
            scheme: Integration scheme label.
        """
        path = self.name_model(system_name, scheme)
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_info": {
                "input_dim": self.net[0].in_features,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "dt": self.dt,
                "s": self.s
            }
        }, path)
        print(f"Model saved to {path}")

    def does_model_exist(self, system_name: str, scheme: str) -> bool:
        """
        Check if a saved model file exists on disk.

        Args:
            system_name: System identifier.
            scheme: Scheme label.

        Returns:
            True if file exists, False otherwise.
        """
        path = self.name_model(system_name, scheme)
        exists = os.path.exists(path)
        print(f"Checked path: {path}")
        return exists


def select_model(
    system_name: str,
    scheme: str,
    hidden_dim: int,
    num_layers: int,
    dt: float
) -> Dict[str, Any]:
    """
    Load a saved NeuralRK model from disk given its configuration.

    Args:
        system_name: Identifier for the dynamical system.
        scheme: Integration scheme label.
        hidden_dim: Hidden dimension used when training.
        num_layers: Number of layers used when training.
        dt: Integration step used when training.

    Returns:
        Dictionary containing model_state_dict and model_info.
    """
    save_dir = os.path.join("Results", system_name, "Models")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"NeuralRK_{system_name}_hd{hidden_dim}_layers{num_layers}_dt{dt}_{scheme}.pt"
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} does not exist.")
    return torch.load(path)


def rk_apply(
    butcher: Dict[str, List[List[float]]],
    x: torch.Tensor,
    dt: float,
    f: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int = 10,
    tol: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single Runge-Kutta integration step for implicit tableau via fixed-point iteration.

    Args:
        butcher: Butcher tableau dict with 'A', 'b'.
        x: Current state tensor.
        dt: Time increment.
        f: Function computing state derivative.
        max_iter: Max iterations for stage solver.
        tol: Convergence tolerance.

    Returns:
        k: Stage derivatives tensor of shape (s, d).
        x_next: Next state tensor of shape (d,).
    """
    A = torch.tensor(butcher['A'], dtype=x.dtype, device=x.device)
    b = torch.tensor(butcher['b'], dtype=x.dtype, device=x.device)
    s = len(b)
    d = x.numel()
    k = torch.zeros((s, d), dtype=x.dtype, device=x.device)

    for i in range(s):
        k_i = k[i].clone()

        def G(ki_guess: torch.Tensor) -> torch.Tensor:
            weighted_sum = sum(
                A[i, j] * (ki_guess if j == i else k[j])
                for j in range(s)
            )
            return f(x + dt * weighted_sum)

        for _ in range(max_iter):
            ki_new = G(k_i)
            if torch.norm(ki_new - k_i) < tol:
                break
            k_i = ki_new

        k[i] = k_i

    x_next = x + dt * torch.sum(b.view(-1, 1) * k, dim=0)
    return k, x_next

@torch.no_grad()
def rollout_neural_model(
    model: NeuralRK,
    x0: torch.Tensor,
    steps: int,
    dt: float
) -> torch.Tensor:
    """
    Generate trajectory using a trained NeuralRK model.

    Args:
        model: Trained NeuralRK instance.
        x0: Initial state tensor of shape (d,).
        steps: Number of integration steps.
        dt: Time step size.

    Returns:
        Tensor of shape (steps+1, d) containing the trajectory.
    """
    model.eval()
    x = x0.unsqueeze(0)
    trajectory = [x.squeeze(0).cpu()]

    for _ in range(steps):
        k_pred = model(x)
        b = torch.tensor(model.butcher['b'], dtype=k_pred.dtype, device=k_pred.device).view(1, -1, 1)
        x = x + dt * torch.sum(b * k_pred, dim=1)
        trajectory.append(x.squeeze(0).cpu())

    return torch.stack(trajectory)


@torch.no_grad()
def rollout_rk4(
    x0: torch.Tensor,
    steps: int,
    dt: float,
    m: int,
    butcher: Dict[str, List[List[float]]],
    f: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    Generate trajectory via classical RK integration.

    Args:
        x0: Initial state tensor.
        steps: Number of steps to simulate.
        dt: Total time step per recorded step.
        m: Substeps per dt.
        butcher: Butcher tableau.
        f: ODE rhs function.

    Returns:
        Trajectory tensor of shape (steps+1, d).
    """
    x = x0.clone()
    trajectory = [x.cpu()]
    for _ in range(steps):
        k, x_next = rk_apply(butcher, x, dt / m, f)
        for _ in range(m - 1):
            k, x_next = rk_apply(butcher, x_next, dt / m, f)
        x = x_next
        trajectory.append(x.cpu())
    return torch.stack(trajectory)
