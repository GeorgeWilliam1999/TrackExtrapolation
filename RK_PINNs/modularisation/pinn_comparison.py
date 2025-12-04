# pinn_comparison.py
"""
Comprehensive Comparison Framework: PINN vs RK-PINN vs Large-RK

This module implements three neural network approaches for solving ODEs:

1. **Standard PINN**: Directly learns solution y(t) with physics residual loss
2. **RK-PINN**: Learns RK stage derivatives with physics-informed constraints  
3. **Large-RK Model**: Data-driven RK stage prediction (many collocation points)

Each can be compared against a high-precision reference solution.

References:
- Raissi et al. "Physics-Informed Neural Networks" (2019)
- RK-PINN papers for Runge-Kutta integration with neural networks
- LHCb tracking for particle physics application
"""

import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Configuration and Constants
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RK4_TABLEAU = {
    "A": torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]),
    "b": torch.tensor([1/6, 1/3, 1/3, 1/6]),
    "c": torch.tensor([0.0, 0.5, 0.5, 1.0]),
}


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50000
    min_epochs: int = 100
    patience: int = 50
    delta_tol: float = 1e-7
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    physics_weight: float = 0.1
    stage_weight: float = 1.0
    step_weight: float = 1.0
    jacobian_weight: float = 0.0
    log_interval: int = 100
    device: torch.device = DEVICE


@dataclass  
class ExperimentResults:
    """Container for experiment results."""
    method: str
    train_time: float
    final_loss: float
    convergence_epoch: int
    loss_history: List[float] = field(default_factory=list)
    trajectory_errors: Optional[torch.Tensor] = None
    final_error: Optional[float] = None
    rollout_time: Optional[float] = None
    mean_error: Optional[float] = None
    max_error: Optional[float] = None


# =============================================================================
# ODE Systems
# =============================================================================

def vdp_system(y: torch.Tensor, mu: float = 1.0) -> torch.Tensor:
    """
    Van der Pol oscillator system.
    
    dy1/dt = y2
    dy2/dt = mu * (1 - y1^2) * y2 - y1
    
    Args:
        y: State tensor of shape (..., 2)
        mu: Nonlinearity parameter
        
    Returns:
        Derivative tensor of same shape as y
    """
    y1, y2 = y[..., 0], y[..., 1]
    dy1 = y2
    dy2 = mu * (1 - y1**2) * y2 - y1
    return torch.stack([dy1, dy2], dim=-1)


def vdp_batched(y: torch.Tensor, mu: float = 1.0) -> torch.Tensor:
    """Vectorized VDP for batched inputs of shape (B, d) or (B, s, d)."""
    return vdp_system(y, mu)


# =============================================================================
# Ground Truth Generator (High-Precision RK)
# =============================================================================

class GroundTruthGenerator:
    """
    Generate high-precision reference solutions using RK4 with many substeps.
    
    This serves as the "true" solution for comparing different methods.
    """
    
    def __init__(
        self,
        f: Callable,
        butcher: Dict[str, torch.Tensor] = None,
        substeps: int = 1000,
        device: torch.device = DEVICE
    ):
        self.f = f
        self.butcher = butcher or RK4_TABLEAU
        self.substeps = substeps
        self.device = device
        
        # Move butcher to device
        self.A = self.butcher["A"].to(device)
        self.b = self.butcher["b"].to(device)
        self.c = self.butcher["c"].to(device)
        
    def _rk4_step(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """Single RK4 step."""
        k1 = self.f(x)
        k2 = self.f(x + 0.5 * dt * k1)
        k3 = self.f(x + 0.5 * dt * k2)
        k4 = self.f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def integrate(
        self,
        x0: torch.Tensor,
        t_end: float,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate from x0 to t_end with high precision.
        
        Args:
            x0: Initial state (d,) or (B, d)
            t_end: Final time
            dt: Output step size (actual integration uses dt/substeps)
            
        Returns:
            times: Time points (N+1,)
            trajectory: States at each time (N+1, d) or (B, N+1, d)
        """
        x0 = x0.to(self.device)
        has_batch = x0.dim() > 1
        if not has_batch:
            x0 = x0.unsqueeze(0)
            
        N = int(t_end / dt)
        sub_dt = dt / self.substeps
        
        B, d = x0.shape
        trajectory = torch.zeros(B, N + 1, d, device=self.device)
        trajectory[:, 0] = x0
        
        x = x0.clone()
        for n in range(N):
            for _ in range(self.substeps):
                x = self._rk4_step(x, sub_dt)
            trajectory[:, n + 1] = x
            
        times = torch.linspace(0, t_end, N + 1, device=self.device)
        
        if not has_batch:
            trajectory = trajectory.squeeze(0)
            
        return times, trajectory
    
    def get_rk_stages(
        self,
        x: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Compute true RK4 stages at a given state.
        
        Args:
            x: State tensor (B, d)
            dt: Step size
            
        Returns:
            K: Stage derivatives (B, 4, d)
        """
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        B, d = x.shape
        k = torch.zeros(B, 4, d, device=self.device)
        
        k[:, 0] = self.f(x)
        k[:, 1] = self.f(x + 0.5 * dt * k[:, 0])
        k[:, 2] = self.f(x + 0.5 * dt * k[:, 1])
        k[:, 3] = self.f(x + dt * k[:, 2])
        
        return k


# =============================================================================
# Model 1: Standard PINN
# =============================================================================

class StandardPINN(nn.Module):
    """
    Standard Physics-Informed Neural Network.
    
    Directly learns the solution y(t) and enforces the ODE as a soft constraint.
    
    Loss = L_data + λ * L_physics
    
    where L_physics = ||dy/dt - f(y)||² evaluated at collocation points.
    """
    
    def __init__(
        self,
        input_dim: int = 1,  # time dimension
        state_dim: int = 2,  # state dimension
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: str = "tanh"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build network
        act = nn.Tanh() if activation == "tanh" else nn.ReLU()
        
        layers = [nn.Linear(input_dim, hidden_dim), act]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, state_dim)]
        
        self.net = nn.Sequential(*layers)
        
        # Store ODE function
        self.f: Optional[Callable] = None
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map time to state.
        
        Args:
            t: Time tensor of shape (N,) or (N, 1)
            
        Returns:
            y: Predicted state (N, state_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(t)
    
    def compute_physics_residual(
        self,
        t: torch.Tensor,
        create_graph: bool = True
    ) -> torch.Tensor:
        """
        Compute physics residual: dy/dt - f(y).
        
        Args:
            t: Time points (N,) or (N, 1)
            create_graph: Whether to create graph for higher-order derivatives
            
        Returns:
            residual: Physics residual (N, state_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.requires_grad_(True)
        
        y = self.forward(t)
        
        # Compute dy/dt using autograd
        dy_dt = torch.zeros_like(y)
        for i in range(self.state_dim):
            grad_outputs = torch.ones_like(y[:, i])
            grads = torch.autograd.grad(
                y[:, i], t,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=True
            )[0]
            dy_dt[:, i] = grads.squeeze(-1)
        
        # Compute f(y)
        f_y = self.f(y)
        
        return dy_dt - f_y
    
    def loss_fn(
        self,
        t_data: torch.Tensor,
        y_data: torch.Tensor,
        t_colloc: torch.Tensor,
        physics_weight: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PINN loss.
        
        Args:
            t_data: Time points with known data (N_d, 1)
            y_data: Known states (N_d, state_dim)
            t_colloc: Collocation points for physics (N_c, 1)
            physics_weight: Weight for physics loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Data loss
        y_pred = self.forward(t_data)
        loss_data = F.mse_loss(y_pred, y_data)
        
        # Physics loss at collocation points
        residual = self.compute_physics_residual(t_colloc)
        loss_physics = torch.mean(residual ** 2)
        
        # Total loss
        total_loss = loss_data + physics_weight * loss_physics
        
        return total_loss, {
            "data": loss_data.item(),
            "physics": loss_physics.item(),
            "total": total_loss.item()
        }


# =============================================================================
# Model 2: RK-PINN (Physics-Informed)
# =============================================================================

class RKPINN(nn.Module):
    """
    Runge-Kutta Physics-Informed Neural Network.
    
    Predicts RK stage derivatives and enforces physics constraints:
    - Stage consistency: K_i = f(x + dt * sum_j A_ij * K_j)
    - Step consistency: x_{n+1} matches true next state
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 2,
        dt: float = 0.1,
        butcher: Dict = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dt = dt
        
        # Butcher tableau
        butcher = butcher or RK4_TABLEAU
        self.register_buffer("A", butcher["A"].float())
        self.register_buffer("b", butcher["b"].float())
        self.register_buffer("c", butcher["c"].float())
        self.s = len(self.b)
        
        # Build MLP
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, self.s * output_dim)]
        self.net = nn.Sequential(*layers)
        
        # ODE function
        self.f: Optional[Callable] = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict RK stages.
        
        Args:
            x: State tensor (B, input_dim)
            
        Returns:
            K: Stage derivatives (B, s, output_dim)
        """
        B = x.shape[0]
        out = self.net(x)
        return out.view(B, self.s, self.output_dim)
    
    def step(self, x: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        Perform one integration step.
        
        Args:
            x: Current state (B, d) or (d,)
            dt: Step size (uses self.dt if None)
            
        Returns:
            x_next: Next state
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        dt = dt or self.dt
        
        K = self.forward(x)  # (B, s, d)
        
        # x_next = x + dt * sum_i b_i * K_i
        b = self.b.view(1, -1, 1)  # (1, s, 1)
        x_next = x + dt * torch.sum(b * K, dim=1)
        
        return x_next.squeeze(0) if x_next.shape[0] == 1 else x_next
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        dt: float = None,
        config: TrainingConfig = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss components.
        
        Args:
            x: State tensor (B, d)
            dt: Step size
            config: Training configuration
            
        Returns:
            loss: Total physics loss
            loss_dict: Individual components
        """
        config = config or TrainingConfig()
        dt = dt or self.dt
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, d = x.shape
        
        K_pred = self.forward(x)  # (B, s, d)
        
        # Stage consistency loss: K_i should equal f(x + dt * sum_j A_ij * K_j)
        loss_stage = torch.tensor(0.0, device=x.device)
        for i in range(self.s):
            # Compute intermediate state for stage i
            A_row = self.A[i].view(1, -1, 1)  # (1, s, 1)
            weighted = A_row * K_pred  # (B, s, d)
            x_stage = x + dt * torch.sum(weighted, dim=1)  # (B, d)
            
            # True derivative at stage point
            k_true = self.f(x_stage)  # (B, d)
            
            # Stage consistency
            loss_stage = loss_stage + F.mse_loss(K_pred[:, i], k_true)
        loss_stage = loss_stage / self.s
        
        # Step consistency loss: predicted next state should match true RK step
        b = self.b.view(1, -1, 1)
        x_pred = x + dt * torch.sum(b * K_pred, dim=1)
        
        # Compute true next state with explicit RK4
        k1 = self.f(x)
        k2 = self.f(x + 0.5 * dt * k1)
        k3 = self.f(x + 0.5 * dt * k2)
        k4 = self.f(x + dt * k3)
        x_true = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        loss_step = F.mse_loss(x_pred, x_true)
        
        # Total physics loss
        loss = config.stage_weight * loss_stage + config.step_weight * loss_step
        
        return loss, {
            "stage": loss_stage.item(),
            "step": loss_step.item(),
            "physics_total": loss.item()
        }
    
    def loss_fn(
        self,
        x: torch.Tensor,
        K_true: torch.Tensor,
        config: TrainingConfig = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combined data + physics loss.
        
        Args:
            x: State tensor (B, d)
            K_true: True RK stages (B, s, d)
            config: Training configuration
            
        Returns:
            loss: Total loss
            loss_dict: Components
        """
        config = config or TrainingConfig()
        
        K_pred = self.forward(x)
        
        # Data loss
        loss_data = F.mse_loss(K_pred, K_true)
        
        # Physics loss
        loss_physics, phys_dict = self.compute_physics_loss(x, config=config)
        
        # Total
        total_loss = loss_data + config.physics_weight * loss_physics
        
        return total_loss, {
            "data": loss_data.item(),
            **phys_dict,
            "total": total_loss.item()
        }


# =============================================================================
# Model 3: Large RK Model (Data-Driven)
# =============================================================================

class LargeRKModel(nn.Module):
    """
    Large-scale data-driven RK model.
    
    Uses extensive training data from many collocation points but no explicit
    physics loss. Tests the hypothesis that sufficient data can replace
    physics constraints.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 5,
        output_dim: int = 2,
        dt: float = 0.1,
        butcher: Dict = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dt = dt
        
        # Butcher tableau
        butcher = butcher or RK4_TABLEAU
        self.register_buffer("A", butcher["A"].float())
        self.register_buffer("b", butcher["b"].float())
        self.s = len(self.b)
        
        # Larger network
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, self.s * output_dim)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict RK stages."""
        B = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.net(x)
        return out.view(B, self.s, self.output_dim)
    
    def step(self, x: torch.Tensor, dt: float = None) -> torch.Tensor:
        """Perform one integration step."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        dt = dt or self.dt
        
        K = self.forward(x)
        b = self.b.view(1, -1, 1)
        x_next = x + dt * torch.sum(b * K, dim=1)
        
        return x_next.squeeze(0) if squeeze else x_next
    
    def loss_fn(
        self,
        x: torch.Tensor,
        K_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Pure data-driven loss."""
        K_pred = self.forward(x)
        loss = F.mse_loss(K_pred, K_true)
        return loss, {"data": loss.item(), "total": loss.item()}


# =============================================================================
# Training Framework
# =============================================================================

class Trainer:
    """Unified training framework for all model types."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig = None
    ):
        self.model = model.to(config.device if config else DEVICE)
        self.config = config or TrainingConfig()
        self.optimizer = Adam(model.parameters(), lr=self.config.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor
        )
        self.loss_history = []
        
    def train_pinn(
        self,
        t_data: torch.Tensor,
        y_data: torch.Tensor,
        t_colloc: torch.Tensor,
        verbose: bool = True
    ) -> ExperimentResults:
        """Train standard PINN."""
        t_data = t_data.to(self.config.device)
        y_data = y_data.to(self.config.device)
        t_colloc = t_colloc.to(self.config.device)
        
        start_time = time.time()
        best_loss = float("inf")
        wait = 0
        
        pbar = tqdm(range(self.config.max_epochs), disable=not verbose)
        
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            loss, loss_dict = self.model.loss_fn(
                t_data, y_data, t_colloc,
                physics_weight=self.config.physics_weight
            )
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            self.loss_history.append(loss_dict["total"])
            
            if epoch % self.config.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.2e}",
                    "data": f"{loss_dict['data']:.2e}",
                    "phys": f"{loss_dict['physics']:.2e}"
                })
            
            # Convergence check
            if epoch >= self.config.min_epochs:
                if abs(loss_dict["total"] - best_loss) < self.config.delta_tol:
                    wait += 1
                    if wait >= self.config.patience:
                        break
                else:
                    best_loss = min(best_loss, loss_dict["total"])
                    wait = 0
                    
        train_time = time.time() - start_time
        
        return ExperimentResults(
            method="PINN",
            train_time=train_time,
            final_loss=self.loss_history[-1],
            convergence_epoch=epoch,
            loss_history=self.loss_history
        )
    
    def train_rk(
        self,
        X: torch.Tensor,
        K: torch.Tensor,
        use_physics: bool = True,
        verbose: bool = True
    ) -> ExperimentResults:
        """Train RK-PINN or Large RK model."""
        X = X.to(self.config.device)
        K = K.to(self.config.device)
        
        start_time = time.time()
        best_loss = float("inf")
        wait = 0
        
        N = X.shape[0]
        method = "RK-PINN" if use_physics else "Large-RK"
        
        pbar = tqdm(range(self.config.max_epochs), disable=not verbose)
        
        for epoch in pbar:
            # Sample batch
            idx = torch.randperm(N, device=self.config.device)[:self.config.batch_size]
            x_batch = X[idx]
            k_batch = K[idx]
            
            self.optimizer.zero_grad()
            
            if use_physics and hasattr(self.model, "compute_physics_loss"):
                loss, loss_dict = self.model.loss_fn(
                    x_batch, k_batch, config=self.config
                )
            else:
                loss, loss_dict = self.model.loss_fn(x_batch, k_batch)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            self.loss_history.append(loss_dict["total"])
            
            if epoch % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss_dict['total']:.2e}"})
            
            # Convergence check
            if epoch >= self.config.min_epochs:
                if abs(loss_dict["total"] - best_loss) < self.config.delta_tol:
                    wait += 1
                    if wait >= self.config.patience:
                        break
                else:
                    best_loss = min(best_loss, loss_dict["total"])
                    wait = 0
                    
        train_time = time.time() - start_time
        
        return ExperimentResults(
            method=method,
            train_time=train_time,
            final_loss=self.loss_history[-1],
            convergence_epoch=epoch,
            loss_history=self.loss_history
        )


# =============================================================================
# Evaluation Framework
# =============================================================================

class Evaluator:
    """Unified evaluation framework."""
    
    def __init__(
        self,
        ground_truth: GroundTruthGenerator,
        device: torch.device = DEVICE
    ):
        self.ground_truth = ground_truth
        self.device = device
        
    @torch.no_grad()
    def rollout_rk_model(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        steps: int,
        dt: float
    ) -> torch.Tensor:
        """Generate trajectory using RK model."""
        model.eval()
        x0 = x0.to(self.device)
        
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        trajectory = [x0]
        x = x0.clone()
        
        for _ in range(steps):
            x = model.step(x, dt)
            # Ensure consistent dimensionality
            if x.dim() == 1:
                x = x.unsqueeze(0)
            trajectory.append(x)
        
        # Stack trajectory: list of (B, d) -> (B, N+1, d)
        return torch.stack(trajectory, dim=1)
    
    @torch.no_grad()
    def rollout_pinn(
        self,
        model: StandardPINN,
        t_eval: torch.Tensor
    ) -> torch.Tensor:
        """Generate trajectory using standard PINN."""
        model.eval()
        t_eval = t_eval.to(self.device)
        return model(t_eval)
    
    def compute_errors(
        self,
        trajectory_pred: torch.Tensor,
        trajectory_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute various error metrics.
        
        Args:
            trajectory_pred: Predicted trajectory (N+1, d) or (B, N+1, d)
            trajectory_true: True trajectory (same shape)
            
        Returns:
            Dictionary of error metrics
        """
        # Ensure same device
        trajectory_true = trajectory_true.to(trajectory_pred.device)
        
        # Pointwise errors
        errors = torch.norm(trajectory_pred - trajectory_true, dim=-1)
        
        # Final time error
        final_error = errors[..., -1].mean().item()
        
        # Mean trajectory error
        mean_error = errors.mean().item()
        
        # Max error
        max_error = errors.max().item()
        
        # Relative error
        norm_true = torch.norm(trajectory_true, dim=-1)
        rel_errors = errors / (norm_true + 1e-10)
        mean_rel_error = rel_errors.mean().item()
        
        return {
            "final_error": final_error,
            "mean_error": mean_error,
            "max_error": max_error,
            "mean_rel_error": mean_rel_error
        }
    
    def time_rollout(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        steps: int,
        dt: float,
        num_runs: int = 10,
        model_type: str = "rk"
    ) -> float:
        """Time the rollout process."""
        model.eval()
        x0 = x0.to(self.device)
        t_end = steps * dt
        
        # Warmup
        if model_type == "rk":
            _ = self.rollout_rk_model(model, x0, steps, dt)
        else:
            t_eval = torch.linspace(0, t_end, steps + 1, device=self.device)
            _ = self.rollout_pinn(model, t_eval)
        
        # Time
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start = time.time()
        
        for _ in range(num_runs):
            if model_type == "rk":
                _ = self.rollout_rk_model(model, x0, steps, dt)
            else:
                t_eval = torch.linspace(0, t_end, steps + 1, device=self.device)
                _ = self.rollout_pinn(model, t_eval)
            
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        total_time = time.time() - start
        
        return total_time / num_runs
    
    def full_evaluation(
        self,
        model: nn.Module,
        x0_set: torch.Tensor,
        t_end: float,
        dt: float,
        model_type: str = "rk"
    ) -> ExperimentResults:
        """
        Complete evaluation of a model.
        
        Args:
            model: Trained model
            x0_set: Set of initial conditions (N_ic, d)
            t_end: Final time
            dt: Step size
            model_type: "rk" or "pinn"
            
        Returns:
            ExperimentResults with all metrics
        """
        steps = int(t_end / dt)
        
        all_errors = []
        
        for i in range(x0_set.shape[0]):
            x0 = x0_set[i]
            
            # Ground truth
            _, traj_true = self.ground_truth.integrate(x0, t_end, dt)
            
            # Model prediction
            if model_type == "rk":
                traj_pred = self.rollout_rk_model(model, x0, steps, dt)
                traj_pred = traj_pred.squeeze(0)
            else:
                t_eval = torch.linspace(0, t_end, steps + 1, device=self.device)
                traj_pred = self.rollout_pinn(model, t_eval)
            
            errors = self.compute_errors(traj_pred, traj_true)
            all_errors.append(errors)
        
        # Aggregate
        mean_final = np.mean([e["final_error"] for e in all_errors])
        mean_mean = np.mean([e["mean_error"] for e in all_errors])
        max_max = np.max([e["max_error"] for e in all_errors])
        
        # Timing - pass model_type to use correct rollout method
        rollout_time = self.time_rollout(model, x0_set[0], steps, dt, model_type=model_type)
        
        return ExperimentResults(
            method=model_type,
            train_time=0,
            final_loss=0,
            convergence_epoch=0,
            final_error=mean_final,
            mean_error=mean_mean,
            max_error=max_max,
            rollout_time=rollout_time
        )


# =============================================================================
# Model Save/Load Functions
# =============================================================================

def save_model(
    model: nn.Module,
    path: str,
    model_type: str,
    config: dict = None,
    training_info: dict = None
) -> None:
    """
    Save a trained model with metadata for reproducibility.
    
    Args:
        model: Trained model (StandardPINN, RKPINN, or LargeRKModel)
        path: File path to save to (should end in .pt)
        model_type: One of 'pinn', 'rk_pinn', 'large_rk'
        config: Configuration dictionary used for training
        training_info: Training results (epochs, loss history, etc.)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "model_config": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
        },
        "config": config,
        "training_info": training_info,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    # Add model-specific attributes
    if hasattr(model, "dt"):
        save_dict["model_config"]["dt"] = model.dt
    if hasattr(model, "input_dim"):
        save_dict["model_config"]["input_dim"] = model.input_dim
    if hasattr(model, "output_dim"):
        save_dict["model_config"]["output_dim"] = model.output_dim
    if hasattr(model, "state_dim"):
        save_dict["model_config"]["state_dim"] = model.state_dim
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(
    path: str,
    device: torch.device = DEVICE
) -> Tuple[nn.Module, dict]:
    """
    Load a saved model.
    
    Args:
        path: Path to the saved model file
        device: Device to load the model onto
        
    Returns:
        model: Loaded model
        metadata: Dictionary with config, training_info, etc.
    """
    checkpoint = torch.load(path, map_location=device)
    
    model_type = checkpoint["model_type"]
    config = checkpoint["model_config"]
    
    if model_type == "pinn":
        model = StandardPINN(
            input_dim=config.get("input_dim", 1),
            state_dim=config.get("state_dim", 2),
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )
    elif model_type == "rk_pinn":
        model = RKPINN(
            input_dim=config.get("input_dim", 2),
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            output_dim=config.get("output_dim", 2),
            dt=config.get("dt", 0.1),
        )
    elif model_type == "large_rk":
        model = LargeRKModel(
            input_dim=config.get("input_dim", 2),
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            output_dim=config.get("output_dim", 2),
            dt=config.get("dt", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    metadata = {
        "config": checkpoint.get("config"),
        "training_info": checkpoint.get("training_info"),
        "timestamp": checkpoint.get("timestamp"),
    }
    
    print(f"Model loaded from {path}")
    return model, metadata


def get_model_save_path(
    base_dir: str,
    model_type: str,
    system: str,
    hidden_dim: int,
    num_layers: int,
    dt: float = None,
    suffix: str = ""
) -> str:
    """
    Generate a standardized save path for a model.
    
    Args:
        base_dir: Base directory (e.g., 'Results/VDP/Models')
        model_type: 'pinn', 'rk_pinn', or 'large_rk'
        system: 'VDP' or 'LHCb'
        hidden_dim: Network hidden dimension
        num_layers: Number of layers
        dt: Time step (for RK models)
        suffix: Optional suffix for the filename
        
    Returns:
        Full path to save the model
    """
    os.makedirs(base_dir, exist_ok=True)
    
    parts = [model_type, system, f"hd{hidden_dim}", f"layers{num_layers}"]
    if dt is not None:
        parts.append(f"dt{dt}")
    if suffix:
        parts.append(suffix)
    
    filename = "_".join(parts) + ".pt"
    return os.path.join(base_dir, filename)


def save_experiment_results(
    results_dict: dict,
    save_dir: str,
    experiment_name: str = "comparison"
) -> None:
    """
    Save comprehensive experiment results including models, metrics, and plots.
    
    Args:
        results_dict: Dictionary with 'models', 'results', 'config' keys
        save_dir: Directory to save all artifacts
        experiment_name: Name prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    models_dir = os.path.join(save_dir, "Models")
    plots_dir = os.path.join(save_dir, "Plots")
    data_dir = os.path.join(save_dir, "Data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save models
    if "models" in results_dict:
        for name, model in results_dict["models"].items():
            model_type = "pinn" if isinstance(model, StandardPINN) else \
                        "rk_pinn" if isinstance(model, RKPINN) else "large_rk"
            path = os.path.join(models_dir, f"{name}_{timestamp}.pt")
            
            training_info = None
            if "results" in results_dict and name in results_dict["results"]:
                res = results_dict["results"][name]
                training_info = {
                    "train_time": res.train_time,
                    "final_loss": res.final_loss,
                    "convergence_epoch": res.convergence_epoch,
                    "mean_error": res.mean_error,
                    "max_error": res.max_error,
                    "final_error": res.final_error,
                }
            
            save_model(model, path, model_type, 
                      config=results_dict.get("config"),
                      training_info=training_info)
    
    # Save results to CSV
    if "results" in results_dict:
        summary_data = []
        for name, result in results_dict["results"].items():
            summary_data.append({
                "Method": name,
                "Train_Time_s": result.train_time,
                "Final_Loss": result.final_loss,
                "Epochs": result.convergence_epoch,
                "Mean_Error": result.mean_error,
                "Max_Error": result.max_error,
                "Final_Error": result.final_error,
                "Rollout_Time_ms": result.rollout_time * 1000 if result.rollout_time else None,
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(data_dir, f"{experiment_name}_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    
    # Save config
    if "config" in results_dict:
        config_path = os.path.join(data_dir, f"{experiment_name}_config_{timestamp}.json")
        import json
        with open(config_path, "w") as f:
            # Convert non-serializable items
            config_serializable = {}
            for k, v in results_dict["config"].items():
                if isinstance(v, torch.device):
                    config_serializable[k] = str(v)
                elif hasattr(v, "item"):
                    config_serializable[k] = v.item()
                else:
                    config_serializable[k] = v
            json.dump(config_serializable, f, indent=2)
        print(f"Config saved to {config_path}")
    
    print(f"\nAll artifacts saved to {save_dir}")


# =============================================================================
# Data Generation
# =============================================================================

def generate_vdp_training_data(
    f: Callable,
    x0_list: List[torch.Tensor],
    t_end: float,
    dt: float,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for VDP system.
    
    Returns:
        X: State points (N_total, d)
        K: RK stages at each point (N_total, s, d)
    """
    gt = GroundTruthGenerator(f, substeps=100, device=device)
    
    X_list = []
    K_list = []
    
    for x0 in x0_list:
        x0 = x0.to(device)
        _, trajectory = gt.integrate(x0, t_end, dt)
        
        # Get states (exclude last point which has no next state)
        X_list.append(trajectory[:-1])
        
        # Get RK stages for each state
        K = gt.get_rk_stages(trajectory[:-1], dt)
        K_list.append(K)
    
    X = torch.cat(X_list, dim=0)
    K = torch.cat(K_list, dim=0)
    
    return X, K


def generate_collocation_points(
    t_start: float,
    t_end: float,
    n_points: int,
    method: str = "uniform",
    device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Generate collocation points for PINN training.
    
    Args:
        t_start, t_end: Time bounds
        n_points: Number of points
        method: "uniform" or "lhs" (Latin Hypercube)
        device: Torch device
        
    Returns:
        Collocation points tensor
    """
    if method == "uniform":
        return torch.linspace(t_start, t_end, n_points, device=device)
    elif method == "lhs":
        # Simple LHS implementation
        samples = (torch.rand(n_points, device=device) + 
                   torch.arange(n_points, device=device)) / n_points
        samples = samples[torch.randperm(n_points)]
        return t_start + samples * (t_end - t_start)
    else:
        return torch.rand(n_points, device=device) * (t_end - t_start) + t_start


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(
    results: List[ExperimentResults],
    x0: torch.Tensor,
    t_end: float,
    dt: float,
    models: Dict[str, nn.Module],
    ground_truth: GroundTruthGenerator,
    save_path: Optional[str] = None
):
    """
    Create comprehensive comparison plots.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Get true trajectory
    _, traj_true = ground_truth.integrate(x0, t_end, dt)
    traj_true = traj_true.cpu().numpy()
    
    steps = int(t_end / dt)
    t = np.linspace(0, t_end, steps + 1)
    
    # 1. Phase space comparison
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(traj_true[:, 0], traj_true[:, 1], 'k-', lw=2, label='Ground Truth')
    
    colors = ['b', 'r', 'g']
    for (name, model), color in zip(models.items(), colors):
        model.eval()
        if hasattr(model, 'step'):
            evaluator = Evaluator(ground_truth)
            traj = evaluator.rollout_rk_model(model, x0, steps, dt)
            traj = traj.squeeze(0).cpu().numpy()
        else:
            t_eval = torch.linspace(0, t_end, steps + 1, device=x0.device)
            traj = model(t_eval).detach().cpu().numpy()
        ax1.plot(traj[:, 0], traj[:, 1], f'{color}--', lw=1.5, label=name)
    
    ax1.set_xlabel('$y_1$')
    ax1.set_ylabel('$y_2$')
    ax1.set_title('Phase Space')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Time series
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, traj_true[:, 0], 'k-', lw=2, label='$y_1$ (True)')
    ax2.plot(t, traj_true[:, 1], 'k--', lw=2, label='$y_2$ (True)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State')
    ax2.set_title('Time Series (Ground Truth)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Training convergence
    ax3 = fig.add_subplot(2, 3, 3)
    for result, color in zip(results, colors):
        ax3.semilogy(result.loss_history, color, label=result.method)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Convergence')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Error over time
    ax4 = fig.add_subplot(2, 3, 4)
    for (name, model), color in zip(models.items(), colors):
        model.eval()
        if hasattr(model, 'step'):
            evaluator = Evaluator(ground_truth)
            traj = evaluator.rollout_rk_model(model, x0, steps, dt)
            traj = traj.squeeze(0).cpu().numpy()
        else:
            t_eval = torch.linspace(0, t_end, steps + 1, device=x0.device)
            traj = model(t_eval).detach().cpu().numpy()
        
        error = np.linalg.norm(traj - traj_true, axis=1)
        ax4.semilogy(t, error, color, label=name)
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Error')
    ax4.set_title('Error Over Time')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Bar chart of metrics
    ax5 = fig.add_subplot(2, 3, 5)
    x_pos = np.arange(len(results))
    width = 0.25
    
    ax5.bar(x_pos - width, [r.mean_error or 0 for r in results], width, label='Mean Error')
    ax5.bar(x_pos, [r.max_error or 0 for r in results], width, label='Max Error')
    ax5.bar(x_pos + width, [r.final_error or 0 for r in results], width, label='Final Error')
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([r.method for r in results])
    ax5.set_ylabel('Error')
    ax5.set_title('Error Comparison')
    ax5.legend()
    ax5.grid(True, axis='y')
    
    # 6. Timing comparison
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(
        [r.method for r in results],
        [r.train_time for r in results],
        color=colors[:len(results)]
    )
    ax6.set_ylabel('Training Time (s)')
    ax6.set_title('Training Time Comparison')
    ax6.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# Main Comparison Function
# =============================================================================

def run_full_comparison(
    f: Callable = vdp_system,
    t_end: float = 30.0,
    dt: float = 0.1,
    n_initial_conditions: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 3,
    config: TrainingConfig = None,
    save_dir: str = None,
    verbose: bool = True
) -> Tuple[Dict[str, ExperimentResults], Dict[str, nn.Module]]:
    """
    Run complete comparison between PINN, RK-PINN, and Large-RK.
    
    Args:
        f: ODE right-hand side function
        t_end: Final integration time
        dt: Time step
        n_initial_conditions: Number of ICs for training
        hidden_dim: Network hidden dimension
        num_layers: Number of layers
        config: Training configuration
        save_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        results: Dictionary of ExperimentResults
        models: Dictionary of trained models
    """
    config = config or TrainingConfig()
    device = config.device
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("PINN vs RK-PINN vs Large-RK Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"t_end: {t_end}, dt: {dt}")
    print(f"Architecture: {hidden_dim} hidden, {num_layers} layers")
    print()
    
    # Generate initial conditions
    torch.manual_seed(42)
    x0_list = [
        torch.tensor([2 * torch.rand(1).item(), 2 * torch.rand(1).item()], device=device)
        for _ in range(n_initial_conditions)
    ]
    
    # Ground truth generator
    gt = GroundTruthGenerator(f, substeps=1000, device=device)
    
    # Generate training data
    print("Generating training data...")
    X_train, K_train = generate_vdp_training_data(f, x0_list, t_end, dt, device)
    print(f"  States: {X_train.shape}, Stages: {K_train.shape}")
    
    # Generate PINN training data
    steps = int(t_end / dt)
    _, traj_pinn = gt.integrate(x0_list[0], t_end, dt)
    t_data = torch.linspace(0, t_end, steps + 1, device=device)
    t_colloc = generate_collocation_points(0, t_end, 1000, method="uniform", device=device)
    
    results = {}
    models = {}
    
    # =========================================================================
    # Train Standard PINN
    # =========================================================================
    if verbose:
        print("\n" + "=" * 40)
        print("Training Standard PINN")
        print("=" * 40)
    
    pinn = StandardPINN(
        input_dim=1,
        state_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    pinn.f = f
    
    trainer_pinn = Trainer(pinn, config)
    result_pinn = trainer_pinn.train_pinn(
        t_data.unsqueeze(-1), traj_pinn, t_colloc, verbose=verbose
    )
    
    # Evaluate
    evaluator = Evaluator(gt, device)
    x0_eval = torch.stack(x0_list[:5])
    eval_result = evaluator.full_evaluation(pinn, x0_eval, t_end, dt, model_type="pinn")
    result_pinn.mean_error = eval_result.mean_error
    result_pinn.max_error = eval_result.max_error
    result_pinn.final_error = eval_result.final_error
    result_pinn.rollout_time = eval_result.rollout_time
    
    results["PINN"] = result_pinn
    models["PINN"] = pinn
    
    # =========================================================================
    # Train RK-PINN (Physics-Informed)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 40)
        print("Training RK-PINN (Physics-Informed)")
        print("=" * 40)
    
    rk_pinn = RKPINN(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=2,
        dt=dt
    ).to(device)
    rk_pinn.f = f
    
    trainer_rk_pinn = Trainer(rk_pinn, config)
    result_rk_pinn = trainer_rk_pinn.train_rk(
        X_train, K_train, use_physics=True, verbose=verbose
    )
    
    eval_result = evaluator.full_evaluation(rk_pinn, x0_eval, t_end, dt, model_type="rk")
    result_rk_pinn.mean_error = eval_result.mean_error
    result_rk_pinn.max_error = eval_result.max_error
    result_rk_pinn.final_error = eval_result.final_error
    result_rk_pinn.rollout_time = eval_result.rollout_time
    
    results["RK-PINN"] = result_rk_pinn
    models["RK-PINN"] = rk_pinn
    
    # =========================================================================
    # Train Large-RK (Data-Driven)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 40)
        print("Training Large-RK (Data-Driven)")
        print("=" * 40)
    
    large_rk = LargeRKModel(
        input_dim=2,
        hidden_dim=hidden_dim * 2,  # Larger network
        num_layers=num_layers + 2,
        output_dim=2,
        dt=dt
    ).to(device)
    
    trainer_large_rk = Trainer(large_rk, config)
    result_large_rk = trainer_large_rk.train_rk(
        X_train, K_train, use_physics=False, verbose=verbose
    )
    
    eval_result = evaluator.full_evaluation(large_rk, x0_eval, t_end, dt, model_type="rk")
    result_large_rk.mean_error = eval_result.mean_error
    result_large_rk.max_error = eval_result.max_error
    result_large_rk.final_error = eval_result.final_error
    result_large_rk.rollout_time = eval_result.rollout_time
    
    results["Large-RK"] = result_large_rk
    models["Large-RK"] = large_rk
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            "Method": name,
            "Train Time (s)": f"{result.train_time:.2f}",
            "Final Loss": f"{result.final_loss:.2e}",
            "Epochs": result.convergence_epoch,
            "Mean Error": f"{result.mean_error:.2e}" if result.mean_error else "N/A",
            "Max Error": f"{result.max_error:.2e}" if result.max_error else "N/A",
            "Rollout (ms)": f"{result.rollout_time*1000:.2f}" if result.rollout_time else "N/A"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    if save_dir:
        df.to_csv(os.path.join(save_dir, "comparison_results.csv"), index=False)
    
    # Plot
    plot_comparison(
        list(results.values()),
        x0_list[0],
        t_end,
        dt,
        models,
        gt,
        save_path=os.path.join(save_dir, "comparison_plot.png") if save_dir else None
    )
    
    return results, models


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Quick test
    config = TrainingConfig(
        max_epochs=5000,
        patience=30,
        physics_weight=0.1
    )
    
    results, models = run_full_comparison(
        t_end=10.0,
        dt=0.1,
        n_initial_conditions=5,
        hidden_dim=32,
        num_layers=3,
        config=config,
        save_dir="Results/VDP/Comparison"
    )
