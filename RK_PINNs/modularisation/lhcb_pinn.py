# lhcb_pinn.py
"""
LHCb Track Extrapolation with Physics-Informed Neural Networks.

This module implements PINN, RK-PINN, and Large-RK approaches for the
6-dimensional LHCb track state propagation through the magnetic field.

State vector: [x, y, z, tx, ty, q/p]
- (x, y, z): Position in cm
- (tx, ty): Direction tangents dx/dz, dy/dz  
- q/p: Charge over momentum (c/GeV)

The equations of motion derive from the Lorentz force with z as the
independent variable (beam direction).

References:
- LHCb tracking reconstruction (Jan van Tilburg thesis)
- Standard LHCb RK4 extrapolation methods
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
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Classes.magnetic_field import LHCb_Field, Quadratic_Field


# =============================================================================
# Constants
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speed of light in appropriate units for q/p conversion
C_LIGHT = 2.99792458e8  # m/s

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
class LHCbTrainingConfig:
    """Configuration for LHCb model training."""
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50000
    min_epochs: int = 200
    patience: int = 50
    delta_tol: float = 1e-8
    scheduler_patience: int = 30
    scheduler_factor: float = 0.5
    physics_weight: float = 0.1
    stage_weight: float = 1.0
    step_weight: float = 1.0
    log_interval: int = 100
    device: torch.device = DEVICE
    
    # LHCb specific
    dz: float = 10.0  # cm, step size in z
    z_start: float = 0.0
    z_end: float = 1000.0  # cm


# =============================================================================
# LHCb ODE System
# =============================================================================

class LHCbODE:
    """
    LHCb track propagation ODE.
    
    Implements the equations of motion for a charged particle in the
    LHCb magnetic field using z as the independent variable.
    
    State: [x, y, z, tx, ty, q_over_p]
    
    Equations:
        dx/dz = tx
        dy/dz = ty
        dz/dz = 1 (trivial, included for completeness)
        dtx/dz = (q/p) * γ * [ty*(tx*Bx + Bz) - (1 + tx²)*By]
        dty/dz = -(q/p) * γ * [tx*(ty*By + Bz) - (1 + ty²)*Bx]
        d(q/p)/dz = 0 (momentum conserved)
        
    where γ = sqrt(1 + tx² + ty²)
    """
    
    def __init__(self, field: Union[LHCb_Field, Quadratic_Field]):
        """
        Initialize with magnetic field.
        
        Args:
            field: Magnetic field object with interpolated_field(x, y, z) method
        """
        self.field = field
        
    def get_field(self, x: float, y: float, z: float) -> np.ndarray:
        """Get magnetic field at position."""
        return self.field.interpolated_field(x, y, z)
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivatives.
        
        Args:
            state: State tensor of shape (..., 6)
                   [x, y, z, tx, ty, q_over_p]
                   
        Returns:
            dstate/dz of same shape
        """
        # Handle batched inputs
        original_shape = state.shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        B = state.shape[0]
        device = state.device
        
        # Extract components
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        tx = state[:, 3]
        ty = state[:, 4]
        q_p = state[:, 5]
        
        # Get magnetic field for each position
        # Field interpolation is numpy-based, so we need to loop
        B_field = torch.zeros(B, 3, device=device)
        for i in range(B):
            B_vec = self.get_field(
                x[i].item(),
                y[i].item(),
                z[i].item()
            )
            B_field[i] = torch.tensor(B_vec, device=device, dtype=state.dtype)
        
        Bx, By, Bz = B_field[:, 0], B_field[:, 1], B_field[:, 2]
        
        # Kinematic factor
        gamma = torch.sqrt(1 + tx**2 + ty**2)
        
        # Derivatives
        dx_dz = tx
        dy_dz = ty
        dz_dz = torch.ones_like(tx)
        
        dtx_dz = q_p * gamma * (ty * (tx * Bx + Bz) - (1 + tx**2) * By)
        dty_dz = -q_p * gamma * (tx * (ty * By + Bz) - (1 + ty**2) * Bx)
        dqp_dz = torch.zeros_like(q_p)
        
        dstate = torch.stack([dx_dz, dy_dz, dz_dz, dtx_dz, dty_dz, dqp_dz], dim=-1)
        
        if len(original_shape) == 1:
            dstate = dstate.squeeze(0)
            
        return dstate
    
    def numpy_call(self, state: np.ndarray) -> np.ndarray:
        """NumPy version for faster field evaluation."""
        x, y, z, tx, ty, q_p = state
        
        B = self.get_field(x, y, z)
        Bx, By, Bz = B[0], B[1], B[2]
        
        gamma = np.sqrt(1 + tx**2 + ty**2)
        
        dx_dz = tx
        dy_dz = ty
        dz_dz = 1.0
        dtx_dz = q_p * gamma * (ty * (tx * Bx + Bz) - (1 + tx**2) * By)
        dty_dz = -q_p * gamma * (tx * (ty * By + Bz) - (1 + ty**2) * Bx)
        dqp_dz = 0.0
        
        return np.array([dx_dz, dy_dz, dz_dz, dtx_dz, dty_dz, dqp_dz])


# =============================================================================
# Ground Truth Generator for LHCb
# =============================================================================

class LHCbGroundTruth:
    """
    High-precision RK4 integrator for LHCb tracks.
    """
    
    def __init__(
        self,
        ode: LHCbODE,
        substeps: int = 100,
        device: torch.device = DEVICE
    ):
        self.ode = ode
        self.substeps = substeps
        self.device = device
        
    def _rk4_step_numpy(self, state: np.ndarray, dz: float) -> np.ndarray:
        """Single RK4 step using numpy (faster for field evaluation)."""
        k1 = self.ode.numpy_call(state)
        k2 = self.ode.numpy_call(state + 0.5 * dz * k1)
        k3 = self.ode.numpy_call(state + 0.5 * dz * k2)
        k4 = self.ode.numpy_call(state + dz * k3)
        return state + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def integrate(
        self,
        state0: torch.Tensor,
        z_end: float,
        dz: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate track from initial state to z_end.
        
        Args:
            state0: Initial state [x, y, z, tx, ty, q/p]
            z_end: Final z position
            dz: Output step size (integration uses dz/substeps)
            
        Returns:
            z_points: Z positions
            trajectory: States at each z
        """
        state = state0.cpu().numpy() if isinstance(state0, torch.Tensor) else state0
        z0 = state[2]
        
        N = int((z_end - z0) / dz)
        sub_dz = dz / self.substeps
        
        trajectory = [state.copy()]
        
        for n in range(N):
            for _ in range(self.substeps):
                state = self._rk4_step_numpy(state, sub_dz)
            trajectory.append(state.copy())
        
        trajectory = np.stack(trajectory)
        z_points = np.array([t[2] for t in trajectory])
        
        return (
            torch.tensor(z_points, device=self.device),
            torch.tensor(trajectory, device=self.device, dtype=torch.float32)
        )
    
    def get_rk_stages(
        self,
        state: torch.Tensor,
        dz: float
    ) -> torch.Tensor:
        """Compute true RK4 stages at given state."""
        state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        
        if state_np.ndim == 1:
            state_np = state_np[np.newaxis, :]
        
        B = state_np.shape[0]
        K = np.zeros((B, 4, 6))
        
        for i in range(B):
            s = state_np[i]
            k1 = self.ode.numpy_call(s)
            k2 = self.ode.numpy_call(s + 0.5 * dz * k1)
            k3 = self.ode.numpy_call(s + 0.5 * dz * k2)
            k4 = self.ode.numpy_call(s + dz * k3)
            K[i, 0] = k1
            K[i, 1] = k2
            K[i, 2] = k3
            K[i, 3] = k4
        
        return torch.tensor(K, device=self.device, dtype=torch.float32)


# =============================================================================
# LHCb RK-PINN Model
# =============================================================================

class LHCbRKPINN(nn.Module):
    """
    RK-PINN for LHCb track extrapolation.
    
    Predicts RK4 stage derivatives for the 6D state with physics constraints.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dz: float = 10.0,
        butcher: Dict = None,
        field: Union[LHCb_Field, Quadratic_Field] = None
    ):
        super().__init__()
        
        self.input_dim = 6
        self.output_dim = 6  # Full state derivatives
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dz = dz
        
        # Butcher tableau
        butcher = butcher or RK4_TABLEAU
        self.register_buffer("A", butcher["A"].float())
        self.register_buffer("b", butcher["b"].float())
        self.s = 4
        
        # Build network
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, self.s * self.output_dim)]
        self.net = nn.Sequential(*layers)
        
        # ODE system
        self.ode = LHCbODE(field) if field else None
        
    def set_field(self, field):
        """Set or update the magnetic field."""
        self.ode = LHCbODE(field)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict RK stages.
        
        Args:
            state: State tensor (B, 6)
            
        Returns:
            K: Stage derivatives (B, 4, 6)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]
        out = self.net(state)
        return out.view(B, self.s, self.output_dim)
    
    def step(self, state: torch.Tensor, dz: float = None) -> torch.Tensor:
        """Perform one integration step."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        dz = dz or self.dz
        K = self.forward(state)
        
        b = self.b.view(1, -1, 1)
        state_next = state + dz * torch.sum(b * K, dim=1)
        
        return state_next.squeeze(0) if squeeze else state_next
    
    def compute_physics_loss(
        self,
        state: torch.Tensor,
        dz: float = None,
        config: LHCbTrainingConfig = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss for LHCb.
        
        Enforces:
        1. Stage consistency: K_i ≈ f(state + dz * sum_j A_ij * K_j)
        2. Step consistency: state_next matches true RK4 step
        """
        if self.ode is None:
            raise ValueError("Must set field before computing physics loss")
            
        config = config or LHCbTrainingConfig()
        dz = dz or self.dz
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        K_pred = self.forward(state)
        
        # Stage consistency
        loss_stage = torch.tensor(0.0, device=state.device)
        for i in range(self.s):
            A_row = self.A[i].view(1, -1, 1)
            state_stage = state + dz * torch.sum(A_row * K_pred, dim=1)
            k_true = self.ode(state_stage)
            loss_stage = loss_stage + F.mse_loss(K_pred[:, i], k_true)
        loss_stage = loss_stage / self.s
        
        # Step consistency (compare with true RK4)
        b = self.b.view(1, -1, 1)
        state_pred = state + dz * torch.sum(b * K_pred, dim=1)
        
        # True next state
        k1 = self.ode(state)
        k2 = self.ode(state + 0.5 * dz * k1)
        k3 = self.ode(state + 0.5 * dz * k2)
        k4 = self.ode(state + dz * k3)
        state_true = state + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        loss_step = F.mse_loss(state_pred, state_true)
        
        loss = config.stage_weight * loss_stage + config.step_weight * loss_step
        
        return loss, {
            "stage": loss_stage.item(),
            "step": loss_step.item(),
            "physics_total": loss.item()
        }
    
    def loss_fn(
        self,
        state: torch.Tensor,
        K_true: torch.Tensor,
        config: LHCbTrainingConfig = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combined data + physics loss."""
        config = config or LHCbTrainingConfig()
        
        K_pred = self.forward(state)
        loss_data = F.mse_loss(K_pred, K_true)
        
        if self.ode is not None:
            loss_phys, phys_dict = self.compute_physics_loss(state, config=config)
            total = loss_data + config.physics_weight * loss_phys
            return total, {"data": loss_data.item(), **phys_dict, "total": total.item()}
        else:
            return loss_data, {"data": loss_data.item(), "total": loss_data.item()}


# =============================================================================
# Large RK Model for LHCb
# =============================================================================

class LHCbLargeRK(nn.Module):
    """
    Large data-driven RK model for LHCb (no physics loss).
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dz: float = 10.0,
        butcher: Dict = None
    ):
        super().__init__()
        
        self.input_dim = 6
        self.output_dim = 6
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dz = dz
        
        butcher = butcher or RK4_TABLEAU
        self.register_buffer("b", butcher["b"].float())
        self.s = 4
        
        # Larger network
        layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, self.s * self.output_dim)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]
        out = self.net(state)
        return out.view(B, self.s, self.output_dim)
    
    def step(self, state: torch.Tensor, dz: float = None) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        dz = dz or self.dz
        K = self.forward(state)
        b = self.b.view(1, -1, 1)
        state_next = state + dz * torch.sum(b * K, dim=1)
        
        return state_next.squeeze(0) if squeeze else state_next
    
    def loss_fn(
        self,
        state: torch.Tensor,
        K_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        K_pred = self.forward(state)
        loss = F.mse_loss(K_pred, K_true)
        return loss, {"data": loss.item(), "total": loss.item()}


# =============================================================================
# Training and Evaluation
# =============================================================================

class LHCbTrainer:
    """Training framework for LHCb models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: LHCbTrainingConfig = None
    ):
        self.model = model.to(config.device if config else DEVICE)
        self.config = config or LHCbTrainingConfig()
        self.optimizer = Adam(model.parameters(), lr=self.config.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor
        )
        self.loss_history = []
        
    def train(
        self,
        X: torch.Tensor,
        K: torch.Tensor,
        use_physics: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Train the model."""
        X = X.to(self.config.device)
        K = K.to(self.config.device)
        
        start_time = time.time()
        best_loss = float("inf")
        wait = 0
        N = X.shape[0]
        
        pbar = tqdm(range(self.config.max_epochs), disable=not verbose)
        
        for epoch in pbar:
            idx = torch.randperm(N, device=self.config.device)[:self.config.batch_size]
            x_batch = X[idx]
            k_batch = K[idx]
            
            self.optimizer.zero_grad()
            
            if use_physics and hasattr(self.model, 'compute_physics_loss'):
                loss, loss_dict = self.model.loss_fn(x_batch, k_batch, self.config)
            else:
                loss, loss_dict = self.model.loss_fn(x_batch, k_batch)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            self.loss_history.append(loss_dict["total"])
            
            if epoch % self.config.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss_dict['total']:.2e}"})
            
            if epoch >= self.config.min_epochs:
                if abs(loss_dict["total"] - best_loss) < self.config.delta_tol:
                    wait += 1
                    if wait >= self.config.patience:
                        break
                else:
                    best_loss = min(best_loss, loss_dict["total"])
                    wait = 0
        
        return {
            "train_time": time.time() - start_time,
            "final_loss": self.loss_history[-1],
            "epochs": epoch,
            "loss_history": self.loss_history
        }


class LHCbEvaluator:
    """Evaluation framework for LHCb models."""
    
    def __init__(
        self,
        ground_truth: LHCbGroundTruth,
        device: torch.device = DEVICE
    ):
        self.gt = ground_truth
        self.device = device
        
    @torch.no_grad()
    def rollout(
        self,
        model: nn.Module,
        state0: torch.Tensor,
        z_end: float,
        dz: float
    ) -> torch.Tensor:
        """Generate trajectory using model."""
        model.eval()
        state = state0.to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        z0 = state[0, 2].item()
        N = int((z_end - z0) / dz)
        
        trajectory = [state.clone()]
        for _ in range(N):
            state = model.step(state, dz)
            trajectory.append(state.clone())
            
        return torch.cat([t.unsqueeze(1) for t in trajectory], dim=1)
    
    def compute_errors(
        self,
        traj_pred: torch.Tensor,
        traj_true: torch.Tensor
    ) -> Dict[str, float]:
        """Compute error metrics."""
        # Position errors (x, y, z)
        pos_pred = traj_pred[..., :3]
        pos_true = traj_true[..., :3]
        pos_error = torch.norm(pos_pred - pos_true, dim=-1)
        
        # Slope errors (tx, ty)
        slope_pred = traj_pred[..., 3:5]
        slope_true = traj_true[..., 3:5]
        slope_error = torch.norm(slope_pred - slope_true, dim=-1)
        
        # Full state error
        full_error = torch.norm(traj_pred - traj_true, dim=-1)
        
        return {
            "pos_final": pos_error[..., -1].mean().item(),
            "pos_mean": pos_error.mean().item(),
            "pos_max": pos_error.max().item(),
            "slope_final": slope_error[..., -1].mean().item(),
            "slope_mean": slope_error.mean().item(),
            "full_mean": full_error.mean().item(),
            "full_max": full_error.max().item()
        }
    
    def full_evaluation(
        self,
        model: nn.Module,
        state0_set: torch.Tensor,
        z_end: float,
        dz: float
    ) -> Dict:
        """Full evaluation over multiple initial states."""
        all_errors = []
        
        for i in range(state0_set.shape[0]):
            state0 = state0_set[i]
            
            # Ground truth
            _, traj_true = self.gt.integrate(state0, z_end, dz)
            
            # Model prediction
            traj_pred = self.rollout(model, state0, z_end, dz)
            traj_pred = traj_pred.squeeze(0)
            
            errors = self.compute_errors(traj_pred, traj_true)
            all_errors.append(errors)
        
        # Aggregate
        result = {}
        for key in all_errors[0]:
            if "max" in key:
                result[key] = max(e[key] for e in all_errors)
            else:
                result[key] = np.mean([e[key] for e in all_errors])
                
        # Timing
        state0 = state0_set[0]
        model.eval()
        
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        start = time.time()
        for _ in range(10):
            _ = self.rollout(model, state0, z_end, dz)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        result["rollout_time"] = (time.time() - start) / 10
        
        return result


# =============================================================================
# Data Generation
# =============================================================================

def generate_lhcb_training_data(
    ode: LHCbODE,
    state0_list: List[torch.Tensor],
    z_end: float,
    dz: float,
    substeps: int = 100,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for LHCb.
    
    Returns:
        X: States (N, 6)
        K: RK stages (N, 4, 6)
    """
    gt = LHCbGroundTruth(ode, substeps=substeps, device=device)
    
    X_list = []
    K_list = []
    
    for state0 in state0_list:
        _, trajectory = gt.integrate(state0, z_end, dz)
        
        # Exclude last state
        states = trajectory[:-1]
        X_list.append(states)
        
        # Get RK stages
        K = gt.get_rk_stages(states, dz)
        K_list.append(K)
    
    X = torch.cat(X_list, dim=0)
    K = torch.cat(K_list, dim=0)
    
    return X, K


def generate_random_tracks(
    n_tracks: int,
    z_start: float = 0.0,
    x_range: Tuple[float, float] = (-100, 100),
    y_range: Tuple[float, float] = (-100, 100),
    tx_range: Tuple[float, float] = (-0.3, 0.3),
    ty_range: Tuple[float, float] = (-0.3, 0.3),
    p_range: Tuple[float, float] = (5e3, 50e3),  # MeV
    charges: List[int] = [-1, 1],
    device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Generate random track initial conditions.
    
    Returns:
        states: (n_tracks, 6) tensor with [x, y, z, tx, ty, q/p]
    """
    states = []
    
    for _ in range(n_tracks):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = z_start
        tx = np.random.uniform(*tx_range)
        ty = np.random.uniform(*ty_range)
        p = np.random.uniform(*p_range)
        q = np.random.choice(charges)
        q_over_p = q / p
        
        states.append([x, y, z, tx, ty, q_over_p])
    
    return torch.tensor(states, dtype=torch.float32, device=device)


# =============================================================================
# Visualization
# =============================================================================

def plot_lhcb_comparison(
    models: Dict[str, nn.Module],
    state0: torch.Tensor,
    z_end: float,
    dz: float,
    gt: LHCbGroundTruth,
    save_path: Optional[str] = None
):
    """
    Plot comparison of LHCb track extrapolation methods.
    """
    evaluator = LHCbEvaluator(gt)
    
    # Ground truth trajectory
    _, traj_true = gt.integrate(state0, z_end, dz)
    traj_true = traj_true.cpu().numpy()
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(traj_true[:, 2], traj_true[:, 0], traj_true[:, 1],
             'k-', lw=2, label='Ground Truth')
    
    colors = ['b', 'r', 'g']
    for (name, model), color in zip(models.items(), colors):
        traj = evaluator.rollout(model, state0, z_end, dz)
        traj = traj.squeeze(0).cpu().numpy()
        ax1.plot(traj[:, 2], traj[:, 0], traj[:, 1],
                 f'{color}--', lw=1.5, label=name)
    
    ax1.set_xlabel('z (cm)')
    ax1.set_ylabel('x (cm)')
    ax1.set_zlabel('y (cm)')
    ax1.set_title('3D Track')
    ax1.legend()
    
    # x-z projection
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(traj_true[:, 2], traj_true[:, 0], 'k-', lw=2, label='Truth')
    for (name, model), color in zip(models.items(), colors):
        traj = evaluator.rollout(model, state0, z_end, dz)
        traj = traj.squeeze(0).cpu().numpy()
        ax2.plot(traj[:, 2], traj[:, 0], f'{color}--', lw=1.5, label=name)
    ax2.set_xlabel('z (cm)')
    ax2.set_ylabel('x (cm)')
    ax2.set_title('x-z Projection')
    ax2.legend()
    ax2.grid(True)
    
    # y-z projection
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(traj_true[:, 2], traj_true[:, 1], 'k-', lw=2, label='Truth')
    for (name, model), color in zip(models.items(), colors):
        traj = evaluator.rollout(model, state0, z_end, dz)
        traj = traj.squeeze(0).cpu().numpy()
        ax3.plot(traj[:, 2], traj[:, 1], f'{color}--', lw=1.5, label=name)
    ax3.set_xlabel('z (cm)')
    ax3.set_ylabel('y (cm)')
    ax3.set_title('y-z Projection')
    ax3.legend()
    ax3.grid(True)
    
    # Position error vs z
    ax4 = fig.add_subplot(2, 3, 4)
    z_vals = traj_true[:, 2]
    for (name, model), color in zip(models.items(), colors):
        traj = evaluator.rollout(model, state0, z_end, dz)
        traj = traj.squeeze(0).cpu().numpy()
        pos_error = np.linalg.norm(traj[:, :3] - traj_true[:, :3], axis=1)
        ax4.semilogy(z_vals, pos_error, color, label=name)
    ax4.set_xlabel('z (cm)')
    ax4.set_ylabel('Position Error (cm)')
    ax4.set_title('Position Error vs z')
    ax4.legend()
    ax4.grid(True)
    
    # Slope errors
    ax5 = fig.add_subplot(2, 3, 5)
    for (name, model), color in zip(models.items(), colors):
        traj = evaluator.rollout(model, state0, z_end, dz)
        traj = traj.squeeze(0).cpu().numpy()
        slope_error = np.linalg.norm(traj[:, 3:5] - traj_true[:, 3:5], axis=1)
        ax5.semilogy(z_vals, slope_error, color, label=name)
    ax5.set_xlabel('z (cm)')
    ax5.set_ylabel('Slope Error')
    ax5.set_title('Slope Error vs z')
    ax5.legend()
    ax5.grid(True)
    
    # Bar chart summary
    ax6 = fig.add_subplot(2, 3, 6)
    model_names = list(models.keys())
    metrics = []
    for model in models.values():
        errors = evaluator.compute_errors(
            evaluator.rollout(model, state0, z_end, dz).squeeze(0),
            torch.tensor(traj_true, device=state0.device)
        )
        metrics.append(errors)
    
    x_pos = np.arange(len(model_names))
    width = 0.3
    ax6.bar(x_pos - width/2, [m["pos_final"] for m in metrics], width, label='Pos Final')
    ax6.bar(x_pos + width/2, [m["slope_final"] for m in metrics], width, label='Slope Final')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(model_names)
    ax6.set_ylabel('Error')
    ax6.set_title('Final Errors')
    ax6.legend()
    ax6.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# Main Comparison Function
# =============================================================================

def run_lhcb_comparison(
    field_path: str = None,
    z_end: float = 500.0,
    dz: float = 10.0,
    n_tracks: int = 20,
    hidden_dim: int = 128,
    num_layers: int = 4,
    config: LHCbTrainingConfig = None,
    save_dir: str = None,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Run full comparison for LHCb track extrapolation.
    
    Args:
        field_path: Path to LHCb field map file (uses Quadratic if None)
        z_end: End z position
        dz: Step size
        n_tracks: Number of training tracks
        hidden_dim: Network width
        num_layers: Network depth
        config: Training configuration
        save_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        results: Dictionary of evaluation metrics
        models: Dictionary of trained models
    """
    config = config or LHCbTrainingConfig(dz=dz, z_end=z_end)
    device = config.device
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("LHCb Track Extrapolation Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"z_end: {z_end} cm, dz: {dz} cm")
    print()
    
    # Setup field
    if field_path and os.path.exists(field_path):
        print(f"Loading LHCb field from {field_path}")
        field = LHCb_Field(field_path)
    else:
        print("Using quadratic test field")
        field = Quadratic_Field(B0=1e-3)
    
    ode = LHCbODE(field)
    gt = LHCbGroundTruth(ode, substeps=100, device=device)
    
    # Generate tracks
    print(f"Generating {n_tracks} training tracks...")
    state0_list = generate_random_tracks(n_tracks, z_start=0.0, device=device)
    
    # Generate training data
    print("Computing RK stages for training data...")
    X_train, K_train = generate_lhcb_training_data(
        ode, [s for s in state0_list], z_end, dz, device=device
    )
    print(f"  States: {X_train.shape}, Stages: {K_train.shape}")
    
    results = {}
    models = {}
    
    # =========================================================================
    # Train RK-PINN
    # =========================================================================
    if verbose:
        print("\n" + "=" * 40)
        print("Training RK-PINN")
        print("=" * 40)
    
    rk_pinn = LHCbRKPINN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dz=dz,
        field=field
    ).to(device)
    
    trainer = LHCbTrainer(rk_pinn, config)
    train_result = trainer.train(X_train, K_train, use_physics=True, verbose=verbose)
    
    # Evaluate
    evaluator = LHCbEvaluator(gt, device)
    eval_result = evaluator.full_evaluation(rk_pinn, state0_list[:5], z_end, dz)
    
    results["RK-PINN"] = {**train_result, **eval_result}
    models["RK-PINN"] = rk_pinn
    
    # =========================================================================
    # Train Large-RK
    # =========================================================================
    if verbose:
        print("\n" + "=" * 40)
        print("Training Large-RK")
        print("=" * 40)
    
    large_rk = LHCbLargeRK(
        hidden_dim=hidden_dim * 2,
        num_layers=num_layers + 2,
        dz=dz
    ).to(device)
    
    trainer = LHCbTrainer(large_rk, config)
    train_result = trainer.train(X_train, K_train, use_physics=False, verbose=verbose)
    
    eval_result = evaluator.full_evaluation(large_rk, state0_list[:5], z_end, dz)
    
    results["Large-RK"] = {**train_result, **eval_result}
    models["Large-RK"] = large_rk
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            "Method": name,
            "Train Time (s)": f"{res['train_time']:.2f}",
            "Final Loss": f"{res['final_loss']:.2e}",
            "Pos Error (cm)": f"{res['pos_mean']:.4f}",
            "Slope Error": f"{res['slope_mean']:.2e}",
            "Rollout (ms)": f"{res['rollout_time']*1000:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    if save_dir:
        df.to_csv(os.path.join(save_dir, "lhcb_comparison.csv"), index=False)
    
    # Plot
    plot_lhcb_comparison(
        models, state0_list[0], z_end, dz, gt,
        save_path=os.path.join(save_dir, "lhcb_comparison.png") if save_dir else None
    )
    
    return results, models


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    config = LHCbTrainingConfig(
        max_epochs=10000,
        patience=50,
        physics_weight=0.1,
        dz=10.0,
        z_end=500.0
    )
    
    results, models = run_lhcb_comparison(
        field_path=None,  # Use test field
        z_end=500.0,
        dz=10.0,
        n_tracks=10,
        hidden_dim=64,
        num_layers=3,
        config=config,
        save_dir="Results/LHCb/Comparison"
    )
