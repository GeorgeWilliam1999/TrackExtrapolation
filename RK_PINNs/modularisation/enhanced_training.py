# enhanced_training.py
"""
Enhanced training utilities for LHCb track extrapolation PINNs.

This module contains improved training techniques based on PINN research:
- Input normalization
- Adaptive loss weighting
- Direct state-to-state neural network stepper
- Diverse track generation

References:
- Wang et al. "Understanding gradient pathologies in PINNs" (2021)
- Wang et al. "An Expert's Guide to Training PINNs" (2023)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Input Normalization
# =============================================================================

class InputNormalizer:
    """
    Normalizes input features to [-1, 1] range for better training stability.
    
    Essential for LHCb track state which has large scale differences:
    - Position: O(1000) cm
    - Momentum: O(10^7) MeV
    - q/p: O(0.01)
    """
    
    def __init__(self, X: np.ndarray):
        """
        Initialize normalizer from training data.
        
        Args:
            X: Training data of shape (N, features)
        """
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals = np.where(self.range_vals < 1e-10, 1.0, self.range_vals)
        
    def normalize(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize to [-1, 1]."""
        if isinstance(X, torch.Tensor):
            min_vals = torch.tensor(self.min_vals, device=X.device, dtype=X.dtype)
            range_vals = torch.tensor(self.range_vals, device=X.device, dtype=X.dtype)
            return 2.0 * (X - min_vals) / range_vals - 1.0
        return 2.0 * (X - self.min_vals) / self.range_vals - 1.0
    
    def denormalize(self, X_norm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize from [-1, 1]."""
        if isinstance(X_norm, torch.Tensor):
            min_vals = torch.tensor(self.min_vals, device=X_norm.device, dtype=X_norm.dtype)
            range_vals = torch.tensor(self.range_vals, device=X_norm.device, dtype=X_norm.dtype)
            return (X_norm + 1.0) / 2.0 * range_vals + min_vals
        return (X_norm + 1.0) / 2.0 * self.range_vals + self.min_vals
    
    def to(self, device: torch.device) -> 'InputNormalizer':
        """Move normalizer tensors to device (for compatibility)."""
        return self


# =============================================================================
# Direct State-to-State Neural Network Stepper
# =============================================================================

class DirectLHCbStepper(nn.Module):
    """
    Direct neural network for track extrapolation.
    
    Instead of learning RK4 stage derivatives K, this model directly learns
    the mapping: state_n -> state_{n+1}
    
    Training data: pairs of (state_n, state_{n+1}) from RK4 reference.
    
    Key insight: Work entirely in normalized state space for better training.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        state_min: np.ndarray = None,
        state_max: np.ndarray = None
    ):
        super().__init__()
        
        # Normalization parameters
        if state_min is not None:
            self.register_buffer('state_min', torch.tensor(state_min, dtype=torch.float32))
            self.register_buffer('state_max', torch.tensor(state_max, dtype=torch.float32))
            state_range = state_max - state_min
            state_range = np.where(state_range < 1e-10, 1.0, state_range)
            self.register_buffer('state_range', torch.tensor(state_range, dtype=torch.float32))
        
        # Neural network - predicts delta_state in normalized space
        layers = []
        layers.append(nn.Linear(6, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
        
        # Output: residual (delta in normalized space)
        layers.append(nn.Linear(hidden_dim, 6))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize final layer with small weights
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
    
    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to [-1, 1] range."""
        return 2.0 * (state - self.state_min) / self.state_range - 1.0
    
    def denormalize(self, state_norm: torch.Tensor) -> torch.Tensor:
        """De-normalize state back to physical units."""
        return (state_norm + 1.0) / 2.0 * self.state_range + self.state_min
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict next state.
        Input: current state in PHYSICAL units [batch, 6]
        Output: next state in PHYSICAL units [batch, 6]
        """
        # Normalize input
        state_norm = self.normalize(state)
        
        # Predict residual in normalized space
        delta_norm = self.net(state_norm)
        
        # Output is current normalized state plus residual
        next_norm = state_norm + delta_norm
        
        # De-normalize
        return self.denormalize(next_norm)
    
    def rollout(self, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Extrapolate track for n_steps.
        Input: initial_state [6] in physical units
        Output: trajectory [n_steps+1, 6] in physical units
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        
        trajectory = [initial_state.clone()]
        state = initial_state.clone()
        
        for _ in range(n_steps):
            state = self.forward(state)
            trajectory.append(state.clone())
        
        return torch.cat(trajectory, dim=0)


# =============================================================================
# Training Data Generation
# =============================================================================

def generate_diverse_initial_conditions(
    n_tracks: int,
    p_range: Tuple[float, float] = (1.0, 100.0),  # GeV
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate diverse track initial conditions.
    
    Args:
        n_tracks: Number of tracks to generate
        p_range: Momentum range in GeV (min, max)
        device: Torch device
        
    Returns:
        states: (n_tracks, 6) tensor with [x, y, z, tx, ty, q/p]
    """
    device = device or DEVICE
    
    states = []
    for _ in range(n_tracks):
        # Position - LHCb-like acceptance
        x = np.random.uniform(-200, 200)  # cm
        y = np.random.uniform(-200, 200)
        z = 0.0
        
        # Slopes - forward spectrometer geometry
        tx = np.random.uniform(-0.4, 0.4)
        ty = np.random.uniform(-0.3, 0.3)
        
        # Momentum (log-uniform for better coverage)
        log_p = np.random.uniform(np.log(p_range[0]), np.log(p_range[1]))
        p = np.exp(log_p)  # GeV
        
        # Random charge
        q = np.random.choice([-1, 1])
        q_over_p = q / p
        
        states.append([x, y, z, tx, ty, q_over_p])
    
    return torch.tensor(states, dtype=torch.float32, device=device)


def generate_state_pairs(ode, n_tracks: int = 200, z_end: float = 200.0, dz: float = 10.0):
    """
    Generate training data: pairs of (current_state, next_state) from RK4 propagation.
    
    Args:
        ode: ODE object with numpy_call method
        n_tracks: Number of tracks to generate
        z_end: End z position in cm
        dz: Step size in cm
        
    Returns:
        X_current, X_next: Arrays of state pairs
    """
    all_current = []
    all_next = []
    
    n_steps = int(z_end / dz)
    
    for _ in range(n_tracks):
        # Generate diverse initial conditions
        x = np.random.uniform(-200, 200)
        y = np.random.uniform(-200, 200)
        z = 0.0
        
        tx = np.random.uniform(-0.4, 0.4)
        ty = np.random.uniform(-0.3, 0.3)
        
        # Momentum: 1-100 GeV -> MeV
        log_p = np.random.uniform(np.log(1.0), np.log(100.0))
        p = np.exp(log_p) * 1e3  # MeV
        
        q = np.random.choice([-1, 1])
        q_over_p = q / p
        
        pz = p / np.sqrt(1 + tx**2 + ty**2)
        px = tx * pz
        py = ty * pz
        
        state = np.array([x, y, z, px, py, q_over_p])
        
        # Propagate and record pairs
        for _ in range(n_steps):
            current_state = state.copy()
            
            # RK4 step
            k1 = ode.numpy_call(state) * dz
            k2 = ode.numpy_call(state + 0.5 * k1) * dz
            k3 = ode.numpy_call(state + 0.5 * k2) * dz
            k4 = ode.numpy_call(state + k3) * dz
            state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
            all_current.append(current_state)
            all_next.append(state.copy())
    
    return np.array(all_current), np.array(all_next)


# =============================================================================
# Adaptive Loss Weighting Trainer
# =============================================================================

class AdaptiveWeightTrainer:
    """
    Trainer with adaptive loss weighting based on gradient statistics.
    
    Implements the method from Wang et al. "Understanding gradient pathologies in PINNs"
    to balance data loss and physics loss during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = None,
        alpha: float = 0.9
    ):
        self.model = model
        self.device = device or DEVICE
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=100, factor=0.5, min_lr=1e-6
        )
        
        self.alpha = alpha
        self.lambda_data = 1.0
        self.lambda_physics = 1.0
        
        self.loss_history = []
    
    def compute_adaptive_weights(self, loss_data: torch.Tensor, loss_physics: torch.Tensor):
        """Compute adaptive weights based on gradient magnitudes."""
        self.optimizer.zero_grad()
        loss_data.backward(retain_graph=True)
        grad_data = torch.cat([p.grad.view(-1) for p in self.model.parameters() 
                               if p.grad is not None])
        grad_data_norm = grad_data.norm()
        
        self.optimizer.zero_grad()
        loss_physics.backward(retain_graph=True)
        grad_physics = torch.cat([p.grad.view(-1) for p in self.model.parameters() 
                                   if p.grad is not None])
        grad_physics_norm = grad_physics.norm()
        
        self.optimizer.zero_grad()
        
        if grad_data_norm > 1e-10 and grad_physics_norm > 1e-10:
            total_grad = grad_data_norm + grad_physics_norm
            new_lambda_data = total_grad / (2 * grad_data_norm)
            new_lambda_physics = total_grad / (2 * grad_physics_norm)
            
            self.lambda_data = self.alpha * self.lambda_data + (1 - self.alpha) * new_lambda_data.item()
            self.lambda_physics = self.alpha * self.lambda_physics + (1 - self.alpha) * new_lambda_physics.item()
    
    def train(
        self,
        X: torch.Tensor,
        K: torch.Tensor,
        max_epochs: int = 5000,
        batch_size: int = 64,
        log_interval: int = 100,
        use_adaptive_weights: bool = True,
        verbose: bool = True
    ) -> Dict:
        """Full training loop."""
        X = X.to(self.device)
        K = K.to(self.device)
        N = X.shape[0]
        
        start_time = time.time()
        
        pbar = tqdm(range(max_epochs), disable=not verbose)
        for epoch in pbar:
            idx = torch.randperm(N, device=self.device)[:batch_size]
            x_batch = X[idx]
            k_batch = K[idx]
            
            self.model.train()
            K_pred = self.model.forward(x_batch)
            loss_data = nn.functional.mse_loss(K_pred, k_batch)
            
            # Physics loss if ODE is available
            loss_physics = torch.tensor(0.0, device=self.device)
            if hasattr(self.model, 'ode') and self.model.ode is not None:
                for i in range(self.model.s):
                    A_row = self.model.A[i].view(1, -1, 1)
                    state_stage = x_batch + self.model.dz * torch.sum(A_row * K_pred, dim=1)
                    k_true = self.model.ode(state_stage)
                    if torch.isfinite(k_true).all():
                        loss_physics = loss_physics + nn.functional.mse_loss(K_pred[:, i], k_true)
                loss_physics = loss_physics / self.model.s
            
            if use_adaptive_weights and loss_physics.item() > 0:
                self.compute_adaptive_weights(loss_data, loss_physics)
            
            total_loss = self.lambda_data * loss_data + self.lambda_physics * loss_physics
            
            self.optimizer.zero_grad()
            if torch.isfinite(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.loss_history.append(total_loss.item() if torch.isfinite(total_loss) else float('nan'))
            
            if epoch % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss.item():.2e}",
                    "λ_d": f"{self.lambda_data:.2f}",
                    "λ_p": f"{self.lambda_physics:.2f}"
                })
        
        train_time = time.time() - start_time
        return {
            "train_time": train_time,
            "final_loss": self.loss_history[-1],
            "loss_history": self.loss_history
        }


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_direct_model(
    model: DirectLHCbStepper, 
    ode, 
    dz: float = 10.0, 
    z_end: float = 200.0, 
    n_test: int = 50,
    device: torch.device = None
) -> Dict:
    """
    Evaluate direct stepper model on diverse test tracks.
    
    Args:
        model: DirectLHCbStepper model
        ode: ODE object for generating ground truth
        dz: Step size
        z_end: End z position
        n_test: Number of test tracks
        device: Torch device
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = device or DEVICE
    model.eval()
    
    # Generate test tracks
    n_steps = int(z_end / dz)
    pos_errors = []
    final_pos_errors = []
    
    with torch.no_grad():
        for _ in range(n_test):
            # Generate initial conditions
            x = np.random.uniform(-200, 200)
            y = np.random.uniform(-200, 200)
            z = 0.0
            tx = np.random.uniform(-0.4, 0.4)
            ty = np.random.uniform(-0.3, 0.3)
            
            log_p = np.random.uniform(np.log(1.0), np.log(100.0))
            p = np.exp(log_p) * 1e3  # MeV
            q = np.random.choice([-1, 1])
            q_over_p = q / p
            
            pz = p / np.sqrt(1 + tx**2 + ty**2)
            px = tx * pz
            py = ty * pz
            
            state = np.array([x, y, z, px, py, q_over_p])
            
            # Generate ground truth trajectory
            true_traj = [state.copy()]
            for _ in range(n_steps):
                k1 = ode.numpy_call(state) * dz
                k2 = ode.numpy_call(state + 0.5 * k1) * dz
                k3 = ode.numpy_call(state + 0.5 * k2) * dz
                k4 = ode.numpy_call(state + k3) * dz
                state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
                true_traj.append(state.copy())
            true_traj = np.array(true_traj)
            
            # Model prediction
            state0 = torch.tensor(true_traj[0], dtype=torch.float32, device=device)
            pred_traj = model.rollout(state0, n_steps).cpu().numpy()
            
            # Compute errors
            n = min(len(true_traj), len(pred_traj))
            for i in range(n):
                pos_err = np.sqrt((true_traj[i, 0] - pred_traj[i, 0])**2 + 
                                  (true_traj[i, 1] - pred_traj[i, 1])**2)
                pos_errors.append(pos_err)
            
            final_pos_errors.append(pos_errors[-1] if pos_errors else 0)
    
    return {
        "pos_mean": np.nanmean(pos_errors),
        "pos_max": np.nanmax(pos_errors),
        "pos_final_mean": np.nanmean(final_pos_errors)
    }
