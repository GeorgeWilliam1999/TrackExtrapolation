from modularisation.vdp_utils import *
from modularisation.model_utils import *
from modularisation.eval_utils import *
from modularisation.magnetic_field import *
from modularisation.particle import *

import os
import time
import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# -----------------------
# Defaults / constants
# -----------------------
DEFAULT_DATA_PATH = os.path.join("RK_PINNs", "Data", "VDP", "VDP_Training.pt")
DEFAULT_RESULTS_DIR = os.path.join("RK_PINNs", "Results", "VDP", "Models")

DEFAULT_DT = 0.1
DEFAULT_T_END = 30.0
DEFAULT_BATCH = 64
DEFAULT_LR = 1e-3
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

RK4_TABLEAU = {
    "A": [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    "b": [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    "c": [0.0, 0.5, 0.5, 1.0],
}


X_loaded, K_loaded = torch.load("Data/VDP/VDP_Training.pt")

X, K = X_loaded.to(device), K_loaded.to(device)

K_ = K.reshape(K.shape[0]//(int((t_end - t0) / dt) + 1),int((t_end - t0) / dt) + 1,4,2)
X_ = X.reshape(X.shape[0]//(int((t_end - t0) / dt) + 1),int((t_end - t0) / dt) + 1,2)



# === Select Device ===
print(f"Using device: {device}")

# === Create Model ===
output_dim = math.prod(K_.shape[1:])
print(f"Output dimension: {output_dim}")
model = NeuralRK(hidden_dim=64, num_layers=4, output_dim=output_dim, butcher=rk4, dt=0.1).to(device)

# === Custom Forward & Loss Functions ===
def forward(x):
    x = x.view(x.size(0), -1)
    k = model.net(x)
    return k

def traj_loss(x_, k_):
    k_pred = model.forward(x_)

    return torch.mean((k_pred - k_)** 2)

model.forward = forward
model.loss_fn = traj_loss

# === Optimizer ===
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Model Configuration: {model.num_layers} layers | {model.hidden_dim} hidden dim | dt = {model.dt}")

# === Prepare Data ===
X_ = X_.to(device)
K_ = K_.to(device)
print(f'Shape on device : X_ = {X_.shape}, K_ = {K_.shape}')

# === Convergence Parameters ===
min_epochs = 100
patience = 20
delta_tol = 1e-9
max_epochs = 1000000
batch_size = 64

# === Training or Load Check ===
if not model.does_model_exist("VDP", "RK4_traj"):
    print("Starting training...")
    print(f"Model: NeuralRK_VDP_hd{model.hidden_dim}_layers{model.num_layers}_dt{model.dt}_RK4.pt")

    best_loss = float("inf")
    wait = 0
    epoch = 0

    while True:
        idx = torch.randperm(X_.size(0), device=device)[:batch_size]
        x_batch = X_[idx,0]
        k_batch = K_[idx]
     
        optimizer.zero_grad()
        loss = model.loss_fn(x_batch, k_batch.reshape(batch_size, -1))
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        if epoch == 0:
            best_loss = loss_val

        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} | Loss = {loss_val:.6e} | Best = {best_loss:.6e} | Wait = {wait}")

        if epoch >= min_epochs:
            if abs(loss_val - best_loss) < delta_tol:
                wait += 1
                if wait >= patience:
                    print(f"\nConverged at epoch {epoch} | loss = {loss_val:.6e}")
                    break
            else:
                best_loss = loss_val
                wait = 0

        epoch += 1
        if epoch >= max_epochs:
            print("\nStopping early: reached max epochs.")
            break

    # Save model
    model.save_model("VDP", "RK4_traj")

else:
    print(f"Model already exists: NeuralRK_VDP_hd{model.hidden_dim}_layers{model.num_layers}_dt{model.dt}_RK4.pt")
    model.load_state_dict(torch.load(model.name_model("VDP", "RK4_traj"))["model_state_dict"])
    print("Model loaded successfully.")
