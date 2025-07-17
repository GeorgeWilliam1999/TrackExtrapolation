import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from mpl_toolkits.mplot3d import Axes3D  # 3D projection
import numpy as np
import os
import time

# parameters
dt = 0.1  # time step
t_end = 30  # total time
t0 = 0  # initial time
N = int(t_end / dt)  # number of time steps
M = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Van der Pol System (device-safe)
# ----------------------------
def vdp(y, mu=1.0):
    return torch.stack([
        y[1],
        mu * (1 - y[0]**2) * y[1] - y[0]
    ]).to(y)

# ----------------------------
# General RK Integrator (Explicit & Implicit)
# ----------------------------
def rk_apply(butcher, x, dt, f, max_iter=10, tol=1e-8):
    A = torch.tensor(butcher['A'], dtype=x.dtype, device=x.device)
    b = torch.tensor(butcher['b'], dtype=x.dtype, device=x.device)

    s = len(b)
    d = x.shape[0]
    k = torch.zeros((s, d), dtype=x.dtype, device=x.device)

    for i in range(s):
        k_i = k[i].clone()


        #The is a banach iterator and I should try a newton method instead
        def G(ki_guess):
            weighted_sum = sum(A[i, j] * (ki_guess if j == i else k[j]) for j in range(s))
            return f(x + dt * weighted_sum)

        for _ in range(max_iter):
            ki_new = G(k_i)
            if torch.norm(ki_new - k_i) < tol:
                break
            k_i = ki_new

        k[i] = k_i

    x_next = x + dt * torch.sum(b.view(-1, 1) * k, dim=0)
    return k, x_next

# ----------------------------
# Training Data Generation
# ----------------------------
def generate_training_data(func, y0, t0, t_end, dt=dt, butcher=None, mu=1.0, M = M, device="cpu"):
    """
    Integrate using rk_apply and collect (x_n, k_n) pairs for NN training.
    """

    if butcher is None:
        raise ValueError("Butcher tableau must be provided.")
    
    N = int((t_end - t0) / dt) + 1
    
    x = y0.to(device)
    d = x.shape[0]
    s = len(butcher['b'])

    X = torch.zeros((N, d), dtype=torch.float32, device=device)
    K = torch.zeros((N, s, d), dtype=torch.float32, device=device)

    for n in range(N):
        X[n] = x
        k, x_next = rk_apply(butcher, x, dt/M, lambda y: func(y, mu=mu))
        for i in range(M-1):
            k, x_next = rk_apply(butcher, x_next, dt/M, lambda y: func(y, mu=mu))
        K[n] = k
        x = x_next

    return X, K

def generate_training_data_all_ics(func, y0s, t0, t_end, dt, butcher, mu=1.0, M = M, device="cpu"):
    data = []
    for y0 in y0s:
        X, K = generate_training_data(func, y0, t0, t_end, dt, butcher, mu=mu, M=M, device=device)
        data.append((X, K))

    X_all = torch.cat([pair[0] for pair in data], dim=0)
    K_all = torch.cat([pair[1] for pair in data], dim=0)
    return X_all, K_all