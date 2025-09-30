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


def run_multi_experiment(
    data_path=DEFAULT_DATA_PATH,
    results_dir=DEFAULT_RESULTS_DIR,
    hidden_dim=64,
    num_layers=4,
    dt=DEFAULT_DT,
    t_end=DEFAULT_T_END,
    batch_size=DEFAULT_BATCH,
    device=DEFAULT_DEVICE,
    train_if_missing=True,
):
    """
    Multi-step experiment runner that mirrors the single-step script's behaviour
    but trains/evaluates the model intended to predict trajectories (RK4_traj).
    """
    print(f"Using device: {device}")

    # Load data using helper (raises a clear error if missing)
    X, K = load_training_data(data_path, device=device)

    # Build model (reuse build_model helper)
    model = build_model(hidden_dim=hidden_dim, num_layers=num_layers, butcher=RK4_TABLEAU, dt=dt, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

    # Ensure results dir
    ensure_dir(results_dir)

    model_name = model.name_model("VDP", "RK4_traj")

    # Train (using the provided single-step trainer which works with X,K shapes)
    if train_if_missing and not model.does_model_exist("VDP", "RK4_traj"):
        print("Starting training (multi-step)...")
        print(f"Model will be saved as: {model_name}")
        # train_model_single_step accepts (model, X, K)
        train_info = train_model_single_step(
            model,
            X,
            K,
            optimizer=optimizer,
            batch_size=batch_size,
            min_epochs=100,
            patience=20,
            delta_tol=1e-6,
            max_epochs=100000,
            verbose=True,
        )
        print(f"Training finished: {train_info}")
        model.save_model("VDP", "RK4_traj")
    else:
        if model.does_model_exist("VDP", "RK4_traj"):
            print(f"Loading existing model: {model_name}")
            model.load_state_dict(torch.load(model_name)["model_state_dict"])
            print("Model loaded successfully.")
        else:
            print("Model not found and training disabled. Exiting.")
            return

    # Example evaluation: rollout neural and classical RK for one initial condition
    x0 = torch.tensor([2.0, 2.0], dtype=torch.float32).to(device)
    steps = int(t_end / dt)
    traj_nn = rollout_neural_model(model, x0, steps, dt, scheme="RK4_traj")
    traj_rk = rollout_rk4(x0, steps, dt, 10, RK4_TABLEAU, vdp)
    try:
        full_mode_analysis(traj_rk, traj_nn, steps, dt)
    except Exception:
        pass

    # Persist simple evaluation CSV
    ensure_dir(results_dir)
    df_nn = evaluate_and_time_saved_models(steps=steps, m=10, butcher=RK4_TABLEAU, f=vdp, x0_eval=sample_initial_conditions(20).to(device), x0=x0, t_end=dt, dt=dt, device=device)
    csv_path = os.path.join(results_dir, "last_evaluation_nn_multi.csv")
    df_nn.to_csv(csv_path, index=False)

    return {"model": model, "df_nn": df_nn}


def grid_experiments(
    hidden_dims=[16, 32],
    num_layers_list=[2, 4],
    repeats=1,
    **single_exp_kwargs
):
    device = single_exp_kwargs.get("device", DEFAULT_DEVICE)
    dt = single_exp_kwargs.get("dt", DEFAULT_DT)
    results = []

    x0_eval = sample_initial_conditions(20).to(device)
    x0 = torch.tensor([2.0, 2.0], dtype=torch.float32).to(device)

    for hidden_dim, num_layers in product(hidden_dims, num_layers_list):
        for r in range(repeats):
            print(f"Running config hidden_dim={hidden_dim}, num_layers={num_layers}, repeat={r+1}/{repeats}")
            model = build_model(hidden_dim=hidden_dim, num_layers=num_layers, butcher=RK4_TABLEAU, dt=dt, device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)

            # Train if missing
            if not model.does_model_exist("VDP", "RK4_traj"):
                train_model_single_step(model, *load_training_data(), optimizer=optimizer, batch_size=single_exp_kwargs.get("batch_size", DEFAULT_BATCH))
                model.save_model("VDP", "RK4_traj")

            # Evaluate timing & accuracy
            df = evaluate_and_time_saved_models(
                steps=single_exp_kwargs.get("steps", 1), m=single_exp_kwargs.get("m", 10), butcher=RK4_TABLEAU, f=vdp,
                x0_eval=x0_eval, x0=x0, t_end=dt, dt=dt, device=device
            )
            results.append({"hidden_dim": hidden_dim, "num_layers": num_layers, "repeat": r, "df": df})

    # Optionally collect results into a single DataFrame
    combined = pd.concat([r["df"] for r in results], ignore_index=True) if results else pd.DataFrame()
    return combined


def main():
    print("Starting multi-step experiment runner...")
    ensure_dir(DEFAULT_RESULTS_DIR)

    out = run_multi_experiment(
        data_path=DEFAULT_DATA_PATH,
        results_dir=DEFAULT_RESULTS_DIR,
        hidden_dim=64,
        num_layers=4,
        dt=DEFAULT_DT,
        t_end=DEFAULT_T_END,
        batch_size=DEFAULT_BATCH,
        device=DEFAULT_DEVICE,
        train_if_missing=True,
    )

    print("Finished. Key outputs:")
    if out is not None:
        print("Saved evaluation CSVs in:", DEFAULT_RESULTS_DIR)


if __name__ == "__main__":
    main()
