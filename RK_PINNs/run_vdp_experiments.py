"""
VDP PINN Comparison Experiments Runner

This script runs comprehensive experiments comparing:
- Standard PINN (learns y(t) directly)
- RK-PINN (learns RK stages with physics loss)
- Large-RK (data-driven RK model)

Usage:
    python run_vdp_experiments.py [--quick] [--output-dir DIR]
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from modularisation.pinn_comparison import (
    StandardPINN, RKPINN, LargeRKModel,
    GroundTruthGenerator, Trainer, Evaluator,
    TrainingConfig, ExperimentResults,
    vdp_system, vdp_batched,
    generate_vdp_training_data, generate_collocation_points,
    RK4_TABLEAU, DEVICE
)


def run_experiment(config: dict, output_dir: str, verbose: bool = True):
    """
    Run a single VDP experiment with the given configuration.
    
    Parameters
    ----------
    config : dict
        Experiment configuration with keys:
        - t_end, dt, mu (VDP parameters)
        - hidden_dim, num_layers (network architecture)
        - n_ics (number of initial conditions)
        - max_epochs, lr, etc. (training parameters)
    output_dir : str
        Directory to save results
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Results dictionary with metrics for each method
    """
    device = DEVICE
    
    if verbose:
        print("\n" + "=" * 60)
        print("VDP EXPERIMENT")
        print("=" * 60)
        print(f"Configuration: {config}")
        print(f"Device: {device}")
    
    # Ground truth generator
    gt = GroundTruthGenerator(
        f=lambda y: vdp_system(y, mu=config["mu"]),
        substeps=1000,
        device=device
    )
    
    # Training configuration
    train_config = TrainingConfig(
        lr=config.get("lr", 1e-3),
        batch_size=config.get("batch_size", 64),
        max_epochs=config.get("max_epochs", 20000),
        min_epochs=config.get("min_epochs", 200),
        patience=config.get("patience", 50),
        delta_tol=config.get("delta_tol", 1e-7),
        physics_weight=config.get("physics_weight", 0.1),
        log_interval=config.get("log_interval", 500),
        device=device
    )
    
    # Generate training data
    torch.manual_seed(42)
    x0_list = [
        torch.tensor([
            2 * torch.rand(1).item() - 1,
            2 * torch.rand(1).item() - 1
        ], device=device)
        for _ in range(config["n_ics"])
    ]
    
    X_train, K_train = generate_vdp_training_data(
        f=lambda y: vdp_system(y, mu=config["mu"]),
        x0_list=x0_list,
        t_end=config["t_end"],
        dt=config["dt"],
        device=device
    )
    
    if verbose:
        print(f"Training samples: {X_train.shape[0]}")
    
    # Reference trajectory for PINN
    x0_ref = torch.tensor([2.0, 0.0], device=device)
    t_ref, traj_ref = gt.integrate(x0_ref, config["t_end"], config["dt"])
    
    results = {}
    
    # ============ STANDARD PINN ============
    if verbose:
        print("\n--- Training Standard PINN ---")
    
    pinn = StandardPINN(
        input_dim=1,
        state_dim=2,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        activation="tanh"
    ).to(device)
    pinn.f = lambda y: vdp_system(y, mu=config["mu"])
    
    t_colloc = generate_collocation_points(
        0, config["t_end"], 1000, method="uniform", device=device
    )
    
    trainer_pinn = Trainer(pinn, train_config)
    result_pinn = trainer_pinn.train_pinn(
        t_ref.unsqueeze(-1), traj_ref, t_colloc, verbose=verbose
    )
    
    results["PINN"] = {
        "train_time": result_pinn.train_time,
        "final_loss": result_pinn.final_loss,
        "epochs": result_pinn.convergence_epoch,
        "n_params": sum(p.numel() for p in pinn.parameters()),
        "loss_history": result_pinn.loss_history
    }
    
    # ============ RK-PINN ============
    if verbose:
        print("\n--- Training RK-PINN ---")
    
    rk_pinn = RKPINN(
        input_dim=2,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        output_dim=2,
        dt=config["dt"]
    ).to(device)
    rk_pinn.f = lambda y: vdp_system(y, mu=config["mu"])
    
    trainer_rk = Trainer(rk_pinn, train_config)
    result_rk = trainer_rk.train_rk(X_train, K_train, use_physics=True, verbose=verbose)
    
    results["RK-PINN"] = {
        "train_time": result_rk.train_time,
        "final_loss": result_rk.final_loss,
        "epochs": result_rk.convergence_epoch,
        "n_params": sum(p.numel() for p in rk_pinn.parameters()),
        "loss_history": result_rk.loss_history
    }
    
    # ============ LARGE-RK ============
    if verbose:
        print("\n--- Training Large-RK ---")
    
    large_rk = LargeRKModel(
        input_dim=2,
        hidden_dim=config["hidden_dim"] * 2,
        num_layers=config["num_layers"] + 2,
        output_dim=2,
        dt=config["dt"]
    ).to(device)
    
    trainer_large = Trainer(large_rk, train_config)
    result_large = trainer_large.train_rk(X_train, K_train, use_physics=False, verbose=verbose)
    
    results["Large-RK"] = {
        "train_time": result_large.train_time,
        "final_loss": result_large.final_loss,
        "epochs": result_large.convergence_epoch,
        "n_params": sum(p.numel() for p in large_rk.parameters()),
        "loss_history": result_large.loss_history
    }
    
    # ============ EVALUATION ============
    if verbose:
        print("\n--- Evaluating Models ---")
    
    evaluator = Evaluator(gt, device)
    
    # Test initial conditions
    x0_test = torch.stack([
        torch.tensor([2.0, 0.0]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([1.5, -0.5]),
        torch.tensor([0.0, 2.0]),
    ]).to(device)
    
    models = {"PINN": pinn, "RK-PINN": rk_pinn, "Large-RK": large_rk}
    
    for name, model in models.items():
        model_type = "pinn" if name == "PINN" else "rk"
        eval_result = evaluator.full_evaluation(
            model, x0_test, config["t_end"], config["dt"], model_type=model_type
        )
        
        results[name].update({
            "mean_error": eval_result.mean_error,
            "max_error": eval_result.max_error,
            "final_error": eval_result.final_error,
            "rollout_time": eval_result.rollout_time
        })
        
        if verbose:
            print(f"\n{name}:")
            print(f"  Mean Error: {eval_result.mean_error:.4e}")
            print(f"  Max Error: {eval_result.max_error:.4e}")
            print(f"  Rollout Time: {eval_result.rollout_time*1000:.2f} ms")
    
    # ============ SAVE RESULTS ============
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary CSV
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            "Method": name,
            "Parameters": res["n_params"],
            "Train_Time_s": res["train_time"],
            "Epochs": res["epochs"],
            "Final_Loss": res["final_loss"],
            "Mean_Error": res["mean_error"],
            "Max_Error": res["max_error"],
            "Final_Error": res["final_error"],
            "Rollout_ms": res["rollout_time"] * 1000
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "vdp_results.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nResults saved to {csv_path}")
    
    # Save loss curves plot
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.semilogy(res["loss_history"], label=name, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    
    # Save trajectory comparison
    plot_trajectories(models, gt, config, output_dir, device)
    
    # Save models
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name.replace('-', '_').lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
    
    return results


def plot_trajectories(models, gt, config, output_dir, device):
    """Plot trajectory comparisons."""
    x0 = torch.tensor([2.0, 0.0], device=device)
    _, traj_true = gt.integrate(x0, config["t_end"], config["dt"])
    traj_true = traj_true.cpu().numpy()
    
    steps = int(config["t_end"] / config["dt"])
    t = np.linspace(0, config["t_end"], steps + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phase space
    ax = axes[0]
    ax.plot(traj_true[:, 0], traj_true[:, 1], "k-", lw=3, label="Ground Truth")
    
    colors = {"PINN": "blue", "RK-PINN": "red", "Large-RK": "green"}
    for name, model in models.items():
        model.eval()
        if name == "PINN":
            t_eval = torch.linspace(0, config["t_end"], steps + 1, device=device)
            traj = model(t_eval.unsqueeze(-1)).detach().cpu().numpy()
        else:
            traj = []
            state = x0.unsqueeze(0)
            for _ in range(steps):
                K = model(state)
                b = torch.tensor(RK4_TABLEAU["b"], device=device)
                state = state + config["dt"] * (b.unsqueeze(0) @ K.transpose(1, 2)).squeeze(-2)
                traj.append(state.squeeze(0).cpu().numpy())
            traj = np.vstack([[x0.cpu().numpy()], traj])
        
        ax.plot(traj[:, 0], traj[:, 1], "--", color=colors[name], lw=2, label=name)
    
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
    ax.set_title("Phase Space")
    ax.legend()
    ax.grid(True)
    
    # Error over time
    ax = axes[1]
    for name, model in models.items():
        model.eval()
        if name == "PINN":
            t_eval = torch.linspace(0, config["t_end"], steps + 1, device=device)
            traj = model(t_eval.unsqueeze(-1)).detach().cpu().numpy()
        else:
            traj = []
            state = x0.unsqueeze(0)
            for _ in range(steps):
                K = model(state)
                b = torch.tensor(RK4_TABLEAU["b"], device=device)
                state = state + config["dt"] * (b.unsqueeze(0) @ K.transpose(1, 2)).squeeze(-2)
                traj.append(state.squeeze(0).cpu().numpy())
            traj = np.vstack([[x0.cpu().numpy()], traj])
        
        error = np.linalg.norm(traj - traj_true, axis=1)
        ax.semilogy(t, error, color=colors[name], lw=2, label=name)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title("Error vs Time")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_comparison.png"), dpi=150)
    plt.close()


def run_parameter_sweep(output_dir: str, quick: bool = False):
    """
    Run experiments over parameter ranges.
    
    Parameters
    ----------
    output_dir : str
        Base output directory
    quick : bool
        If True, run quick experiments with fewer epochs
    """
    base_config = {
        "t_end": 30.0,
        "dt": 0.1,
        "mu": 1.0,
        "n_ics": 10,
        "hidden_dim": 64,
        "num_layers": 3,
        "max_epochs": 2000 if quick else 20000,
        "patience": 20 if quick else 50,
    }
    
    all_results = []
    
    # Sweep over dt values
    dt_values = [0.05, 0.1, 0.2] if quick else [0.05, 0.1, 0.2, 0.5]
    
    for dt in dt_values:
        config = base_config.copy()
        config["dt"] = dt
        
        exp_dir = os.path.join(output_dir, f"dt_{dt}")
        results = run_experiment(config, exp_dir, verbose=True)
        
        for name, res in results.items():
            all_results.append({
                "dt": dt,
                "method": name,
                **{k: v for k, v in res.items() if k != "loss_history"}
            })
    
    # Save aggregate results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "sweep_results.csv"), index=False)
    
    # Plot sweep results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ["PINN", "RK-PINN", "Large-RK"]
    colors = {"PINN": "blue", "RK-PINN": "red", "Large-RK": "green"}
    
    for method in methods:
        method_df = df[df["method"] == method]
        axes[0].semilogy(
            method_df["dt"], method_df["mean_error"],
            "o-", color=colors[method], label=method, lw=2
        )
        axes[1].plot(
            method_df["dt"], method_df["train_time"],
            "o-", color=colors[method], label=method, lw=2
        )
    
    axes[0].set_xlabel("Time Step (dt)")
    axes[0].set_ylabel("Mean Error")
    axes[0].set_title("Accuracy vs Time Step")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel("Time Step (dt)")
    axes[1].set_ylabel("Train Time (s)")
    axes[1].set_title("Training Time vs Time Step")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sweep_summary.png"), dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*60}")
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description="Run VDP PINN comparison experiments")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick experiments with fewer epochs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="Results/VDP",
        help="Output directory for results"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run parameter sweep experiments"
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Parameter sweep: {args.sweep}")
    
    if args.sweep:
        run_parameter_sweep(output_dir, quick=args.quick)
    else:
        config = {
            "t_end": 30.0,
            "dt": 0.1,
            "mu": 1.0,
            "n_ics": 10,
            "hidden_dim": 64,
            "num_layers": 3,
            "max_epochs": 2000 if args.quick else 20000,
            "patience": 20 if args.quick else 50,
        }
        run_experiment(config, output_dir, verbose=True)


if __name__ == "__main__":
    main()
