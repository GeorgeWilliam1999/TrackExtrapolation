"""
LHCb Track Extrapolation Experiments Runner

This script runs experiments comparing neural network approaches for 
LHCb track propagation through the magnetic field:
- RK-PINN with Lorentz force physics constraints
- Large-RK data-driven approach

The LHCb state is 6D: (x, y, z, tx, ty, q/p)
where tx = dx/dz, ty = dy/dz, q/p = charge/momentum

Usage:
    python run_lhcb_experiments.py [--quick] [--output-dir DIR] [--field quadratic|lhcb]
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from modularisation.lhcb_pinn import (
    LHCbODE, LHCbGroundTruth, LHCbRKPINN, LHCbLargeRK,
    LHCbTrainer, LHCbEvaluator, LHCbTrainingConfig,
    generate_lhcb_training_data, generate_random_tracks,
    plot_lhcb_comparison, run_lhcb_comparison
)

# Try to import real LHCb field, fall back to quadratic
try:
    from Classes.magnetic_field import LHCb_Field, Quadratic_Field
    HAS_LHCB_FIELD = True
except ImportError:
    from Classes.magnetic_field import Quadratic_Field
    HAS_LHCB_FIELD = False
    print("Warning: LHCb_Field not available, using Quadratic_Field")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment(config: dict, output_dir: str, field, verbose: bool = True):
    """
    Run a single LHCb experiment with the given configuration.
    
    Parameters
    ----------
    config : dict
        Experiment configuration
    output_dir : str
        Directory to save results
    field : MagneticField
        Magnetic field object
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
        print("LHCb TRACK EXTRAPOLATION EXPERIMENT")
        print("=" * 60)
        print(f"Configuration: {config}")
        print(f"Device: {device}")
        print(f"Field type: {type(field).__name__}")
    
    # Setup ODE and ground truth
    lhcb_ode = LHCbODE(field)
    lhcb_gt = LHCbGroundTruth(lhcb_ode, substeps=100, device=device)
    
    # Training configuration
    train_config = LHCbTrainingConfig(
        lr=config.get("lr", 1e-3),
        batch_size=config.get("batch_size", 64),
        max_epochs=config.get("max_epochs", 15000),
        min_epochs=config.get("min_epochs", 200),
        patience=config.get("patience", 50),
        physics_weight=config.get("physics_weight", 0.1),
        dz=config["dz"],
        z_end=config["z_end"],
        device=device
    )
    
    # Generate training tracks
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_tracks = generate_random_tracks(
        n_tracks=config["n_train_tracks"],
        z_start=config.get("z_start", 0.0),
        x_range=config.get("x_range", (-50, 50)),
        y_range=config.get("y_range", (-50, 50)),
        tx_range=config.get("tx_range", (-0.2, 0.2)),
        ty_range=config.get("ty_range", (-0.2, 0.2)),
        p_range=config.get("p_range", (5e3, 50e3)),
        device=device
    )
    
    if verbose:
        print(f"Generated {config['n_train_tracks']} training tracks")
    
    # Generate training data (states and RK stages)
    if verbose:
        print("Generating training data...")
    
    X_train, K_train = generate_lhcb_training_data(
        lhcb_ode,
        [t for t in train_tracks],
        config["z_end"],
        config["dz"],
        substeps=100,
        device=device
    )
    
    if verbose:
        print(f"Training states: {X_train.shape}")
        print(f"Training stages: {K_train.shape}")
    
    results = {}
    models = {}
    
    # ============ RK-PINN (Physics-Informed) ============
    if verbose:
        print("\n--- Training LHCb RK-PINN ---")
    
    rk_pinn = LHCbRKPINN(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dz=config["dz"],
        field=field
    ).to(device)
    
    trainer_rk = LHCbTrainer(rk_pinn, train_config)
    result_rk = trainer_rk.train(X_train, K_train, use_physics=True, verbose=verbose)
    
    results["RK-PINN"] = {
        "train_time": result_rk["train_time"],
        "final_loss": result_rk["final_loss"],
        "epochs": result_rk["epochs"],
        "n_params": sum(p.numel() for p in rk_pinn.parameters()),
        "loss_history": result_rk["loss_history"]
    }
    models["RK-PINN"] = rk_pinn
    
    if verbose:
        print(f"RK-PINN: {result_rk['epochs']} epochs, loss={result_rk['final_loss']:.2e}")
    
    # ============ LARGE-RK (Data-Driven) ============
    if verbose:
        print("\n--- Training LHCb Large-RK ---")
    
    large_rk = LHCbLargeRK(
        hidden_dim=config["hidden_dim"] * 2,
        num_layers=config["num_layers"] + 2,
        dz=config["dz"]
    ).to(device)
    
    trainer_large = LHCbTrainer(large_rk, train_config)
    result_large = trainer_large.train(X_train, K_train, use_physics=False, verbose=verbose)
    
    results["Large-RK"] = {
        "train_time": result_large["train_time"],
        "final_loss": result_large["final_loss"],
        "epochs": result_large["epochs"],
        "n_params": sum(p.numel() for p in large_rk.parameters()),
        "loss_history": result_large["loss_history"]
    }
    models["Large-RK"] = large_rk
    
    if verbose:
        print(f"Large-RK: {result_large['epochs']} epochs, loss={result_large['final_loss']:.2e}")
    
    # ============ EVALUATION ============
    if verbose:
        print("\n--- Evaluating Models ---")
    
    # Generate test tracks (different from training)
    torch.manual_seed(123)
    test_tracks = generate_random_tracks(
        n_tracks=config.get("n_test_tracks", 5),
        z_start=config.get("z_start", 0.0),
        x_range=config.get("x_range", (-50, 50)),
        y_range=config.get("y_range", (-50, 50)),
        tx_range=config.get("tx_range", (-0.2, 0.2)),
        ty_range=config.get("ty_range", (-0.2, 0.2)),
        p_range=config.get("p_range", (5e3, 50e3)),
        device=device
    )
    
    evaluator = LHCbEvaluator(lhcb_gt, device)
    
    for name, model in models.items():
        eval_result = evaluator.full_evaluation(
            model, test_tracks, config["z_end"], config["dz"]
        )
        
        results[name].update({
            "pos_mean": eval_result["pos_mean"],
            "pos_max": eval_result["pos_max"],
            "slope_mean": eval_result["slope_mean"],
            "slope_max": eval_result["slope_max"],
            "rollout_time": eval_result["rollout_time"]
        })
        
        if verbose:
            print(f"\n{name}:")
            print(f"  Position Error: {eval_result['pos_mean']:.4f} ± {eval_result['pos_max']:.4f} cm")
            print(f"  Slope Error: {eval_result['slope_mean']:.2e}")
            print(f"  Rollout Time: {eval_result['rollout_time']*1000:.2f} ms")
    
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
            "Pos_Mean_cm": res["pos_mean"],
            "Pos_Max_cm": res["pos_max"],
            "Slope_Mean": res["slope_mean"],
            "Rollout_ms": res["rollout_time"] * 1000
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "lhcb_results.csv")
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nResults saved to {csv_path}")
    
    # Save training curves
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.semilogy(res["loss_history"], label=name, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LHCb Training Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    
    # Save trajectory comparison
    plot_track_comparison(models, lhcb_gt, test_tracks[0], config, output_dir)
    
    # Save 3D trajectory
    plot_3d_track(models, lhcb_gt, test_tracks[0], config, output_dir)
    
    # Save models
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name.replace('-', '_').lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
    
    return results, models


def plot_track_comparison(models, gt, x0, config, output_dir):
    """Plot 2D track projections comparison."""
    device = x0.device
    steps = int(config["z_end"] / config["dz"])
    
    # Ground truth trajectory
    _, traj_true = gt.integrate(x0, config["z_end"], config["dz"])
    traj_true = traj_true.cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {"RK-PINN": "red", "Large-RK": "green"}
    z = np.linspace(0, config["z_end"], steps + 1)
    
    # x vs z
    ax = axes[0, 0]
    ax.plot(z, traj_true[:, 0], "k-", lw=3, label="Ground Truth")
    for name, model in models.items():
        model.eval()
        traj = rollout_model(model, x0, steps, config["dz"])
        ax.plot(z, traj[:, 0], "--", color=colors[name], lw=2, label=name)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("x (cm)")
    ax.set_title("x vs z")
    ax.legend()
    ax.grid(True)
    
    # y vs z
    ax = axes[0, 1]
    ax.plot(z, traj_true[:, 1], "k-", lw=3, label="Ground Truth")
    for name, model in models.items():
        model.eval()
        traj = rollout_model(model, x0, steps, config["dz"])
        ax.plot(z, traj[:, 1], "--", color=colors[name], lw=2, label=name)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("y vs z")
    ax.legend()
    ax.grid(True)
    
    # tx vs z
    ax = axes[1, 0]
    ax.plot(z, traj_true[:, 3], "k-", lw=3, label="Ground Truth")
    for name, model in models.items():
        model.eval()
        traj = rollout_model(model, x0, steps, config["dz"])
        ax.plot(z, traj[:, 3], "--", color=colors[name], lw=2, label=name)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("tx")
    ax.set_title("tx vs z")
    ax.legend()
    ax.grid(True)
    
    # Position error vs z
    ax = axes[1, 1]
    for name, model in models.items():
        model.eval()
        traj = rollout_model(model, x0, steps, config["dz"])
        pos_error = np.sqrt((traj[:, 0] - traj_true[:, 0])**2 + 
                          (traj[:, 1] - traj_true[:, 1])**2)
        ax.semilogy(z, pos_error, color=colors[name], lw=2, label=name)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("Position Error (cm)")
    ax.set_title("Position Error vs z")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "track_comparison.png"), dpi=150)
    plt.close()


def plot_3d_track(models, gt, x0, config, output_dir):
    """Plot 3D trajectory comparison."""
    device = x0.device
    steps = int(config["z_end"] / config["dz"])
    
    # Ground truth trajectory
    _, traj_true = gt.integrate(x0, config["z_end"], config["dz"])
    traj_true = traj_true.cpu().numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth
    ax.plot(traj_true[:, 0], traj_true[:, 1], traj_true[:, 2], 
            "k-", lw=3, label="Ground Truth")
    
    colors = {"RK-PINN": "red", "Large-RK": "green"}
    for name, model in models.items():
        model.eval()
        traj = rollout_model(model, x0, steps, config["dz"])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
               "--", color=colors[name], lw=2, label=name)
    
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("3D Track Comparison")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "track_3d.png"), dpi=150)
    plt.close()


def rollout_model(model, x0, steps, dz):
    """Rollout a model and return numpy trajectory."""
    device = x0.device
    RK4_b = torch.tensor([1/6, 1/3, 1/3, 1/6], device=device, dtype=x0.dtype)
    
    trajectory = [x0.cpu().numpy()]
    state = x0.unsqueeze(0)
    
    for _ in range(steps):
        with torch.no_grad():
            K = model(state)  # [1, 4, 6]
            delta = dz * torch.einsum("s,bsd->bd", RK4_b, K)
            state = state + delta
            
            # Update z coordinate
            new_state = state.clone()
            new_state[:, 2] = new_state[:, 2] + dz
            state = new_state
            
        trajectory.append(state.squeeze(0).cpu().numpy())
    
    return np.array(trajectory)


def run_dz_sweep(output_dir: str, field, quick: bool = False):
    """Run experiments with different dz values."""
    base_config = {
        "z_end": 500.0,
        "n_train_tracks": 10 if quick else 20,
        "n_test_tracks": 3 if quick else 5,
        "hidden_dim": 64 if quick else 128,
        "num_layers": 3 if quick else 4,
        "max_epochs": 2000 if quick else 15000,
        "patience": 20 if quick else 50,
    }
    
    dz_values = [5.0, 10.0, 20.0] if quick else [2.0, 5.0, 10.0, 20.0, 50.0]
    
    all_results = []
    
    for dz in dz_values:
        print(f"\n{'='*60}")
        print(f"Running dz = {dz} cm")
        print(f"{'='*60}")
        
        config = base_config.copy()
        config["dz"] = dz
        
        exp_dir = os.path.join(output_dir, f"dz_{dz}")
        results, _ = run_experiment(config, exp_dir, field, verbose=True)
        
        for name, res in results.items():
            all_results.append({
                "dz": dz,
                "method": name,
                **{k: v for k, v in res.items() if k != "loss_history"}
            })
    
    # Save aggregate results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "dz_sweep_results.csv"), index=False)
    
    # Plot sweep results
    plot_dz_sweep(df, output_dir)
    
    print(f"\n{'='*60}")
    print("DZ SWEEP COMPLETE")
    print(f"{'='*60}")
    print(df.to_string())


def plot_dz_sweep(df, output_dir):
    """Plot results of dz sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = df["method"].unique()
    colors = {"RK-PINN": "red", "Large-RK": "green"}
    
    for method in methods:
        method_df = df[df["method"] == method]
        
        # Position error
        axes[0].semilogy(method_df["dz"], method_df["pos_mean"],
                        "o-", color=colors.get(method, "blue"), label=method, lw=2)
        
        # Training time
        axes[1].plot(method_df["dz"], method_df["train_time"],
                    "o-", color=colors.get(method, "blue"), label=method, lw=2)
        
        # Rollout time
        axes[2].plot(method_df["dz"], method_df["rollout_time"] * 1000,
                    "o-", color=colors.get(method, "blue"), label=method, lw=2)
    
    axes[0].set_xlabel("dz (cm)")
    axes[0].set_ylabel("Mean Position Error (cm)")
    axes[0].set_title("Accuracy vs Step Size")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel("dz (cm)")
    axes[1].set_ylabel("Train Time (s)")
    axes[1].set_title("Training Time vs Step Size")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_xlabel("dz (cm)")
    axes[2].set_ylabel("Rollout Time (ms)")
    axes[2].set_title("Inference Time vs Step Size")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dz_sweep_summary.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run LHCb track extrapolation experiments")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick experiments with fewer epochs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="Results/LHCb",
        help="Output directory for results"
    )
    parser.add_argument(
        "--field", type=str, default="quadratic", choices=["quadratic", "lhcb"],
        help="Magnetic field type"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run dz parameter sweep"
    )
    parser.add_argument(
        "--field-file", type=str, default=None,
        help="Path to LHCb field data file (for --field lhcb)"
    )
    
    args = parser.parse_args()
    
    # Setup magnetic field
    if args.field == "lhcb":
        if not HAS_LHCB_FIELD:
            print("Error: LHCb_Field not available. Using quadratic field instead.")
            field = Quadratic_Field(B0=1e-3)
        elif args.field_file:
            field = LHCb_Field(args.field_file)
        else:
            print("Warning: No field file specified for LHCb. Using default or quadratic.")
            try:
                field = LHCb_Field()
            except:
                field = Quadratic_Field(B0=1e-3)
    else:
        field = Quadratic_Field(B0=1e-3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    print(f"Output directory: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Field type: {type(field).__name__}")
    print(f"Device: {DEVICE}")
    
    if args.sweep:
        run_dz_sweep(output_dir, field, quick=args.quick)
    else:
        config = {
            "z_end": 500.0,
            "dz": 10.0,
            "n_train_tracks": 10 if args.quick else 20,
            "n_test_tracks": 3 if args.quick else 5,
            "hidden_dim": 64 if args.quick else 128,
            "num_layers": 3 if args.quick else 4,
            "max_epochs": 2000 if args.quick else 15000,
            "patience": 20 if args.quick else 50,
        }
        run_experiment(config, output_dir, field, verbose=True)


if __name__ == "__main__":
    main()
