import os
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.fft import rfft, rfftfreq

from modularisation.vdp_utils import vdp, rk_apply
from modularisation.model_utils import (
    NeuralRK,
    generate_training_data,
    generate_training_data_all_ics,
    rollout_neural_model,
    rollout_rk4,
)


def evaluate_final_error(
    model: NeuralRK,
    m: int,
    butcher: torch.Tensor,
    f: callable,
    x0_set: torch.Tensor,
    t_end: float,
    dt: float,
) -> torch.Tensor:
    """
    Compute final-time L2 errors between a NeuralRK model and classical RK integration.

    Args:
        model: Trained NeuralRK instance.
        m: Number of RK stages.
        butcher: Butcher tableau for classical RK.
        f: Vector field function (e.g., Van der Pol system).
        x0_set: Tensor of shape (batch_size, state_dim), initial conditions.
        t_end: Final time for integration.
        dt: Time step size.

    Returns:
        Tensor of shape (batch_size,) with L2 errors at final time.
    """
    model.eval()
    steps = int(t_end / dt)
    errors = []

    for x0 in x0_set:
        x0 = x0.to(next(model.parameters()).device)

        traj_nn = rollout_neural_model(model, x0, steps, dt)
        traj_rk = rollout_rk4(x0, steps, dt, m, butcher, f)

        err = torch.norm((traj_nn[-1] - traj_rk[-1]) / traj_nn[-1], p=2)
        errors.append(err.item())

    return torch.tensor(errors)


def sample_initial_conditions(n: int) -> torch.Tensor:
    """
    Generate n random initial conditions uniformly in [-2, 2]^2.

    Args:
        n: Number of samples.

    Returns:
        Tensor of shape (n, 2) with sampled initial conditions.
    """
    return torch.stack([
        torch.tensor([
            4 * torch.rand(1).item() - 2,
            4 * torch.rand(1).item() - 2,
        ])
        for _ in range(n)
    ])


def _save_current_figure(name: str) -> None:
    """
    Save the current matplotlib figure to the current working directory with a timestamped filename.
    """
    fname = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(os.getcwd(), fname)
    try:
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved figure to {path}")
    except Exception as ex:
        print(f"Failed to save figure {fname}: {ex}")


def full_mode_analysis(
    traj_rk: torch.Tensor,
    traj_nn: torch.Tensor,
    steps: int,
    dt: float,
) -> None:
    """
    Perform full error analysis between classical RK and NeuralRK trajectories.

    Calculates time-series errors, component errors, frequency decomposition,
    and visualizes results in various subplots.

    Args:
        traj_rk: Tensor of shape (T+1, state_dim) from classical RK4.
        traj_nn: Tensor of same shape from NeuralRK.
        steps: Number of time steps (T).
        dt: Time step size.
    """
    t = torch.arange(steps + 1) * dt
    error_l2 = torch.norm(traj_rk - traj_nn, dim=1)
    error_x1 = torch.abs(traj_rk[:, 0] - traj_nn[:, 0])
    error_x2 = torch.abs(traj_rk[:, 1] - traj_nn[:, 1])

    freqs = rfftfreq(steps + 1, dt)
    fft_err_x1 = np.abs(rfft(error_x1.cpu().numpy()))
    fft_err_x2 = np.abs(rfft(error_x2.cpu().numpy()))

    lin_coef_x1 = np.polyfit(freqs, fft_err_x1, 1)
    lin_coef_x2 = np.polyfit(freqs, fft_err_x2, 1)
    fit_x1 = np.polyval(lin_coef_x1, freqs)
    fit_x2 = np.polyval(lin_coef_x2, freqs)

    cyc_x1 = fft_err_x1 - fit_x1
    cyc_x2 = fft_err_x2 - fit_x2

    # Plotting
    plt.figure(figsize=(22, 12))
    # Phase space
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(traj_rk[:, 0], traj_rk[:, 1], label="Classical RK4", lw=2)
    ax1.plot(traj_nn[:, 0], traj_nn[:, 1], '--', label="Neural RK", lw=2)
    ax1.set(title="Phase Space (x₁ vs x₂)", xlabel="x₁", ylabel="x₂"); ax1.legend(); ax1.grid()

    # Time series
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(t, traj_rk[:, 0], label="x₁ RK4")
    ax2.plot(t, traj_nn[:, 0], '--', label="x₁ NN")
    ax2.plot(t, traj_rk[:, 1], label="x₂ RK4")
    ax2.plot(t, traj_nn[:, 1], '--', label="x₂ NN")
    ax2.set(title="Time Series", xlabel="Time", ylabel="State"); ax2.legend(); ax2.grid()

    # Pointwise error
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(t, error_l2)
    ax3.set(title="Pointwise ℓ² Error: ||x_RK4 - x_NN||", xlabel="Time", ylabel="Error"); ax3.grid()

    # Component errors
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(t, error_x1, label="|Δx₁|")
    ax4.plot(t, error_x2, label="|Δx₂|")
    ax4.set(title="Component-wise Error", xlabel="Time", ylabel="Abs Error"); ax4.legend(); ax4.grid()

    # Error spectrum
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(freqs, fft_err_x1, label="FFT |x₁ error|")
    ax5.plot(freqs, fft_err_x2, label="FFT |x₂ error|")
    ax5.set(title="Error Spectrum (FFT)", xlabel="Frequency [Hz]", ylabel="Magnitude"); ax5.legend(); ax5.grid()

    # Linear drift
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(freqs, fit_x1, '--', label="Drift x₁")
    ax6.plot(freqs, fit_x2, '--', label="Drift x₂")
    ax6.set(title="Linear Drift in Frequency Domain", xlabel="Freq [Hz]", ylabel="Fit Magnitude"); ax6.legend(); ax6.grid()

    # Cyclical residuals
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(freqs, cyc_x1, label="Cyclical x₁")
    ax7.plot(freqs, cyc_x2, label="Cyclical x₂")
    ax7.set(title="Cyclical Error (Residuals)", xlabel="Freq [Hz]", ylabel="Residual"); ax7.legend(); ax7.grid()

    plt.tight_layout()
    _save_current_figure("full_mode_analysis")
    plt.show()
    plt.close()




def plot_mean_and_max_errors_separately(df: pd.DataFrame) -> None:
    """
    Plot 3D surfaces of mean and max final-time errors
    versus hidden dimension and number of layers.

    Args:
        df: DataFrame with columns ['model','mean_error','max_error'] containing results.
    """
    df['hidden_dim'] = df['model'].str.extract(r'hd(\d+)').astype(float)
    df['num_layers'] = df['model'].str.extract(r'layers(\d+)').astype(float)
    hidden_dims = sorted(df['hidden_dim'].unique())
    num_layers = sorted(df['num_layers'].unique())
    X, Y = np.meshgrid(hidden_dims, num_layers)

    Z_mean = df.pivot_table(index='num_layers', columns='hidden_dim', values='mean_error').values
    Z_max  = df.pivot_table(index='num_layers', columns='hidden_dim', values='max_error').values

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_mean, edgecolor='k', linewidth=0.4, alpha=0.9)
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.1).set_label("Mean Final-Time Error")
    ax1.set(title="Mean Error vs Hidden Dim & Layers", xlabel="Hidden Dim", ylabel="Num Layers", zlabel="Mean Error")
    ax1.view_init(elev=30, azim=135); ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_max, edgecolor='k', linewidth=0.4, alpha=0.9)
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.1).set_label("Max Final-Time Error")
    ax2.set(title="Max Error vs Hidden Dim & Layers", xlabel="Hidden Dim", ylabel="Num Layers", zlabel="Max Error")
    ax2.view_init(elev=30, azim=135); ax2.grid()

    plt.tight_layout()
    # save figure
    try:
        fname = f"mean_max_errors_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(os.getcwd(), fname)
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved figure to {path}")
    except Exception as ex:
        print(f"Failed to save figure mean_max_errors: {ex}")
    plt.show()
    plt.close(fig)


def time_rollout(
    model: NeuralRK,
    x0: torch.Tensor,
    steps: int,
    dt: float,
) -> float:
    """
    Measure time to perform a NeuralRK rollout from an initial state.

    Args:
        model: NeuralRK instance.
        x0: Initial state tensor.
        steps: Number of time steps.
        dt: Step size.

    Returns:
        Rollout duration in seconds.
    """
    start = time.time()
    rollout_neural_model(model, x0, steps, dt)
    return time.time() - start


def evaluate_and_time_saved_models(
    steps: int,
    m: int,
    butcher: torch.Tensor,
    f: callable,
    x0_eval: torch.Tensor,
    x0: torch.Tensor,
    t_end: float,
    dt: float,
    device: torch.device,
) -> pd.DataFrame:
    """
    Load, evaluate final-error, and time rollout for all saved NeuralRK models.

    Args:
        steps: Number of integration steps.
        m: Number of RK stages.
        butcher: Butcher tableau tensor.
        f: Vector field function.
        x0_eval: Tensor of initial conditions for error eval.
        x0: Single initial state for timing.
        dt: Step size.
        device: Compute device.

    Returns:
        DataFrame with columns ['model','mean_error','max_error','rollout_time','hidden_dim','num_layers','dt'].
    """
    results = []
    model_dir = r"C:\Users\GeorgeWilliam\Documents\GitHub\TrackExtrapolation\RK_PINNs\Results\VDP\Models"

    for fname in os.listdir(model_dir):
        if (not fname.endswith("RK4.pt")) and (not fname.endswith("RK4_PHYS.pt")):
            continue
        path = os.path.join(model_dir, fname)
        try:
            ckpt = torch.load(path, map_location=device)
            info = ckpt.get("model_info", {})
            model = NeuralRK(
                input_dim=info.get("input_dim", 2),
                hidden_dim=info.get("hidden_dim", 64),
                num_layers=info.get("num_layers", 3),
                output_dim=info.get("output_dim", 2),
                butcher=butcher,
            ).to(device)
            model.dt = info.get("dt", dt)
            model.load_state_dict(ckpt["model_state_dict"]);

            rollout_time = time_rollout(model, x0, steps, model.dt)
            errs = evaluate_final_error(model, m, butcher, f, x0_eval, t_end, model.dt)
            results.append({
                "model": fname,
                "mean_error": errs.mean().item(),
                "max_error": errs.max().item(),
                "rollout_time": rollout_time,
            })
        except Exception as ex:
            print(f"Failed {fname}: {ex}")

    df = pd.DataFrame(results)
    if not df.empty:
        df['hidden_dim'] = df['model'].str.extract(r'hd(\d+)').astype(int)
        df['num_layers'] = df['model'].str.extract(r'layers(\d+)').astype(int)
        df['dt'] = df['model'].str.extract(r'dt([0-9.]+)').astype(float)
        df.to_csv("rk_nn_saved_models_evaluation_with_timing.csv", index=False)
        print("Saved results to CSV.")
    else:
        print("No models evaluated.")
    return df


def plot_rollout_time(df: pd.DataFrame) -> None:
    """
    Plot rollout time as a 3D surface over hidden dimensions and layers.

    Args:
        df: DataFrame with 'rollout_time', 'model' columns.
    """
    df['hidden_dim'] = df['model'].str.extract(r'hd(\d+)').astype(float)
    df['num_layers'] = df['model'].str.extract(r'layers(\d+)').astype(float)
    hidden_dims = sorted(df['hidden_dim'].unique())
    num_layers = sorted(df['num_layers'].unique())
    X, Y = np.meshgrid(hidden_dims, num_layers)
    Z = df.pivot_table(index='num_layers', columns='hidden_dim', values='rollout_time').values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, edgecolor='k', linewidth=0.4, alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1).set_label("Rollout Time (s)")
    ax.set(title="Rollout Time vs Hidden Dim & Layers", xlabel="Hidden Dim", ylabel="Num Layers", zlabel="Time (s)")
    ax.view_init(elev=30, azim=135); ax.grid(True)
    plt.tight_layout()
    try:
        fname = f"rollout_time_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(os.getcwd(), fname)
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved figure to {path}")
    except Exception as ex:
        print(f"Failed to save figure rollout_time: {ex}")
    plt.show()
    plt.close(fig)


def plot_training_data(X: torch.Tensor, K: torch.Tensor) -> None:
    """
    Scatter plot of initial conditions and quiver of mean RK stage vectors.

    Args:
        X: Tensor (n_points, 2) of initial conditions.
        K: Tensor (n_points, stages, 2) of RK stage vectors.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), alpha=0.5, label='Initial Conditions')
    plt.quiver(
        X[:, 0].cpu(), X[:, 1].cpu(),
        K[:, :, 0].cpu().mean(dim=1), K[:, :, 1].cpu().mean(dim=1),
        alpha=0.5, label='RK Stages'
    )
    plt.title("Training Data: Initial Conditions and RK Stages")
    plt.xlabel("x₁"); plt.ylabel("x₂"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    _save_current_figure("training_data")
    plt.show()
    plt.close()


def rk_timmer(
    steps: int,
    m: int,
    Ms: list,
    model: NeuralRK,
    butcher: torch.Tensor,
    x0_eval: torch.Tensor,
    x0: torch.Tensor,
    t_end: float,
    dt: float,
    f: callable,
) -> pd.DataFrame:
    """
    Time RK4 rollouts for multiple M values and compare errors.

    Args:
        steps: Number of time steps.
        Ms: List of M (RK stage counts) to test.
        model: NeuralRK model for error comparison.
        butcher: Butcher tableau.
        x0_eval: Initial conditions tensor for error eval.
        x0: Single initial condition for timing.
        dt: Step size.
        f: Vector field.

    Returns:
        DataFrame with columns ['method','mean_error','max_error','rollout_time'].
    """
    records = []
    for m_val in Ms:
        print(f"Evaluating RK4 with M={m_val}")
        start = time.time()
        traj_rk = rollout_rk4(x0, int(t_end / dt), dt, m_val, butcher, f)
        duration = time.time() - start
        true_rk = rollout_rk4(x0, int(t_end / dt), dt, m, butcher, f)
        errs = torch.norm(traj_rk - true_rk, dim=1)
        mean_error = errs.mean().item()
        max_error = errs.max().item()
        print(f"Mean Error: {mean_error}, Max Error: {max_error}, Duration: {duration:.4f}s")
        records.append({
            'method': f'RK4 (M={m_val})',
            'mean_error': mean_error,
            'max_error': max_error,
            'rollout_time': duration,
        })
       
    return pd.DataFrame(records)


def plot_accuracy_and_timing_comparison(
    df_nn: pd.DataFrame,
    df_rk: pd.DataFrame,
) -> None:
    """
    Plot bar comparisons of rollout time, mean error, and max error
    for NeuralRK and classical RK methods.

    Args:
        df_nn: DataFrame for NeuralRK with columns ['model','mean_error','max_error','rollout_time'].
        df_rk: DataFrame for RK with ['method','mean_error','max_error','rollout_time'].
    """
    df_nn_plot = df_nn.rename(columns={'model': 'method'})
    df_nn_plot['source'] = 'NeuralRK'
    df_rk['source'] = 'RK'
    combined = pd.concat([df_nn_plot, df_rk], ignore_index=True)
    combined.sort_values('rollout_time', inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].barh(combined['method'], combined['rollout_time']); axes[0].set(title="Rollout Time (s)", xlabel="Time (s)"); axes[0].grid(True)
    axes[1].barh(combined['method'], combined['mean_error']); axes[1].set(title="Mean Final-Time Error", xlabel="Error"); axes[1].grid(True)
    axes[2].barh(combined['method'], combined['max_error']); axes[2].set(title="Max Final-Time Error", xlabel="Error"); axes[2].grid(True)
    fig.suptitle("NeuralRK vs Classical RK Comparison", fontsize=16)
    plt.tight_layout()
    try:
        fname = f"accuracy_timing_comparison_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(os.getcwd(), fname)
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved figure to {path}")
    except Exception as ex:
        print(f"Failed to save figure accuracy_timing_comparison: {ex}")
    plt.show()
    plt.close(fig)
