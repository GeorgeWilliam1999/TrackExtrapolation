#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

##############################################################################
# 1) Runge-Kutta class for reference solutions
##############################################################################

class ButcherTableRK:
    """
    Holds the coefficients (A, b, c) for a general s-stage Runge-Kutta method.
    Provides a method 'rk_step' for a single step of that method.
    """
    def __init__(self, A, b, c):
        self.A = torch.tensor(A, dtype=torch.float64)
        self.b = torch.tensor(b, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)
        self.s = len(b)
        # Basic validation
        assert len(c) == self.s, "Mismatch in length of c vs b"
        assert len(A) == self.s, "Number of rows in A must match s"
        for row in A:
            assert len(row) == self.s, "A must be s x s"

    def rk_step(self, func, t, y, dt, mu=1.0):
        """
        Performs one step of this Runge-Kutta method from time t to t+dt,
        starting with state y (Tensor).
        
        func(t, y, mu) => dy/dt as a Tensor of the same shape as y.
        """
        A = self.A.to(y.device, y.dtype)
        b = self.b.to(y.device, y.dtype)
        c = self.c.to(y.device, y.dtype)

        k_list = []
        for i in range(self.s):
            ti = t + c[i].item() * dt
            accum = 0.0
            for j in range(i):
                accum += A[i, j].item() * k_list[j]
            yi = y + dt * accum
            ki = func(ti, yi, mu)
            k_list.append(ki)

        next_y = y + dt * sum(b[i].item() * k_list[i] for i in range(self.s))
        return next_y

##############################################################################
# 2) PINN_Experiment class with on-the-fly reference generation and user inputs
##############################################################################

class PINN_Experiment:
    """
    - ODE: Van der Pol
    - A "true" RK solver with a small dt (used as ground truth) for each sample
      when computing the cost function (on-the-fly).
    - A NN that approximates f(t,y) => dy/dt.
    - Toggle for 'use_ode_loss': 
       * True => RK-PINN (final-step mismatch + ODE residual)
       * False => pure NN mode (final-step mismatch only).

    User parameters:
       domain_min, domain_max: sampling region for initial states.
       mu: Van der Pol parameter.
    """
    def __init__(self, butcher_table, device="cpu", domain_min=-10.0, domain_max=10.0, mu=1.0):
        """
        domain_min, domain_max => define the region from which y0 is sampled.
        mu => Van der Pol parameter for the ODE.
        """
        self.butcher_table = butcher_table
        self.device = torch.device(device)
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.mu = mu
        self.model = None

    def van_der_pol(self, t, y, mu=None):
        """
        Van der Pol in first-order form:
          dy1/dt = y2,
          dy2/dt = mu*(1 - y1^2)*y2 - y1.
        """
        if mu is None:
            mu = self.mu
        dydt = torch.zeros_like(y)
        dydt[0] = y[1]
        dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
        return dydt

    def rk_many_small_steps(self, t, y, big_dt, substeps=10):
        """
        For a single large step big_dt, do multiple small RK steps
        for a high-accuracy final state. Called inside the cost function.
        """
        small_dt = big_dt / substeps
        y_approx = y.clone()
        for _ in range(substeps):
            y_approx = self.butcher_table.rk_step(
                func=self.van_der_pol,
                t=t,
                y=y_approx,
                dt=small_dt,
                mu=self.mu
            )
            t += small_dt
        return y_approx

    def create_model(self, hidden_dim=32, num_layers=2):
        """
        The neural network takes input (t, y1, y2) and outputs (dy1/dt, dy2/dt).
        """
        class DerivativeNN(nn.Module):
            def __init__(self, in_dim, hid_dim, layers):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(in_dim, hid_dim))
                for _ in range(layers-1):
                    self.layers.append(nn.Linear(hid_dim, hid_dim))
                self.out = nn.Linear(hid_dim, 2)
                self.activation = nn.Tanh()
                # Xavier initialization
                for p in self.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

            def forward(self, t, y):
                # t: (batch_size, 1), y: (batch_size, 2)
                inp = torch.cat([t, y], dim=-1)  # => (batch_size, 3)
                x = inp
                for layer in self.layers:
                    x = self.activation(layer(x))
                return self.out(x)

        # Ensure the model uses double precision.
        self.model = DerivativeNN(in_dim=3, hid_dim=hidden_dim, layers=num_layers).to(self.device).double()
        return self.model

    def nn_next_state(self, t, y, big_dt):
        """
        One big step:
          y_{n+1} = y_n + big_dt * NN(t, y_n).
        """
        f_nn = self.model(t, y)
        return y + big_dt * f_nn

    def compute_batch_loss(self, batch_size, big_dt, substeps, alpha, use_ode_loss):
        """
        1) For each sample in the batch:
            - Sample random y0 in [domain_min, domain_max]^2.
            - Set t0=0.
            - Compute "true" y_next via many small RK steps.
            - Compute NN prediction.
        2) step_loss = mean((y_pred - y_true)^2).
        3) If use_ode_loss, add residual = mean(|f_nn - f_true|^2).
        4) Total loss = step_loss + alpha*residual_loss.
        """
        t0 = torch.zeros(batch_size, 1, device=self.device, dtype=torch.float64)
        y0 = (self.domain_max - self.domain_min) * torch.rand(batch_size, 2, device=self.device, dtype=torch.float64) + self.domain_min

        y_true_list = []
        for i in range(batch_size):
            y_ref = self.rk_many_small_steps(
                t=0.0,
                y=y0[i],
                big_dt=big_dt,
                substeps=substeps
            )
            y_true_list.append(y_ref)
        y_true = torch.stack(y_true_list, dim=0)

        y_pred = self.nn_next_state(t0, y0, big_dt)
        step_loss = torch.mean((y_pred - y_true)**2)

        residual_loss = 0.0
        if use_ode_loss:
            f_nn = self.model(t0, y0)
            with torch.no_grad():
                f_true_list = []
                for i in range(batch_size):
                    f_t = self.van_der_pol(0.0, y0[i])
                    f_true_list.append(f_t)
                f_true = torch.stack(f_true_list, dim=0)
            residual_loss = torch.mean((f_nn - f_true)**2)

        total_loss = step_loss + alpha * residual_loss
        return total_loss, step_loss.item(), residual_loss.item() if use_ode_loss else 0.0

    def train_model_online(
        self,
        big_dt=0.5,
        substeps=10,
        batch_size=32,
        lr=1e-3,
        epochs=2000,
        alpha=1.0,
        use_ode_loss=True,
        clip_grad_norm=0.0
    ):
        """
        Online training:
          - At each epoch, compute batch loss with fresh random samples.
          - Backpropagate.

        TROUBLESHOOTING:
          1) Reduce domain (domain_min/domain_max) if values blow up.
          2) Decrease big_dt (e.g., 0.1 or 0.2).
          3) Lower lr (e.g., 1e-4 or 1e-5).
          4) Use gradient clipping by setting clip_grad_norm > 0.
        """
        if self.model is None:
            raise ValueError("No model. Call create_model() first.")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        mode_str = "RK-PINN" if use_ode_loss else "NN"

        for epoch in range(1, epochs+1):
            optimizer.zero_grad()
            loss, step_l, res_l = self.compute_batch_loss(
                batch_size=batch_size,
                big_dt=big_dt,
                substeps=substeps,
                alpha=alpha,
                use_ode_loss=use_ode_loss
            )
            loss.backward()
            if clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            if epoch % 100 == 0:
                print(f"[{mode_str}] epoch {epoch:4d} | loss={loss.item():.4e} | step={step_l:.4e} | ODE={res_l:.4e}")

    def nn_integrate_to_time(self, y0, t0, t_end, dt):
        """
        Repeated large-step integration with the trained NN from t0 to t_end.
        """
        y = y0.clone().to(self.device)
        t = t0
        while t < t_end - 1e-12:
            leftover = t_end - t
            step_use = dt if leftover > dt else leftover
            y = self.nn_next_state(
                torch.tensor([t], device=self.device, dtype=torch.float64).unsqueeze(0),
                y.unsqueeze(0),
                step_use
            ).squeeze(0)
            t += step_use
        return y

    def final_time_error(self, y0, t0, t_end, dt, ref_steps=1000):
        """
        Compute final-time error between reference solution (using many small RK steps)
        and the NN integrated solution.
        """
        y_true = self.integrate_to_time(y0, t0, t_end, ref_steps)
        y_pred = self.nn_integrate_to_time(y0, t0, t_end, dt)
        return torch.norm(y_true - y_pred, p=2).item()

    def integrate_to_time(self, y0, t0, t_end, steps):
        dt = (t_end - t0) / steps
        y = y0.clone().to(self.device)
        t = t0
        for _ in range(steps):
            y = self.butcher_table.rk_step(
                func=self.van_der_pol,
                t=t,
                y=y,
                dt=dt,
                mu=self.mu
            )
            t += dt
        return y

##############################################################################
# 3) Functions for running sweeps and collecting results
##############################################################################

def run_sweeps_and_collect_results(
    rk4_table,
    dt_list,
    hidden_dim_list,
    num_layers_list,
    use_ode_loss,
    epochs=500,
    substeps=10,
    batch_size=32,
    alpha=1.0,
    device="cpu",
    T_end=5.0,
    ref_steps=500,
    domain_min=-2.0,
    domain_max=2.0,
    mu=1.0
):
    """
    For each combination of (dt, hidden_dim, num_layers), train a model and
    evaluate the final-time error over [0, T_end].
    Returns a list of dicts with the results.
    """
    results = []
    mode_str = "RK-PINN" if use_ode_loss else "NN"
    for dt in dt_list:
        for hd in hidden_dim_list:
            for nl in num_layers_list:
                experiment = PINN_Experiment(
                    rk4_table, device=device,
                    domain_min=domain_min, domain_max=domain_max, mu=mu
                )
                experiment.create_model(hidden_dim=hd, num_layers=nl)
                print(f"=== Training {mode_str}, dt={dt}, hd={hd}, nl={nl} ===")
                experiment.train_model_online(
                    big_dt=dt,
                    substeps=substeps,
                    batch_size=batch_size,
                    lr=1e-4,
                    epochs=epochs,
                    alpha=alpha,
                    use_ode_loss=use_ode_loss,
                    clip_grad_norm=1.0
                )
                # Evaluate final-time error from t=0 to T_end
                y0_test = torch.tensor([2.0, 0.0], dtype=torch.float64)
                error = experiment.final_time_error(y0_test, 0.0, T_end, dt, ref_steps)
                row = {
                    "Mode": mode_str,
                    "dt": dt,
                    "HiddenDim": hd,
                    "NumLayers": nl,
                    "FinalTimeError": error
                }
                results.append(row)
                print(f"[{mode_str}] dt={dt}, hd={hd}, nl={nl} => error={error:.3e}")
    return results

def write_results_to_csv(results, csv_path):
    """
    Write results (list of dicts) to a CSV file.
    """
    if not results:
        print("No results to write.")
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

##############################################################################
# 4) Plotting with Plotly
##############################################################################

def plot_results(results):
    """
    Generate interactive Plotly plots for:
      - Final-time error vs. dt (for fixed HiddenDim=32, NumLayers=2)
      - Final-time error vs. HiddenDim (for fixed dt=0.5, NumLayers=2)
      - Final-time error vs. NumLayers (for fixed dt=0.5, HiddenDim=32)
      - A 3D surface: error as a function of HiddenDim and NumLayers (for fixed dt=0.5)
    Also generate log-log plots.
    """
    df = pd.DataFrame(results)

    # Plot error vs. dt for fixed HiddenDim=32, NumLayers=2
    df_dt = df[(df["HiddenDim"] == 32) & (df["NumLayers"] == 2)]
    fig_dt = px.scatter(df_dt, x="dt", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. dt (HiddenDim=32, NumLayers=2)")
    fig_dt.update_traces(mode="lines+markers")
    fig_dt.show()

    fig_dt_log = px.scatter(df_dt, x="dt", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. dt - Log-Log (HiddenDim=32, NumLayers=2)",
                        log_x=True, log_y=True)
    fig_dt_log.update_traces(mode="lines+markers")
    fig_dt_log.show()

    # Plot error vs. HiddenDim for fixed dt=0.5, NumLayers=2
    df_hd = df[(df["dt"] == 0.5) & (df["NumLayers"] == 2)]
    fig_hd = px.scatter(df_hd, x="HiddenDim", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. HiddenDim (dt=0.5, NumLayers=2)")
    fig_hd.update_traces(mode="lines+markers")
    fig_hd.show()

    fig_hd_log = px.scatter(df_hd, x="HiddenDim", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. HiddenDim - Log-Log (dt=0.5, NumLayers=2)",
                        log_x=True, log_y=True)
    fig_hd_log.update_traces(mode="lines+markers")
    fig_hd_log.show()

    # Plot error vs. NumLayers for fixed dt=0.5, HiddenDim=32
    df_nl = df[(df["dt"] == 0.5) & (df["HiddenDim"] == 32)]
    fig_nl = px.scatter(df_nl, x="NumLayers", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. NumLayers (dt=0.5, HiddenDim=32)")
    fig_nl.update_traces(mode="lines+markers")
    fig_nl.show()

    fig_nl_log = px.scatter(df_nl, x="NumLayers", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. NumLayers - Log-Log (dt=0.5, HiddenDim=32)",
                        log_x=True, log_y=True)
    fig_nl_log.update_traces(mode="lines+markers")
    fig_nl_log.show()

    # 3D Surface: error as a function of HiddenDim and NumLayers for fixed dt=0.5
    df_3d = df[(df["dt"] == 0.5)]
    fig_3d = px.scatter_3d(df_3d, x="HiddenDim", y="NumLayers", z="FinalTimeError",
                           color="Mode",
                           title="3D Scatter: Final-time Error vs. HiddenDim and NumLayers (dt=0.5)")
    fig_3d.show()

def save_plots_to_html(results, html_path="results/interactive_plots.html"):
    """
    Combine Plotly figures into an HTML file.
    """
    import plotly.io as pio
    df = pd.DataFrame(results)

    figs = []
    df_dt = df[(df["HiddenDim"] == 32) & (df["NumLayers"] == 2)]
    fig_dt = px.scatter(df_dt, x="dt", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. dt (HiddenDim=32, NumLayers=2)")
    fig_dt.update_traces(mode="lines+markers")
    figs.append(fig_dt)

    df_hd = df[(df["dt"] == 0.5) & (df["NumLayers"] == 2)]
    fig_hd = px.scatter(df_hd, x="HiddenDim", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. HiddenDim (dt=0.5, NumLayers=2)")
    fig_hd.update_traces(mode="lines+markers")
    figs.append(fig_hd)

    df_nl = df[(df["dt"] == 0.5) & (df["HiddenDim"] == 32)]
    fig_nl = px.scatter(df_nl, x="NumLayers", y="FinalTimeError", color="Mode",
                        title="Final-time Error vs. NumLayers (dt=0.5, HiddenDim=32)")
    fig_nl.update_traces(mode="lines+markers")
    figs.append(fig_nl)

    html_str = ""
    for fig in figs:
        html_str += pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html_str)
    print(f"Interactive plots saved to {html_path}")

##############################################################################
# 5) Main function: run sweeps, save table, and generate Plotly plots
##############################################################################

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Define classical 4-stage RK for reference solutions
    c = [0.0, 0.5, 0.5, 1.0]
    A = [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = [1/6, 1/3, 1/3, 1/6]
    rk4_table = ButcherTableRK(A, b, c)

    # 2) Define parameter sweeps
    dt_list = [0.01, 0.05, 0.1]          # NN large step sizes
    hidden_dim_list = [16, 32, 64]         # hidden layer widths
    num_layers_list = [1, 2, 3]            # number of layers

    # 3) Run experiments in both modes:
    print("=== Running Pure NN sweeps ===")
    results_nn = run_sweeps_and_collect_results(
        rk4_table=rk4_table,
        dt_list=dt_list,
        hidden_dim_list=hidden_dim_list,
        num_layers_list=num_layers_list,
        use_ode_loss=False,   # Pure NN mode
        epochs=500,
        substeps=10,
        batch_size=32,
        alpha=1.0,
        device=device,
        T_end=5.0,
        ref_steps=500,
        domain_min=-2.0,
        domain_max=2.0,
        mu=1.0
    )

    print("\n=== Running RK-PINN sweeps ===")
    results_pinn = run_sweeps_and_collect_results(
        rk4_table=rk4_table,
        dt_list=dt_list,
        hidden_dim_list=hidden_dim_list,
        num_layers_list=num_layers_list,
        use_ode_loss=True,    # RK-PINN mode
        epochs=500,
        substeps=10,
        batch_size=32,
        alpha=1.0,
        device=device,
        T_end=5.0,
        ref_steps=500,
        domain_min=-2.0,
        domain_max=2.0,
        mu=1.0
    )

    all_results = results_nn + results_pinn

    # 4) Save results to CSV
    csv_path = "PINN_VDP_TESTING/RK_PINN_vs_RK_NN/tables/pinn_experiment_results.csv"
    write_results_to_csv(all_results, csv_path)

    # 5) Generate interactive Plotly plots
    plot_results(all_results)
    save_plots_to_html(all_results, html_path="PINN_VDP_TESTING/RK_PINN_vs_RK_NN/plots/interactive_plots.html")

if __name__ == "__main__":
    main()

