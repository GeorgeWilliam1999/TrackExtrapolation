import torch
import torch.nn as nn

import plotly.graph_objects as go
from plotly.subplots import make_subplots

##############################################################################
# 1) Van der Pol ODE and RK4 Solver
##############################################################################

def van_der_pol(t, y, mu=1.0):
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
    return dydt

def rk4_step(func, t, y, dt, mu=1.0):
    k1 = func(t, y, mu)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1, mu)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2, mu)
    k4 = func(t + dt,     y + dt*k3,    mu)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_rk4(func, y0, t0, t_end, N, mu=1.0, device="cpu"):
    dt = (t_end - t0) / N
    times = []
    ys = []
    t = t0
    y = y0.clone().to(device)
    for _ in range(N+1):
        times.append(t)
        ys.append(y.unsqueeze(0))
        y = rk4_step(func, t, y, dt, mu=mu)
        t += dt

    return (torch.tensor(times, device=device),
            torch.cat(ys, dim=0))

##############################################################################
# 2) Neural Network (Predict y_{n+1} from y_n)
##############################################################################

class RK_PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

##############################################################################
# 3) Generate Training Data & Helpers
##############################################################################

def generate_training_data(func, y0, t0, t_end, N, mu=1.0, device="cpu"):
    """
    Integrate ODE with RK4 from t0 to t_end in N steps,
    build pairs (y_n, y_{n+1}).
    """
    _, all_ys = integrate_rk4(func, y0, t0, t_end, N, mu=mu, device=device)
    X = all_ys[:-1]
    Y = all_ys[1:]
    return X, Y

def integrate_nn(model, y0, t0, t_end, N, device="cpu"):
    dt = (t_end - t0) / N
    times = []
    ys = []
    t = t0
    y = y0.clone().to(device).unsqueeze(0)
    for _ in range(N+1):
        times.append(t)
        ys.append(y)
        y = model(y)
        t += dt
    return (torch.tensor(times, device=device),
            torch.cat(ys, dim=0))

def compute_trajectory_mse(model, y0, t0, t_end, N, mu=1.0, device="cpu"):
    """
    Integrate both with RK4 and the NN, returning MSE over the entire trajectory.
    """
    _, y_rk4 = integrate_rk4(van_der_pol, y0, t0, t_end, N, mu=mu, device=device)
    _, y_nn  = integrate_nn(model, y0, t0, t_end, N, device=device)
    return torch.mean((y_rk4 - y_nn)**2).item()

def compute_final_time_error(model, y0, t0, t_end, N, mu=1.0, device="cpu"):
    """
    || y_RK4(t_end) - y_NN(t_end) || (Euclidean distance).
    """
    _, y_rk4 = integrate_rk4(van_der_pol, y0, t0, t_end, N, mu=mu, device=device)
    _, y_nn  = integrate_nn(model, y0, t0, t_end, N, device=device)
    return torch.norm(y_rk4[-1] - y_nn[-1]).item()

##############################################################################
# 4) Training Loop: Track Training Loss & Global Errors for Each IC
##############################################################################

def train_model_multiple_ics(
    model,
    X_train, Y_train,
    ics_for_eval,
    epochs=1000, lr=1e-3,
    t0=0.0, t_end=5.0, N_eval=200, mu=1.0,
    device="cpu"
):
    """
    Train on (X_train, Y_train) to learn y_{n+1} ~ model(y_n).
    After each epoch, compute trajectory MSE for each IC in ics_for_eval.
    Returns (training_losses, global_errors),
      where global_errors[i] is a list of length 'epochs' for i-th IC.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    training_losses = []
    global_errors = {i: [] for i in range(len(ics_for_eval))}

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

        for i, ic in enumerate(ics_for_eval):
            err = compute_trajectory_mse(
                model, ic, t0, t_end, N_eval, mu=mu, device=device
            )
            global_errors[i].append(err)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.6f}, "
                  f"GlobErr(IC0)={global_errors[0][-1]:.6f}")

    return training_losses, global_errors

##############################################################################
# 5) Main Code + Plots
##############################################################################

if __name__ == "__main__":
    # -------------------------------------------
    # 5.1) Choose device (GPU vs. CPU)
    # -------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------------
    # 5.2) Global hyperparameters
    # -------------------------------------------
    t0, t_end = 0.0, 5.0
    mu = 1.0
    N_data = 100       # steps for reference solution in training set
    epochs = 300       # training epochs
    learning_rate = 1e-3
    N_eval = 200       # steps for measuring trajectory error each epoch
    N_eval_final = 100 # steps for final trajectory comparisons

    # "Large" range of hidden dims and layer counts
    hidden_dims = [16, 32, 64]
    layer_counts = [2, 3, 4]

    # We have multiple initial conditions
    ics_train = [
        torch.tensor([2.0, 0.0]),
        torch.tensor([1.0, -1.0])
    ]
    ics_eval = ics_train  # same ICs for final plots & final-time error

    # -------------------------------------------
    # 5.3) Build a combined training set
    # -------------------------------------------
    X_train_list = []
    Y_train_list = []
    for ic in ics_train:
        X_ic, Y_ic = generate_training_data(
            van_der_pol, ic, t0, t_end, N_data, mu=mu, device=device
        )
        X_train_list.append(X_ic)
        Y_train_list.append(Y_ic)
    X_train = torch.cat(X_train_list, dim=0)
    Y_train = torch.cat(Y_train_list, dim=0)

    # -------------------------------------------
    # Create a list of all (hd, nl) combos
    # -------------------------------------------
    configs = []
    for hd in hidden_dims:
        for nl in layer_counts:
            configs.append((hd, nl))
    num_configs = len(configs)

    # We'll store final-time errors for each config and each IC
    # final_time_errors[(hd, nl)] = [err_IC0, err_IC1, ...]
    final_time_errors = {}

    # We'll create one big figure with subplots for the "usual" plots.
    total_rows = 2 * num_configs
    total_cols = 4

    fig = make_subplots(
        rows=total_rows,
        cols=total_cols,
        vertical_spacing=0.02,  # must be <= 1/(rows-1) if many rows
        horizontal_spacing=0.05,
    )

    # ---------------------------------------------------------------
    # 5.4) For each (hidden_dim, num_layers), train & produce subplots
    # ---------------------------------------------------------------
    for config_idx, (hd, nl) in enumerate(configs):
        print("\n=============================================")
        print(f"Training model with hidden_dim={hd}, num_layers={nl}")
        print("=============================================")

        # 1) Instantiate & Train
        model = RK_PINN(input_dim=2, hidden_dim=hd, output_dim=2, num_layers=nl).to(device)
        training_losses, global_errors = train_model_multiple_ics(
            model,
            X_train, Y_train,
            ics_for_eval=ics_eval,
            epochs=epochs,
            lr=learning_rate,
            t0=t0, t_end=t_end, N_eval=N_eval, mu=mu,
            device=device
        )

        # 2) Evaluate final trajectories for each IC & store final-time error
        final_errs_this_config = []
        results = []
        for ic in ics_eval:
            t_rk4, y_rk4 = integrate_rk4(van_der_pol, ic, t0, t_end, N_eval_final, mu=mu, device=device)
            t_nn,  y_nn  = integrate_nn(model, ic, t0, t_end, N_eval_final, device=device)
            pointwise_error = torch.norm(y_rk4 - y_nn, dim=1)

            ft_err = torch.norm(y_rk4[-1] - y_nn[-1]).item()
            final_errs_this_config.append(ft_err)

            results.append((ic, t_rk4, y_rk4, t_nn, y_nn, pointwise_error))

        # Store final-time errors in our dictionary
        final_time_errors[(hd, nl)] = final_errs_this_config

        # We'll just handle 2 ICs in the subplots below
        ic0, t_rk4_0, y_rk4_0, t_nn_0, y_nn_0, err_0 = results[0]
        ic1, t_rk4_1, y_rk4_1, t_nn_1, y_nn_1, err_1 = results[1]

        # Convert to CPU numpy for Plotly
        def to_np(tensor):
            return tensor.detach().cpu().numpy()

        t_rk4_0_np = to_np(t_rk4_0)
        y_rk4_0_np = to_np(y_rk4_0)
        t_nn_0_np  = to_np(t_nn_0)
        y_nn_0_np  = to_np(y_nn_0)
        err_0_np   = to_np(err_0)

        t_rk4_1_np = to_np(t_rk4_1)
        y_rk4_1_np = to_np(y_rk4_1)
        t_nn_1_np  = to_np(t_nn_1)
        y_nn_1_np  = to_np(y_nn_1)
        err_1_np   = to_np(err_1)

        epochs_axis = list(range(epochs))
        glob_err_ic0 = global_errors[0]
        glob_err_ic1 = global_errors[1]

        row_start = 2 * config_idx + 1
        row2 = row_start + 1
        config_label = f"HD={hd},NL={nl}"

        # ---------------------------------------------------
        # a) (row_start, col=1): Training Loss vs. Epoch
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=epochs_axis,
                y=training_losses,
                mode='lines',
                name=f"[{config_label}] TrainLoss"
            ),
            row=row_start, col=1
        )
        fig.update_xaxes(title_text="Epoch",  row=row_start, col=1)
        fig.update_yaxes(
            title_text=f"TrainLoss<br>({config_label})", 
            row=row_start, col=1
        )

        # ---------------------------------------------------
        # b) (row_start, col=2): Global errors vs. epoch, IC0/IC1
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=epochs_axis,
                y=glob_err_ic0,
                mode='lines',
                name=f"[{config_label}] GlobErr(IC0)"
            ),
            row=row_start, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_axis,
                y=glob_err_ic1,
                mode='lines',
                name=f"[{config_label}] GlobErr(IC1)"
            ),
            row=row_start, col=2
        )
        fig.update_xaxes(title_text="Epoch", row=row_start, col=2)
        fig.update_yaxes(
            title_text=f"Global MSE<br>({config_label})", 
            row=row_start, col=2
        )

        # ---------------------------------------------------
        # c) (row_start, col=3): y0(t), IC0 => RK4 vs. NN
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=t_rk4_0_np,
                y=y_rk4_0_np[:, 0],
                mode='lines',
                name=f"[{config_label}] RK4 y0(IC0)"
            ),
            row=row_start, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=t_nn_0_np,
                y=y_nn_0_np[:, 0],
                mode='lines',
                name=f"[{config_label}] NN y0(IC0)"
            ),
            row=row_start, col=3
        )
        fig.update_xaxes(title_text="t", row=row_start, col=3)
        fig.update_yaxes(
            title_text=f"y0(t),IC0<br>({config_label})", 
            row=row_start, col=3
        )

        # ---------------------------------------------------
        # d) (row_start, col=4): y1(t), IC0 => RK4 vs. NN
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=t_rk4_0_np,
                y=y_rk4_0_np[:, 1],
                mode='lines',
                name=f"[{config_label}] RK4 y1(IC0)"
            ),
            row=row_start, col=4
        )
        fig.add_trace(
            go.Scatter(
                x=t_nn_0_np,
                y=y_nn_0_np[:, 1],
                mode='lines',
                name=f"[{config_label}] NN y1(IC0)"
            ),
            row=row_start, col=4
        )
        fig.update_xaxes(title_text="t", row=row_start, col=4)
        fig.update_yaxes(
            title_text=f"y1(t),IC0<br>({config_label})", 
            row=row_start, col=4
        )

        # ---------------------------------------------------
        # e) (row2, col=1): error(t), IC0
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=t_rk4_0_np,
                y=err_0_np,
                mode='lines',
                name=f"[{config_label}] Error(IC0)"
            ),
            row=row2, col=1
        )
        fig.update_xaxes(title_text="t", row=row2, col=1)
        fig.update_yaxes(
            title_text=f"||RK4 - NN||,IC0<br>({config_label})",
            row=row2, col=1
        )

        # ---------------------------------------------------
        # f) (row2, col=2): y0(t), IC1 => RK4 vs. NN
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=t_rk4_1_np,
                y=y_rk4_1_np[:, 0],
                mode='lines',
                name=f"[{config_label}] RK4 y0(IC1)"
            ),
            row=row2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=t_nn_1_np,
                y=y_nn_1_np[:, 0],
                mode='lines',
                name=f"[{config_label}] NN y0(IC1)"
            ),
            row=row2, col=2
        )
        fig.update_xaxes(title_text="t", row=row2, col=2)
        fig.update_yaxes(
            title_text=f"y0(t),IC1<br>({config_label})",
            row=row2, col=2
        )

        # ---------------------------------------------------
        # g) (row2, col=3): y1(t), IC1 => RK4 vs. NN
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=t_rk4_1_np,
                y=y_rk4_1_np[:, 1],
                mode='lines',
                name=f"[{config_label}] RK4 y1(IC1)"
            ),
            row=row2, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=t_nn_1_np,
                y=y_nn_1_np[:, 1],
                mode='lines',
                name=f"[{config_label}] NN y1(IC1)"
            ),
            row=row2, col=3
        )
        fig.update_xaxes(title_text="t", row=row2, col=3)
        fig.update_yaxes(
            title_text=f"y1(t),IC1<br>({config_label})",
            row=row2, col=3
        )

        # ---------------------------------------------------
        # h) (row2, col=4): error(t), IC1
        # ---------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=t_rk4_1_np,
                y=err_1_np,
                mode='lines',
                name=f"[{config_label}] Error(IC1)"
            ),
            row=row2, col=4
        )
        fig.update_xaxes(title_text="t", row=row2, col=4)
        fig.update_yaxes(
            title_text=f"||RK4 - NN||,IC1<br>({config_label})",
            row=row2, col=4
        )

    # -------------------------------------------------------
    # 5.5) Done with big subplots for each config.
    #      Show that figure first.
    # -------------------------------------------------------
    fig.update_layout(
        title=("Van der Pol PINN Experiments (All in One Page)<br>"
               "Varying Hidden Dim & Num Layers, plus Trajectory Plots"),
        height=3000 + 200 * num_configs,  # enlarge if many configs
        width=1800,
        showlegend=True
    )
    fig.show()

    # ----------------------------------------------------------------
    # 5.6) New figure: LINE PLOTS of final-time error vs. network size
    #
    # We'll do a 1-row, 2-column layout:
    #  - Left: final-time error vs. hidden_dim (lines for each num_layers)
    #  - Right: final-time error vs. num_layers (lines for each hidden_dim)
    #
    # Each line is also distinguished by which IC it's for (IC0, IC1, etc.).
    # ----------------------------------------------------------------
    unique_hds = sorted({hd for (hd, nl) in configs})
    unique_nls = sorted({nl for (hd, nl) in configs})
    num_ics = len(ics_eval)

    fig_line = make_subplots(rows=1, cols=2,
                             subplot_titles=("Final-Time Error vs. Hidden Dim",
                                             "Final-Time Error vs. Num Layers"),
                             horizontal_spacing=0.1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # (1) Final-time error vs. hidden_dim
    # For each num_layers, and each IC, we plot a line over hidden_dims.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    for nl in unique_nls:
        # We'll gather (x=hd, y= final_time_errors[(hd,nl)][ic]) for each hd
        # We want multiple lines: one line per (nl, ic).
        # Let's do one line per ic. So the name is e.g. "NL=2, IC0"
        for ic_idx in range(num_ics):
            x_vals = []
            y_vals = []
            for hd in unique_hds:
                if (hd, nl) in final_time_errors:
                    ft_errs = final_time_errors[(hd, nl)]
                    # ft_errs is a list [errIC0, errIC1, ...]
                    if ic_idx < len(ft_errs):
                        x_vals.append(hd)
                        y_vals.append(ft_errs[ic_idx])

            fig_line.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=f"NL={nl}, IC={ic_idx}",
                ),
                row=1, col=1
            )

    fig_line.update_xaxes(title_text="Hidden Dimension", row=1, col=1)
    fig_line.update_yaxes(title_text="Final-Time Error", row=1, col=1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # (2) Final-time error vs. num_layers
    # For each hidden_dim, and each IC, we plot a line over layer_counts.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    for hd in unique_hds:
        for ic_idx in range(num_ics):
            x_vals = []
            y_vals = []
            for nl in unique_nls:
                if (hd, nl) in final_time_errors:
                    ft_errs = final_time_errors[(hd, nl)]
                    if ic_idx < len(ft_errs):
                        x_vals.append(nl)
                        y_vals.append(ft_errs[ic_idx])

            fig_line.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines+markers',
                    name=f"HD={hd}, IC={ic_idx}",
                ),
                row=1, col=2
            )

    fig_line.update_xaxes(title_text="Number of Layers", row=1, col=2)
    fig_line.update_yaxes(title_text="Final-Time Error", row=1, col=2)

    fig_line.update_layout(
        title="Line Plots of Final-Time Error vs. Network Parameters",
        width=1200,
        height=600,
        showlegend=True
    )

    fig_line.show()
