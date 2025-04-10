import torch
import torch.nn as nn
import random
import plotly.graph_objects as go
import numpy as np

##############################################################################
# 1) Single RK4 Step
##############################################################################

def single_step_rk4(func, t, y, dt, mu=1.0):
    """
    Perform exactly ONE RK4 step from time t to t+dt.
    Returns y(t+dt).
    func: your ODE function, e.g. van_der_pol(t, y, mu)
    t:    current time (float)
    y:    current state (Tensor)
    dt:   step size
    mu:   ODE parameter
    """
    k1 = func(t, y, mu)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1, mu)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2, mu)
    k4 = func(t + dt,     y + dt*k3,    mu)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

##############################################################################
# 2) Partial Integration from t0 to t_rand in N uniform steps
##############################################################################

def partial_rk4(func, y0, t0, t_rand, N=100, mu=1.0, device="cpu"):
    """
    Integrate from t0 to t_rand in N uniform steps using single_step_rk4.
    If t_rand == t0 or N <= 0, we just return y0.
    
    dt = (t_rand - t0) / N

    Returns y(t_rand).
    """
    if N <= 0:
        return y0
    dt = (t_rand - t0) / N
    if dt <= 0:
        return y0  # means t_rand <= t0
    y = y0.clone().to(device)
    t = t0
    for _ in range(N):
        y = single_step_rk4(func, t, y, dt, mu=mu)
        t += dt
    return y

##############################################################################
# 3) Data Generation: Random One-Step from a Random t_rand
##############################################################################

def generate_training_data_random_one_step(
    func, y0, t0, t_end, dt_single,
    sample_size,
    N=100,    # number of sub-steps for partial_rk4
    mu=1.0, device="cpu"
):
    """
    For each sample in [sample_size]:
      1) Pick random t_rand in [t0, t_end - dt_single]
      2) partial_rk4 => y(t_rand) using N sub-steps, each step = (t_rand - t0)/N
      3) single_step_rk4 => y(t_rand + dt_single)
      4) store (y_rand, y_next)

    Returns X, Y each shape (sample_size, 2).

    dt_single: the single step to integrate from t_rand to t_rand+dt_single (the next-step).
    N: number of partial sub-steps from t0 to t_rand.
    """
    X_list = []
    Y_list = []
    for _ in range(sample_size):
        t_rand = random.uniform(t0, t_end - dt_single)
        y_rand = partial_rk4(func, y0, t0, t_rand, N=N, mu=mu, device=device)
        y_next = single_step_rk4(func, t_rand, y_rand, dt_single, mu=mu)
        X_list.append(y_rand)
        Y_list.append(y_next)

    X = torch.stack(X_list).to(device)
    Y = torch.stack(Y_list).to(device)
    return X, Y

##############################################################################
# 4) Van der Pol and other helpers
##############################################################################

def van_der_pol(t, y, mu=1.0):
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu*(1 - y[0]**2)*y[1] - y[0]
    return dydt

def integrate_nn(model, y0, t0, t_end, N_eval, dt_single, device="cpu"):
    """
    Repeatedly apply model (like an Euler approach, but here you do 1 NN step)
    consistent with how data was generated: a single step of size dt_single.
    """
    times = []
    ys = []
    t = t0
    y = y0.clone().unsqueeze(0).to(device)
    while t <= t_end + 1e-12:
        times.append(t)
        ys.append(y)
        y = model(y)  # next-step from y(t)
        y = y  # shape (1,2)
        t += dt_single

        if t > t_end and t - t_end < 0.5*dt_single:
            break

    return (torch.tensor(times, device=device),
            torch.cat(ys, dim=0))

def compute_final_time_error(model, y0, t0, t_end, N_eval, dt_single, mu=1.0, device="cpu"):
    """
    We define a discrete approach to measure final-time error:
    - "True" solution y(t_end) ~ partial_rk4 with M=N_eval sub-steps
    - NN-based solution: repeated calls with step dt_single
    - Compare final states (Euclidean norm).
    """
    # 1) "True" solution at t_end
    y_true = partial_rk4(van_der_pol, y0, t0, t_end, N=N_eval, mu=mu, device=device)

    # 2) NN-based solution with repeated dt_single steps
    times_nn, ys_nn = integrate_nn(model, y0, t0, t_end, N_eval, dt_single, device=device)
    y_nn_final = ys_nn[-1]  # last state

    return torch.norm(y_true - y_nn_final).item()

##############################################################################
# 5) Simple PINN, plus an "until convergence" training
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

def train_model_until_convergence(
    model,
    X_train, Y_train,
    lr=1e-3,
    min_epochs=100,
    max_epochs=2000,
    loss_tol=1e-7,
    patience=20,
    device="cpu"
):
    model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    training_losses = []
    no_improve_count = 0
    prev_loss = None

    for epoch in range(1, max_epochs+1):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        training_losses.append(curr_loss)

        if prev_loss is not None:
            improvement = prev_loss - curr_loss
        else:
            improvement = float("inf")

        if improvement < loss_tol:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if (epoch >= min_epochs) and (no_improve_count >= patience):
            print(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
            return training_losses, epoch

        prev_loss = curr_loss

    print(f"Reached max_epochs={max_epochs} without early stopping.")
    return training_losses, max_epochs

##############################################################################
# 6) Main Experiment: Vary the Number of partial sub-steps "N"
##############################################################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters / Settings
    t0, t_end = 0.0, 5.0
    dt_single = 0.05    # single-step size from t_rand -> t_rand+dt_single
    sample_size = 100   # how many random pairs we generate
    hidden_dim = 32
    num_layers = 2
    min_epochs = 100
    max_epochs = 2000
    loss_tol = 1e-7
    patience = 20
    lr = 1e-3

    # We'll define multiple N values to test, i.e. the # sub-steps
    # for partial_rk4 from t0..t_rand
    N_list = [10, 50, 100, 200]

    # We can define multiple initial conditions if desired
    ics = [
        torch.tensor([2.0, 0.0]),
        torch.tensor([1.0, -1.0])
    ]

    # We'll measure final-time error for each N, averaged over ICs
    N_errors = []

    for N_val in N_list:
        print(f"\n===== Training with N={N_val} sub-steps for partial_rk4. =====")

        # Build training data from random approach
        # For multiple ICs, we'll just concatenate
        X_all = []
        Y_all = []
        for ic in ics:
            X_ic, Y_ic = generate_training_data_random_one_step(
                func=van_der_pol, y0=ic,
                t0=t0, t_end=t_end,
                dt_single=dt_single,
                sample_size=sample_size,
                N=N_val,          # <--- here is the key difference
                mu=1.0,
                device=device
            )
            X_all.append(X_ic)
            Y_all.append(Y_ic)
        X_train = torch.cat(X_all, dim=0)
        Y_train = torch.cat(Y_all, dim=0)

        # Instantiate and train
        model = RK_PINN(input_dim=2, hidden_dim=hidden_dim, output_dim=2, num_layers=num_layers)
        train_losses, final_epoch = train_model_until_convergence(
            model,
            X_train, Y_train,
            lr=lr,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            loss_tol=loss_tol,
            patience=patience,
            device=device
        )
        print(f"Training done at epoch={final_epoch}, final loss={train_losses[-1]:.6e}")

        # Evaluate final-time error for each IC, then average
        ic_errs = []
        for ic in ics:
            err_ic = compute_final_time_error(
                model, ic, t0, t_end, N_eval=200, dt_single=dt_single, mu=1.0, device=device
            )
            ic_errs.append(err_ic)
            print(f"    IC={ic.tolist()} => final-time error={err_ic:.6e}")
        avg_err = sum(ic_errs)/len(ic_errs)
        N_errors.append(avg_err)
        print(f"   => Average final-time error (over {len(ics)} ICs) = {avg_err:.6e}")

    # Now we have a list of errors N_errors corresponding to N_list
    # We'll do a line plot in Plotly
    fig = go.Figure(
        data=[
            go.Scatter(
                x=N_list,
                y=N_errors,
                mode='lines+markers',
                name="Final-Time Error"
            )
        ]
    )
    fig.update_layout(
        title="Final-Time Error vs. N (sub-steps for partial_rk4)",
        xaxis_title="N (# sub-steps from t0 to t_rand)",
        yaxis_title="Average Final-Time Error",
        width=800, height=500
    )
    fig.write_html('PINN_VDP_TESTING/RK_PINN_parameter_tests/plots/time_step_test.html')
    fig.show()
