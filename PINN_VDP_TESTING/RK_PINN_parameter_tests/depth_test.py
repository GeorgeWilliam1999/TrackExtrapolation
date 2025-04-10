import torch
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np

##############################################################################
# 1) Van der Pol ODE and RK4 Solver
##############################################################################

def van_der_pol(t, y, mu=1.0):
    """
    Van der Pol system:
      dy1/dt = y2
      dy2/dt = mu * (1 - y1^2)*y2 - y1
    """
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
    return dydt

def rk4_step(func, t, y, dt, mu=1.0):
    """
    One step of 4th-order Runge-Kutta for dy/dt = func(t,y,mu).
    """
    k1 = func(t, y, mu)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1, mu)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2, mu)
    k4 = func(t + dt,     y + dt*k3,    mu)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_rk4(func, y0, t0, t_end, N, mu=1.0, device="cpu"):
    """
    Integrate from t0 to t_end in N steps with RK4. 
    Returns (times, states).
    times: shape (N+1,)
    states: shape (N+1, 2)
    """
    dt = (t_end - t0) / N
    t = t0
    y = y0.clone().to(device)
    times = []
    ys = []
    for _ in range(N+1):
        times.append(t)
        ys.append(y.unsqueeze(0))
        y = rk4_step(func, t, y, dt, mu)
        t += dt
    return torch.tensor(times, device=device), torch.cat(ys, dim=0)

##############################################################################
# 2) PINN Model: 2 Hidden Layers (num_layers=2) - We'll vary hidden_dim
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
# 3) Build Training Pairs (y_n, y_{n+1})
##############################################################################

def generate_training_data(func, y0, t0, t_end, N, mu=1.0, device="cpu"):
    """
    Integrate with RK4 and form pairs (y_n, y_{n+1}) for n=0..N-1.
    Returns (X, Y) each shape (N, 2).
    """
    _, all_ys = integrate_rk4(func, y0, t0, t_end, N, mu=mu, device=device)
    X = all_ys[:-1]
    Y = all_ys[1:]
    return X, Y

##############################################################################
# 4) Evaluate Final-Time Error
##############################################################################

def compute_final_time_error(model, y0, t0, t_end, N_eval, mu=1.0, device="cpu"):
    """
    Integrate RK4 and NN from t0 to t_end in N_eval steps, 
    then return || y_RK4(t_end) - y_NN(t_end) || (Euclidean).
    """
    _, y_rk4 = integrate_rk4(van_der_pol, y0, t0, t_end, N_eval, mu=mu, device=device)
    _, y_nn = integrate_nn(model, y0, t0, t_end, N_eval, device=device)
    return torch.norm(y_rk4[-1] - y_nn[-1]).item()

def integrate_nn(model, y0, t0, t_end, N_eval, device="cpu"):
    """
    Repeatedly apply model to get y_{n+1} from y_n, 
    for n=0..N_eval, from t0..t_end
    """
    dt = (t_end - t0) / N_eval
    t = t0
    y = y0.clone().to(device).unsqueeze(0)
    times = []
    ys = []
    for _ in range(N_eval+1):
        times.append(t)
        ys.append(y)
        y = model(y)
        t += dt
    return torch.tensor(times, device=device), torch.cat(ys, dim=0)

##############################################################################
# 5) Training Until Convergence (Early Stopping)
##############################################################################

def train_model_until_convergence(
    model,
    X_train, Y_train,
    optimizer_fn=torch.optim.Adam,
    lr=1e-3,
    min_epochs=100,
    max_epochs=5000,
    loss_tol=1e-6,
    patience=10,
    device="cpu"
):
    """
    Simple 'early stopping':
      - Must train at least `min_epochs`.
      - If improvement in training loss < loss_tol for `patience` consecutive
        epochs (after min_epochs), we stop early.
    Returns: (training_losses, final_epoch).
    """
    model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    optimizer = optimizer_fn(model.parameters(), lr=lr)
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

        # Print occasionally
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss={curr_loss:.6e}")

        # improvement
        if prev_loss is not None:
            improvement = prev_loss - curr_loss
        else:
            improvement = float('inf')

        if improvement < loss_tol:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if (epoch >= min_epochs) and (no_improve_count >= patience):
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            return training_losses, epoch

        prev_loss = curr_loss

    print(f"Reached max_epochs={max_epochs} without early stopping.")
    return training_losses, max_epochs

##############################################################################
# 6) Main Experiment: 
#    - Use exactly 2 hidden layers
#    - Vary hidden_dim
#    - Train with early stopping
#    - Compute final-time error for multiple ICs
#    - Plot final-time error vs hidden_dim
##############################################################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Experiment settings
    t0, t_end = 0.0, 5.0
    mu = 1.0
    N_data = 100          # steps for training data
    N_eval = 200          # steps for evaluating final-time error
    min_epochs = 200      # train at least 200 epochs
    max_epochs = 5000
    loss_tol = 1e-7
    patience = 20
    lr = 1e-3

    # We'll fix the number of hidden layers = 2
    # Vary hidden_dim across some set
    hidden_dims = [i for i in range(16, 65)]

    # Multiple initial conditions
    ics = [
        torch.tensor([2.0, 0.0]),
        torch.tensor([1.0, -1.0])
    ]

    # Build training data from all ICs
    X_train_list = []
    Y_train_list = []
    for ic in ics:
        X_ic, Y_ic = generate_training_data(van_der_pol, ic, t0, t_end, N_data, mu=mu, device=device)
        X_train_list.append(X_ic)
        Y_train_list.append(Y_ic)
    X_train = torch.cat(X_train_list, dim=0)
    Y_train = torch.cat(Y_train_list, dim=0)

    # We'll measure final-time error for each IC separately
    # final_time_errors[hd] = [errIC0, errIC1, ...]
    final_time_errors = {}

    # 2) Loop over hidden_dims
    for hd in hidden_dims:
        print(f"\n===== Training with hidden_dim={hd}, 2 hidden layers =====")
        model = RK_PINN(input_dim=2, hidden_dim=hd, output_dim=2, num_layers=2)

        # Train until convergence
        train_losses, final_epoch = train_model_until_convergence(
            model,
            X_train, Y_train,
            optimizer_fn=torch.optim.Adam,
            lr=lr,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            loss_tol=loss_tol,
            patience=patience,
            device=device
        )
        print(f"Finished at epoch {final_epoch} with final training loss={train_losses[-1]:.6e}")

        # Evaluate final-time error for each IC
        errs = []
        for ic_idx, ic in enumerate(ics):
            err_ft = compute_final_time_error(model, ic, t0, t_end, N_eval, mu=mu, device=device)
            errs.append(err_ft)
            print(f"  IC={ic.tolist()} => final-time error={err_ft:.6e}")
        final_time_errors[hd] = errs

    # 3) Plot final-time error vs hidden_dim (lines for each IC)
    # Using Plotly
    fig = go.Figure()
    num_ics = len(ics)
    for ic_idx in range(num_ics):
        x_vals = []
        y_vals = []
        for hd in hidden_dims:
            x_vals.append(hd)
            y_vals.append(final_time_errors[hd][ic_idx])
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=f"IC{ic_idx}"
            )
        )

    fig.update_layout(
        title="Final-Time Error vs. Hidden Dimension (2 Layers, Early Stopping)",
        xaxis_title="Hidden Dimension",
        yaxis_title="|| y_RK4(t_end) - y_NN(t_end) ||",
        width=800,
        height=500,
        showlegend=True
    )
    fig.write_html('PINN_VDP_TESTING/RK_PINN_parameter_tests/plots/depth_test.html')
    fig.show()
