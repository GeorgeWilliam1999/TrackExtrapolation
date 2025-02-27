import torch
import numpy as np
import torch.nn as nn
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import plotly.io as pio

class RK_PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(RK_PINN, self).__init__()
        
        # Create ModuleList to hold hidden layers
        self.hidden_layers = nn.ModuleList()
        
        # First hidden layer (from input_dim to hidden_dim)
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Additional hidden layers (from hidden_dim to hidden_dim)
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Output layer (no activation)
        x = self.output_layer(x)
        return x

def van_der_pol(t, y, mu=1.0):
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
    return dydt

def implicit_midpoint_step(model, y0, t0, dt):
    y_mid = y0 + 0.5 * dt * van_der_pol(t0, y0)
    
    def func(y_mid):
        return y_mid - y0 - 0.5 * dt * (van_der_pol(t0, y0) + van_der_pol(t0 + 0.5 * dt, y_mid))
    
    for _ in range(10):
        y_mid = y0 + 0.5 * dt * (van_der_pol(t0, y0) + van_der_pol(t0 + 0.5 * dt, y_mid))
    
    y_next = y0 + dt * van_der_pol(t0 + 0.5 * dt, y_mid)
    return y_next

def train(model, optimizer, criterion, y0, t0, dt, epochs):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(y0)
        y_true = implicit_midpoint_step(model, y0, t0, dt)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return losses

if __name__ == "__main__":
    # Define parameter ranges
    parameter_ranges = {
        'input_dim': [2],
        'hidden_dim': [i for i in range(1, 101, 10)],
        'output_dim': [2],
        'num_layers': [i for i in range(1, 11)],
        'learning_rate': [i * 1e-3 for i in range(1, 11)],
        'epochs': [1000],
        'dt': [i * 1e-2 for i in range(1, 11)]
    }

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        parameter_ranges['input_dim'],
        parameter_ranges['hidden_dim'],
        parameter_ranges['output_dim'],
        parameter_ranges['num_layers'],
        parameter_ranges['learning_rate'],
        parameter_ranges['epochs'],
        parameter_ranges['dt']
    ))

    # Convert to list of dictionaries
    all_params = []
    for combo in param_combinations:
        param_dict = {
            'input_dim': combo[0],
            'hidden_dim': combo[1],
            'output_dim': combo[2],
            'num_layers': combo[3],
            'learning_rate': combo[4],
            'epochs': combo[5],
            'dt': combo[6]
        }
        all_params.append(param_dict)

    print(f"Generated {len(all_params)} parameter combinations")

    results = {}

    for params in all_params:
        results[str(params)] = []

        input_dim, hidden_dim, output_dim, num_layers, learning_rate, epochs, dt = params.values()
    
        model = RK_PINN(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float32)
        t0 = 0.0

        save_path = 'Van_Der_Pol_plots'
        os.makedirs(save_path, exist_ok=True)

        losses = train(model, optimizer, criterion, y0, t0, dt, epochs)
        results[str(params)].append(losses)

        initial_conditions = [
            torch.tensor([2.0, 0.0], dtype=torch.float32),
            # torch.tensor([1.0, 0.0], dtype=torch.float32),
            # torch.tensor([0.5, 0.0], dtype=torch.float32)
        ]
    # After all parameter combinations are evaluated, create a dashboard for visualization
    print("Creating hyperparameter dashboard...")

    # Create dashboard to visualize hyperparameter effects
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Hidden Dimension vs Loss", "Number of Layers vs Loss", 
                       "Learning Rate vs Loss", "Time Step (dt) vs Loss")
    )

    # Prepare data for each plot
    hidden_dim_data = {}
    num_layers_data = {}  
    learning_rate_data = {}
    dt_data = {}

    for param_str, loss_lists in results.items():
        if loss_lists:  # Check if there are any results for this parameter set
            final_loss = loss_lists[0][-1]  # Get the last loss from the first training run
            
            # Extract parameters from the string representation
            param_dict = eval(param_str)  # Safe here as we control the string format
            
            # Group by hidden_dim
            hd = param_dict['hidden_dim']
            if hd not in hidden_dim_data:
                hidden_dim_data[hd] = []
            hidden_dim_data[hd].append(final_loss)
            
            # Group by num_layers
            nl = param_dict['num_layers']
            if nl not in num_layers_data:
                num_layers_data[nl] = []
            num_layers_data[nl].append(final_loss)
            
            # Group by learning_rate
            lr = param_dict['learning_rate']
            if lr not in learning_rate_data:
                learning_rate_data[lr] = []
            learning_rate_data[lr].append(final_loss)
            
            # Group by dt
            dt = param_dict['dt']
            if dt not in dt_data:
                dt_data[dt] = []
            dt_data[dt].append(final_loss)

    # Calculate average loss for each parameter value
    hidden_dim_avg = {k: sum(v)/len(v) for k, v in hidden_dim_data.items()}
    num_layers_avg = {k: sum(v)/len(v) for k, v in num_layers_data.items()}
    learning_rate_avg = {k: sum(v)/len(v) for k, v in learning_rate_data.items()}
    dt_avg = {k: sum(v)/len(v) for k, v in dt_data.items()}

    # Sort the dictionaries by keys for better visualization
    hidden_dim_avg = {k: hidden_dim_avg[k] for k in sorted(hidden_dim_avg.keys())}
    num_layers_avg = {k: num_layers_avg[k] for k in sorted(num_layers_avg.keys())}
    learning_rate_avg = {k: learning_rate_avg[k] for k in sorted(learning_rate_avg.keys())}
    dt_avg = {k: dt_avg[k] for k in sorted(dt_avg.keys())}

    # Add traces for each plot
    fig.add_trace(
        go.Scatter(x=list(hidden_dim_avg.keys()), y=list(hidden_dim_avg.values()),
                   mode='lines+markers', name='Hidden Dim'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=list(num_layers_avg.keys()), y=list(num_layers_avg.values()),
                   mode='lines+markers', name='Num Layers'),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=list(learning_rate_avg.keys()), y=list(learning_rate_avg.values()),
                   mode='lines+markers', name='Learning Rate'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=list(dt_avg.keys()), y=list(dt_avg.values()),
                   mode='lines+markers', name='dt'),
        row=2, col=2
    )

    # Update axis titles
    fig.update_xaxes(title_text="Hidden Dimension", row=1, col=1)
    fig.update_xaxes(title_text="Number of Layers", row=1, col=2)
    fig.update_xaxes(title_text="Learning Rate", row=2, col=1)
    fig.update_xaxes(title_text="Time Step (dt)", row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text="Loss", row=i, col=j)

    # Update layout
    fig.update_layout(
        title="Hyperparameter Effects on Model Loss",
        height=800,
        width=1000,
    )

    # Save the dashboard
    dashboard_path = os.path.join(save_path, 'hyperparameter_dashboard.html')
    pio.write_html(fig, dashboard_path)
    print(f"Dashboard saved to {dashboard_path}")

    # Show the dashboard
    fig.show()


