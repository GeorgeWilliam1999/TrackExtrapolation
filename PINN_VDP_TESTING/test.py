import torch
import numpy as np
import torch.nn as nn
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import webbrowser
import plotly.io as pio

class RK_PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving differential equations.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output features
        num_layers (int): Number of hidden layers
    """
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
        """Forward pass through the network"""
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

def van_der_pol(t, y, mu=1.0):
    """
    Van der Pol oscillator differential equation.
    
    Args:
        t (float): Time
        y (tensor): State vector [position, velocity]
        mu (float): Oscillator parameter
        
    Returns:
        tensor: State derivatives [dy/dt]
    """
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
    return dydt

def implicit_midpoint_step(model, y0, t0, dt):
    """
    Implicit midpoint method for numerical integration.
    
    Args:
        model: The neural network model
        y0 (tensor): Initial state
        t0 (float): Initial time
        dt (float): Time step
        
    Returns:
        tensor: Next state after time step
    """
    y_mid = y0 + 0.5 * dt * van_der_pol(t0, y0)
    
    # Iterative refinement of midpoint estimate
    for _ in range(10):
        y_mid = y0 + 0.5 * dt * (van_der_pol(t0, y0) + van_der_pol(t0 + 0.5 * dt, y_mid))
    
    y_next = y0 + dt * van_der_pol(t0 + 0.5 * dt, y_mid)
    return y_next

def train(model, optimizer, criterion, y0, t0, dt, epochs):
    """
    Train the PINN model.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        criterion: Loss function
        y0 (tensor): Initial state
        t0 (float): Initial time
        dt (float): Time step
        epochs (int): Number of training epochs
        
    Returns:
        list: Training losses
    """
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(y0)
        y_true = implicit_midpoint_step(model, y0, t0, dt)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return losses

def generate_parameter_combinations(parameter_ranges):
    """
    Generate all combinations of hyperparameters for grid search.
    
    Args:
        parameter_ranges (dict): Ranges for each hyperparameter
        
    Returns:
        list: Dictionaries of parameter combinations
    """
    param_combinations = list(itertools.product(
        parameter_ranges['input_dim'],
        parameter_ranges['hidden_dim'],
        parameter_ranges['output_dim'],
        parameter_ranges['num_layers'],
        parameter_ranges['learning_rate'],
        parameter_ranges['epochs'],
        parameter_ranges['dt']
    ))

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
        
    return all_params

def create_interactive_dashboard(results, save_path):
    """
    Create an interactive visualization dashboard with separate tabs for each hyperparameter.
    For each measured parameter, all the *other* parameters appear as dropdown controls.
    The plot is made larger for better visibility.
    
    Args:
        results (dict): Training results for each parameter combination
        save_path (str): Directory to save visualization files
    """
    # We treat param_keys as the stringified dicts, param_values as the actual dicts.
    param_keys = list(results.keys())
    param_values = [eval(k) for k in param_keys]

    # Gather all distinct hyperparameter names from any param_dict
    # (We expect input_dim, hidden_dim, output_dim, num_layers, learning_rate, epochs, dt, etc.)
    all_hparam_names = set()
    for pdict in param_values:
        all_hparam_names.update(pdict.keys())
    all_hparam_names = sorted(all_hparam_names)

    # Build a dict of possible values for each parameter
    possible_values = {}
    for name in all_hparam_names:
        # Gather all distinct values for this name
        vals = set()
        for pdict in param_values:
            if name in pdict:
                vals.add(pdict[name])
        possible_values[name] = sorted(vals)

    # We'll define a function to create the figure for a measured key.
    def create_fig(measured_key):
        """
        Creates a figure measuring the effect of 'measured_key'.
        All other keys become updatemenus (controls) in the top bar.
        """
        # We'll produce one trace per param_dict that has this measured_key
        # and store them in the figure.
        traces = []
        for pk in param_keys:
            pdict = eval(pk)
            # Make sure this param set includes the measured_key
            # If so, add a new trace of its loss.
            if measured_key in pdict:
                # The associated loss history
                loss_history = results[pk][0]
                epochs = list(range(len(loss_history)))
                # We'll label the trace according to the measured_key's value.
                traces.append(
                    go.Scatter(
                        x=epochs,
                        y=loss_history,
                        mode='lines',
                        name=f"{measured_key}={pdict[measured_key]}",
                        visible=True
                    )
                )

        fig = go.Figure(traces)
        fig.update_layout(
            title=f"Loss Convergence vs. {measured_key}",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            legend_title=measured_key,
            width=1200,   # Make the plot larger
            height=800
        )

        # Now we add updatemenus for all the other parameters.
        # Each parameter is a separate dropdown.
        # We'll place them horizontally across the top.

        # We want all other keys except the measured_key.
        other_keys = [k for k in all_hparam_names if k != measured_key]

        # We'll build one updatemenu entry per other_key.
        updatemenus = []
        # We'll space them out in x. We'll place them in a single row.
        # e.g. if we have N = len(other_keys), we can do something like
        # x positions from 0.1, 0.3, 0.5, 0.7, etc.
        # We'll do a small function that yields positions.
        def positions_for_n(n):
            # We'll produce positions from 0.1, 0.25, 0.4, 0.55, etc.
            # or something that fits.
            # We'll just linearly space them from 0.1 to 0.9.
            if n == 1:
                return [0.15]
            step = (0.9 - 0.1) / (n - 1)
            return [0.1 + i*step for i in range(n)]

        x_positions = positions_for_n(len(other_keys))

        for i, key in enumerate(other_keys):
            # Build a set of possible values for this key
            key_values = possible_values[key]

            buttons = []
            for val in key_values:
                # For each possible value, we set 'visible' to True for those traces that match.
                # We have the same number of traces as param_keys, so we'll do a list comprehension
                # that checks if param_dict's key is this val.
                visible_list = []
                # We'll iterate again over param_keys in the same order we appended traces.
                for pk in param_keys:
                    pdict = eval(pk)
                    # A trace is visible if:
                    # - measured_key in pdict (since we only appended traces if measured_key in pdict),
                    # - and pdict[key] == val.
                    # Otherwise, hide.
                    if measured_key in pdict:
                        if pdict.get(key, None) == val:
                            visible_list.append(True)
                        else:
                            visible_list.append(False)
                    else:
                        visible_list.append(False)

                buttons.append(
                    {
                        "label": f"{key}={val}",
                        "method": "update",
                        "args": [
                            {"visible": visible_list},
                            {"title": f"Loss for {measured_key} with {key}={val}"}
                        ]
                    }
                )

            updatemenus.append(
                {
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": x_positions[i],
                    "y": 1.15,
                    "xanchor": "center"
                }
            )

        fig.update_layout(updatemenus=updatemenus)
        return fig

    # We'll create one figure/tab for each hyperparameter in all_hparam_names.
    # If you only want certain keys, you can filter them.
    # E.g., if you want to exclude input_dim or epochs, you can skip them.

    # Let's assume you want all of them. We'll just create them all.
    # But if you want just certain ones, you can do e.g.:
    # measured_keys = ["hidden_dim", "num_layers", "learning_rate", "dt"]

    measured_keys = ["input_dim", "hidden_dim", "output_dim", "num_layers", "learning_rate", "epochs", "dt"]
    # Filter out anything that isn't in all_hparam_names, if needed.
    measured_keys = [k for k in measured_keys if k in all_hparam_names]

    figs = {}
    for mk in measured_keys:
        # We'll gather all the distinct values for mk in param_values
        # But we actually don't strictly need that as an argument to create_fig now,
        # since we do an inline check. We'll keep a small set:
        mk_vals = possible_values.get(mk, [])
        figs[mk] = create_fig(mk)

    # Build the tabs HTML.
    tabs_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .tab { overflow: hidden; }
            .tab button { float: left; padding: 10px; cursor: pointer; }
            .tabcontent { display: none; padding: 10px; border-top: none; }
            .tab button.active { background-color: #ccc; }
        </style>
    </head>
    <body>
        <div class="tab">
    """

    div_contents = ""
    first_tab = True
    for name, fig in figs.items():
        tab_id = name.replace(" ", "_")
        tabs_html += f'<button class="tablinks" onclick="openTab(event, \'{tab_id}\')">{name}</button>'
        div_contents += f'<div id="{tab_id}" class="tabcontent">'
        div_contents += pio.to_html(fig, include_plotlyjs=False, full_html=False)
        div_contents += '</div>'
        if first_tab:
            first_tab = False

    tabs_html += f"""
        </div>
    {div_contents}
        <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        document.getElementsByClassName("tablinks")[0].click();
        </script>
    </body>
    </html>
    """

    interactive_dashboard_path = os.path.join(save_path, 'interactive_dashboard.html')
    with open(interactive_dashboard_path, 'w') as f:
        f.write(tabs_html)

    print(f"Interactive dashboard saved to {save_path}")
    webbrowser.open('file://' + os.path.abspath(interactive_dashboard_path))



def main():
    """Main function to run hyperparameter search and visualization"""
    # Define parameter ranges for grid search
    parameter_ranges = {
        'input_dim': [2],
        'hidden_dim': [i for i in range(1, 3)],
        'output_dim': [2],
        'num_layers': [i for i in range(1, 3)],
        'learning_rate': [i * 1e-3 for i in range(1, 3)],
        'epochs': [100],
        'dt': [i * 1e-2 for i in range(1, 3)]
    }
    
    # Generate parameter combinations
    all_params = generate_parameter_combinations(parameter_ranges)
    print(f"Generated {len(all_params)} parameter combinations")
    
    # Setup output directory
    save_path = 'Van_Der_Pol_plots'
    os.makedirs(save_path, exist_ok=True)
    
    # Train models with different parameters
    results = {}
    for params in all_params:
        print(f'Training parameters: {params}')
        results[str(params)] = []

        model = RK_PINN(**{k: params[k] for k in 
                          ['input_dim', 'hidden_dim', 'output_dim', 'num_layers']})
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float32)
        t0 = 0.0

        losses = train(model, optimizer, criterion, y0, t0, params['dt'], params['epochs'])
        results[str(params)].append(losses)
    
    # Create and save interactive visualization dashboard
    create_interactive_dashboard(results, save_path)

if __name__ == "__main__":
    main()
