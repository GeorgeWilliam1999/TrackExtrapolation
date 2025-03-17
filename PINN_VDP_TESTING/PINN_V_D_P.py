import torch
import numpy as np
import torch.nn as nn
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import webbrowser
import plotly.io as p
from math import ceil
from collections import defaultdict

class RK_PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(RK_PINN, self).__init__()
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

def van_der_pol(t, y, mu=1.0):
    dydt = torch.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[0]**2) * y[1] - y[0]
    return dydt

def implicit_midpoint_step(model, y0, t0, dt):
    y_mid = y0 + 0.5 * dt * van_der_pol(t0, y0)
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
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return losses

def generate_parameter_combinations(parameter_ranges):
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

def create_dashboard(results, save_path):
    fig_params = make_subplots(rows=2, cols=2, subplot_titles=("HD vs Final Loss", "Layers vs Final Loss",
                                                               "LR vs Final Loss", "dt vs Final Loss"))
    hidden_dim_data = defaultdict(list)
    num_layers_data = defaultdict(list)
    learning_rate_data = defaultdict(list)
    dt_data = defaultdict(list)

    for param_str, loss_lists in results.items():
        if not loss_lists:
            continue
        loss_history = loss_lists[0]
        final_loss = loss_history[-1]
        param_dict = eval(param_str)

        hd = param_dict['hidden_dim']
        hidden_dim_data[hd].append(final_loss)

        nl = param_dict['num_layers']
        num_layers_data[nl].append(final_loss)

        lr = param_dict['learning_rate']
        learning_rate_data[lr].append(final_loss)

        d_t = param_dict['dt']
        dt_data[d_t].append(final_loss)

    def average_dict(data_dict):
        return {k: sum(v)/len(v) for k, v in data_dict.items()}

    hd_avg = average_dict(hidden_dim_data)
    nl_avg = average_dict(num_layers_data)
    lr_avg = average_dict(learning_rate_data)
    dt_avg = average_dict(dt_data)

    fig_params.add_trace(go.Scatter(x=list(hd_avg.keys()), y=list(hd_avg.values()),
                                    mode='lines+markers', name='HD'),
                         row=1, col=1)
    fig_params.add_trace(go.Scatter(x=list(nl_avg.keys()), y=list(nl_avg.values()),
                                    mode='lines+markers', name='Layers'),
                         row=1, col=2)
    fig_params.add_trace(go.Scatter(x=list(lr_avg.keys()), y=list(lr_avg.values()),
                                    mode='lines+markers', name='LR'),
                         row=2, col=1)
    fig_params.add_trace(go.Scatter(x=list(dt_avg.keys()), y=list(dt_avg.values()),
                                    mode='lines+markers', name='dt'),
                         row=2, col=2)

    labels = ["HD","Layers","LR","dt"]
    idx = 0
    for i in range(1, 3):
        for j in range(1, 3):
            fig_params.update_xaxes(title_text=labels[idx], row=i, col=j)
            fig_params.update_yaxes(title_text="Final Loss", row=i, col=j)
            idx += 1
    fig_params.update_layout(title="Hyperparameter Effects on Final Loss")

    fig_convergence = make_subplots(rows=1, cols=1, subplot_titles=["Loss Convergence"])
    for param_str, loss_lists in results.items():
        if not loss_lists:
            continue
        param_dict = eval(param_str)
        label = f"HD={param_dict['hidden_dim']}, NL={param_dict['num_layers']}, LR={param_dict['learning_rate']}, dt={param_dict['dt']}"
        fig_convergence.add_trace(
            go.Scatter(x=list(range(len(loss_lists[0]))), y=loss_lists[0], mode='lines', name=label),
            row=1, col=1
        )
    fig_convergence.update_xaxes(title_text="Epochs")
    fig_convergence.update_yaxes(title_text="Loss")
    fig_convergence.update_layout(title="Loss Convergence for Different Hyperparameters")

    param_div = p.to_html(fig_params, include_plotlyjs=False, full_html=False)
    convergence_div = p.to_html(fig_convergence, include_plotlyjs=False, full_html=False)

    combined_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PINN VDP Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .tab {{
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
            }}
            .tab button {{
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }}
            .tab button:hover {{
                background-color: #ddd;
            }}
            .tab button.active {{
                background-color: #ccc;
            }}
            .tabcontent {{
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
            }}
        </style>
    </head>
    <body>
        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'ParamEffects')" id="defaultOpen">Hyperparameter Effects</button>
            <button class="tablinks" onclick="openTab(event, 'Convergence')">Loss Convergence</button>
        </div>

        <div id="ParamEffects" class="tabcontent">
            {param_div}
        </div>

        <div id="Convergence" class="tabcontent">
            {convergence_div}
        </div>

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
        document.getElementById("defaultOpen").click();
        </script>
    </body>
    </html>
    '''

    combined_dashboard_path = os.path.join(save_path, 'combined_dashboard.html')
    with open(combined_dashboard_path, 'w') as f:
        f.write(combined_html)

    params_dashboard_path = os.path.join(save_path, 'hyperparameter_effects.html')
    convergence_dashboard_path = os.path.join(save_path, 'convergence_dashboard.html')

    p.write_html(fig_params, params_dashboard_path)
    p.write_html(fig_convergence, convergence_dashboard_path)

    print(f"Dashboards saved to {save_path}")
    webbrowser.open('file://' + os.path.abspath(combined_dashboard_path))

def main():
    parameter_ranges = {
        'input_dim': [2],
        'hidden_dim': [i for i in range(1, 101, 20)],
        'output_dim': [2],
        'num_layers': [i for i in range(1, 6)],
        'learning_rate': [i * 1e-3 for i in range(1, 3)],
        'epochs': [100],
        'dt': [1e-2]
    }
    all_params = generate_parameter_combinations(parameter_ranges)
    print(f"Generated {len(all_params)} parameter combinations")

    save_path = 'Van_Der_Pol_plots'
    os.makedirs(save_path, exist_ok=True)

    results = {}
    for params in all_params:
        print(f'Training parameters: {params}')
        results[str(params)] = []
        model = RK_PINN(
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            num_layers=params['num_layers']
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()

        y0 = torch.tensor([2.0, 0.0], dtype=torch.float32)
        t0 = 0.0

        losses = train(model, optimizer, criterion, y0, t0, params['dt'], params['epochs'])
        results[str(params)].append(losses)

    create_dashboard(results, save_path)

if __name__ == "__main__":
    main()
