import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Classes.magnetic_field import MagneticField, Quadratic_Field, LHCb_Field

# ===========================
# 0. Set up the Problem
# ===========================

data = 'Data/twodip.rtf'
Qfield = Quadratic_Field(0)
LHCbField = LHCb_Field(data)

# ===========================
# 1. Define RK4 Step Function
# ===========================

def rk4_step(s, dz, f, z, B):
    """
    Computes a single Runge-Kutta 4th order step.

    Parameters:
    - s: Tensor, state vector (x, y, tx, tz)
    - dz: Step size
    - f: Function computing ds/dz = f(s, t)
    - z: Current time step

    Returns:
    - s_next: Estimated next state using RK4
    """

    k1 = f(s, z, B)
    k2 = f(s + dz * k1 / 2, z + dz / 2, B)
    k3 = f(s + dz * k2 / 2, z + dz / 2, B)
    k4 = f(s + dz * k3, z + dz, B)

    K = torch.stack([k1, k2, k3, k4]).reshape(32, -1)
    
    return K

# ===========================
# 2. Define Custom RK4 Loss Function
# ===========================

def rk4_loss(model, s, dz, f, z, B):
    """
    Custom loss function enforcing RK4 constraints.

    Parameters:
    - model: Neural network approximating s(z)
    - s: Current state
    - dz: Step size
    - f: Function computing ds/dz = f(s, t)
    - t: Current time step

    Returns:
    - loss: RK4 residual loss
    """
    s_pred = model(s, z, B)  # Neural network output
    s_rk4 = rk4_step(s, dz, f, z, B)  # RK4 computed next step


    # Compute loss: Enforce that model output should match RK4 step
    loss = torch.mean(torch.norm(s_pred - s_rk4, p=2) ** 2)
    return loss

# ===========================
# 3. Define Neural Network
# ===========================

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)  # Input layer
        self.layer2 = nn.Linear(64, 32)  # Hidden layer
        self.layer3 = nn.Linear(32, 20)  # Output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, s, z, B):
        x,y,z = s[:,0], s[:,1], z
        B = torch.tensor(B.interpolated_field(x, y, z)).T
        s = torch.cat([s, B], dim=1)  # Concatenate B field to input
        s = self.relu(self.layer1(s))
        s = self.relu(self.layer2(s))
        s = self.layer3(s)  # Output state vector
        return s

# ===========================
# 4. Define Function f(s, t)
# ===========================

def f(s, z, B):
    """
    Computes ds/dz = f(s, t), where s = (x, y, tx, tz)

    Parameters:
    - s: State vector (Tensor)
    - t: Time scalar

    Returns:
    - ds/dz: Tensor of the same shape as s
    """
    x, y, tx, ty, q_p = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]

    # Interpolate magnetic field at current position
    B = B.interpolated_field(x, y, z)

    dx_dz = tx
    dy_dz = ty
    dtx_dz = q_p * np.sqrt(1 + tx**2 + ty**2) * (ty*(tx*B[0] + B[2]) - (1 + tx**2)*B[1])
    dtz_dz = -q_p * np.sqrt(1 + tx**2 + ty**2) * (tx*(ty*B[1] + B[2]) - (1 + ty**2)*B[0])
    print('====================')
    print(f'B: {B}')
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'tx: {tx}')
    print(f'ty: {ty}')
    print(f'q_p: {q_p}')
    print(f'dx_dz: {dx_dz}')
    print(f'dy_dz: {dy_dz}')
    print(f'dtx_dz: {dtx_dz}')
    print(f'dtz_dz: {dtz_dz}')
    print(f'q_p: {q_p}')

    return torch.stack([dx_dz, dy_dz, dtx_dz, dtz_dz, q_p], dim=1)

# ===========================
# 5. Load and Prepare Data
# ===========================

# Load dataset from CSV
df = pd.read_csv("simulated_data_for_pinn").iloc[:,1:]  # Replace with your actual file path
Z = df['z'].values # z values

X = df.iloc[:, :].drop(columns=['z']).values  # Features (all columns except z)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Z = torch.tensor(Z, dtype=torch.float32)

# ===========================
# 6. Train the Model
# ===========================

# Initialize model
input_size = X.shape[1] + 3  # Number of features
model = NeuralNet(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

B = Quadratic_Field(1e-7)

epochs = 1000
batch_size = 32  # Mini-batch gradient descent
train_size = X.shape[0]
dz = 0.1  # Step size

for epoch in range(1):
    model.train()  # Set model to training mode
    epoch_loss = 0

    for i in range(0, train_size, batch_size):
        X_batch = X[i:i + batch_size]
        z = Z[i:i + batch_size]

        optimizer.zero_grad()  # Zero gradients
        loss = rk4_loss(model, X_batch, dz, f, z, B)  # Compute custom RK4 loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        epoch_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss / (train_size/batch_size):.6f}")

# ===========================
# 7. Evaluate the Model
# ===========================

model.eval()  # Set model to evaluation mode
with torch.no_grad():  # No gradients needed
    y_test_pred = model(X)
    test_loss = rk4_loss(model, X, dz, f, t=0).item()

print(f"Test Loss: {test_loss:.6f}")
