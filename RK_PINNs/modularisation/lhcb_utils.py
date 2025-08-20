import numpy as np
import torch
from typing import Sequence, Union
from Classes.magnetic_field import LHCb_Field

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


def get_b_field(state: ArrayLike, B_interpolator: LHCb_Field) -> np.ndarray:
    """
    Query the magnetic field interpolator at the given position.

    Parameters
    - state: array-like with at least 3 entries [x, y, z, ...]. Can be numpy, torch tensor or a sequence.
    - B_interpolator: LHCb_Field instance providing interpolated_field(x, y, z) -> np.ndarray

    Returns
    - B: numpy array shape (3,) representing the magnetic field vector [Bx, By, Bz] at (x, y, z).
    """
    # Ensure we have a numpy array for the interpolator
    if isinstance(state, torch.Tensor):
        state_np = state.detach().cpu().numpy()
    else:
        state_np = np.asarray(state)

    x, y, z = state_np[0], state_np[1], state_np[2]
    B = B_interpolator.interpolated_field(x, y, z)
    return np.asarray(B)


def b_field(state: ArrayLike, B_interpolator: LHCb_Field) -> torch.Tensor:
    """
    Compute the derivatives of the LHCb tracking state under the magnetic field.

    The expected input state is:
        [x, y, z, tx, ty, q_over_p]
    where tx = dx/dz and ty = dy/dz.

    The returned tensor contains the derivatives with respect to z:
        [dx/dz, dy/dz, dz/dz, dtx/dz, dty/dz, dq_over_p/dz]

    Device-safety:
    - If `state` is a torch.Tensor, the returned tensor will use the same dtype and device.
    - If `state` is numpy/sequence, the return will be a CPU torch.tensor with default dtype.

    Parameters
    - state: array-like length >= 6 (numpy array, torch tensor or sequence)
    - B_interpolator: LHCb_Field instance

    Returns
    - torch.Tensor of shape (6,) with derivatives.
    """
    input_is_torch = isinstance(state, torch.Tensor)

    # Convert to numpy for interpolation and arithmetic (interpolator likely expects numpy)
    if input_is_torch:
        state_np = state.detach().cpu().numpy()
    else:
        state_np = np.asarray(state)

    if state_np.size < 6:
        raise ValueError("state must have at least 6 elements: [x, y, z, tx, ty, q_over_p]")

    x, y, z = state_np[0], state_np[1], state_np[2]
    tx, ty, q_p = state_np[3], state_np[4], state_np[5]

    # Query magnetic field (numpy)
    B = get_b_field(state_np, B_interpolator)
    Bx, By, Bz = B[0], B[1], B[2]

    # Kinematic derivatives (numpy)
    dx_dz = tx
    dy_dz = ty
    dz_dz = 1.0

    gamma_factor = np.sqrt(1.0 + tx ** 2 + ty ** 2)
    dtx_dz = q_p * gamma_factor * (ty * (tx * Bx + Bz) - (1.0 + tx ** 2) * By)
    dty_dz = -q_p * gamma_factor * (tx * (ty * By + Bz) - (1.0 + ty ** 2) * Bx)

    dq_p_dz = 0.0  # assumed constant in this model

    derivs_np = np.array([dx_dz, dy_dz, dz_dz, dtx_dz, dty_dz, dq_p_dz], dtype=float)

    # Convert back to torch and preserve dtype/device if input was torch
    if input_is_torch:
        return torch.as_tensor(derivs_np, dtype=state.dtype, device=state.device)
    else:
        return torch.as_tensor(derivs_np)