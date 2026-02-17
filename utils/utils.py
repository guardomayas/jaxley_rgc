import jax.numpy as jnp
from jax.nn import softplus
import jax
def upsample_hold(x, dt_stim=1/60, dt_cell=0.025e-3):
    """
    Zero-order hold upsampling.

    x: (T,) or (trials, T) or (..., T)
    dt_stim: seconds per LN sample (e.g. 1/60)
    dt_cell: seconds per biophys step (e.g. 0.025 ms = 0.000025 s)

    returns: x_up with time axis length T_up = T * rep
    """
    x = jnp.asarray(x)
    rep = int(jnp.round(dt_stim / dt_cell))
    if rep < 1:
        raise ValueError(f"dt_cell ({dt_cell}) must be <= dt_stim ({dt_stim}).")

    # repeat along last axis (time)
    x_up = jnp.repeat(x, rep, axis=-1)
    return x_up, rep

def upsample_hold_edges(I_frame, dt_stim, dt_cell, T_cell):
    # frame index for each cell time bin
    t_cell = jnp.arange(T_cell) * dt_cell
    idx = jnp.floor(t_cell / dt_stim).astype(jnp.int32)
    idx = jnp.clip(idx, 0, I_frame.shape[-1]-1)
    return I_frame[..., idx]  # broadcasts over batch

# def current_map(drive, params):
#     # params contains DC and scale
#     return params["DC"] + params["scale"] * drive  # pA, at 60 Hz
