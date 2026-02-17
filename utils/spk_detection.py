import jax
import jax.numpy as jnp
from jax import lax

# -----------------------------
# 1) Upward-crossing + refractory spike detector (JAX)
# -----------------------------
def upward_crossings(V, vth):
    """Boolean array marking upward threshold crossings."""
    above = V >= vth
    # crossing at t if above[t] and not above[t-1]
    return jnp.concatenate([jnp.array([False]), above[1:] & (~above[:-1])], axis=0)

def enforce_refractory(crossings, refrac_samps: int):
    """
    crossings: (T,) bool
    returns:   (T,) bool accepted spikes with absolute refractory
    """
    def step(last_spk_t, x):
        t, c = x
        ok = c & ((t - last_spk_t) >= refrac_samps)
        last_spk_t_new = jnp.where(ok, t, last_spk_t)
        return last_spk_t_new, ok

    T = crossings.shape[0]
    t_idx = jnp.arange(T, dtype=jnp.int32)
    init_last = jnp.int32(-10**9)  # effectively -inf
    _, accepted = lax.scan(step, init_last, (t_idx, crossings))
    return accepted

def detect_spikes_one(V, *, vth, dt_v, refrac_ms=2.0):
    """
    V: (T_v,)
    returns:
      spk_bool: (T_v,) bool
      spk_idx_padded: (Smax,) int32 (filled with -1)
      n_spk: int32
    """
    refrac_samps = jnp.int32(jnp.round((refrac_ms * 1e-3) / dt_v))
    crossings = upward_crossings(V, vth)
    spk_bool = enforce_refractory(crossings, refrac_samps)

    n_spk = spk_bool.sum().astype(jnp.int32)
    # padded indices (static shape), convenient for jit/vmap
    spk_idx_padded = jnp.where(spk_bool, size=spk_bool.shape[0], fill_value=-1)[0].astype(jnp.int32)
    return spk_bool, spk_idx_padded, n_spk

# -----------------------------
# 2) Bin spikes into frames (60 Hz) using precomputed edges
# -----------------------------
def spike_bool_to_frame_counts(spk_bool, edges):
    """
    spk_bool: (T_v,) bool
    edges: (T_f+1,) int32 sample indices
    returns counts_per_frame: (T_f,) int32
    """
    spk_int = spk_bool.astype(jnp.int32)
    cum = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(spk_int)], axis=0)  # (T_v+1,)
    return cum[edges[1:]] - cum[edges[:-1]]  # (T_f,)

# -----------------------------
# 3) Full batch wrapper
# -----------------------------
def voltage_batch_to_spikes(
    V_batch, *,
    dt_v,              # seconds/sample (same units as your dt_cell)
    edges,             # from make_edges(...)
    vth=0.0,           # mV threshold for spike detection
    refrac_ms=2.0
):
    """
    V_batch: (N, T_v)
    returns:
      spk_bool_batch: (N, T_v) bool
      spk_idx_padded: (N, T_v) int32 (padded with -1)
      n_spk:          (N,) int32
      spk_counts_60:  (N, T_f) int32  (spike train at 60Hz as counts/frame)
    """
    spk_bool_batch, spk_idx_padded, n_spk = jax.vmap(
        lambda V: detect_spikes_one(V, vth=vth, dt_v=dt_v, refrac_ms=refrac_ms),
        in_axes=0
    )(V_batch)

    spk_counts_60 = jax.vmap(lambda b: spike_bool_to_frame_counts(b, edges), in_axes=0)(spk_bool_batch)
    return spk_bool_batch, spk_idx_padded, n_spk, spk_counts_60

# -----------------------------
# 4) Convert padded indices -> padded spike times (seconds)
# -----------------------------
def idx_to_times(spk_idx_padded, dt_v):
    # -1 stays -1*dt_v; we'll mask later using n_spk
    return spk_idx_padded.astype(jnp.float32) * dt_v
