import jax
import jax.numpy as jnp
from jax.nn import sigmoid, softplus


# ---------- 1) Voltage -> rate (batch) ----------
def voltage_to_rate_softplus_clipped(V, vth=-45.0, beta=0.5, gain_hz=85.0, vcap=-20.0, eps=1e-8):
    Vc = jnp.minimum(V, vcap)
    return gain_hz * softplus(beta * (Vc - vth)) + eps

# ---------- 2) Precompute frame edges once ----------
def make_edges(T_v, sr_v=50_000.0, fr=60.0):
    T_f = int(jnp.floor(T_v * fr / sr_v))
    edges = jnp.rint(jnp.arange(T_f + 1) * (sr_v / fr)).astype(jnp.int32)
    edges = jnp.clip(edges, 0, T_v)
    return edges  # (T_f+1,)

def rate_to_mu_with_edges(rate_hz, edges, sr_v=50_000.0):
    dt_v = 1.0 / sr_v
    cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(rate_hz) * dt_v])  # (T_v+1,)
    return cum[edges[1:]] - cum[edges[:-1]]                                # (T_f,)

def voltage_to_rate_sigmoid_clipped(V, vth=-45.0, beta=0.5, r_max_hz=150.0, vcap=-20.0, eps=1e-8):
    Vc = jnp.minimum(V, vcap)
    p  = 1.0 / (1.0 + jnp.exp(-beta * (Vc - vth)))
    return r_max_hz * p + eps

# ---------- 3) Vectorize across trials ----------
def voltage_batch_to_mu(V_batch, edges, vth=-45.0, beta=0.5, r_max_hz=150.0, vcap=-20.0,
                        sr_v=50_000.0):
    """
    V_batch: (N, T_v)
    returns mu_batch: (N, T_f)
    """
    # vmap voltage->rate
    rate_batch = jax.vmap(
        lambda V: voltage_to_rate_sigmoid_clipped(V, vth=vth, beta=beta, r_max_hz=r_max_hz, vcap=vcap),
        in_axes=0
    )(V_batch)

    # vmap rate->mu
    mu_batch = jax.vmap(
        lambda r: rate_to_mu_with_edges(r, edges, sr_v=sr_v),
        in_axes=0
    )(rate_batch)

    return rate_batch, mu_batch