import jax
import jax.numpy as jnp
from jax.nn import sigmoid, softplus
import jaxley as jx
from jax.lax import stop_gradient
# from jax.numpy.linalg import norm

def K_func(tau_j, tau_h, n, alpha_j, alpha_h, K_T, dt, eps=1e-12):
    # JIT-safe: arange uses an integer stop, not a tracer step
    t = jnp.arange(K_T, dtype=jnp.float32) * dt

    x1 = t / (tau_j + eps)
    x2 = t / (tau_h + eps)
    K_j = alpha_j * (x1**n) * jnp.exp(-n*(x1 - 1.0))
    K_h = alpha_h * (x2**n) * jnp.exp(-n*(x2 - 1.0))
    return K_j - K_h

def ln_forward(params, stim, *, K, dt, n, max_tau, norm="L2", eps=1e-12):
    # --- time constants (bounded to (0, max_tau)) ---
    tau_j = sigmoid(params["logit_tau1"]) * max_tau
    tau_h = sigmoid(params["logit_tau2"]) * max_tau

    # --- biphasic kernel ---
    k = K_func(tau_j, tau_h, n, params["alpha_j"], params["alpha_h"], K, dt)

    if norm == "L2":
        k = k / (jnp.sqrt(jnp.sum(k**2) * dt) + eps)

    # --- convolution per trial ---
    def conv1(tr):
        return jnp.convolve(tr, k, mode="full")[: tr.shape[0]]

    L = jax.vmap(conv1)(stim) if stim.ndim == 2 else conv1(stim)

    # --- static nonlinearity ---
    beta  = softplus(params["log_beta"]) + 1e-6  # slope > 0
    shift = params["shift"]
    drive = softplus(beta * (L - shift))

    return drive, L, k

def current_map(drive, params):
    # params contains DC and scale
    return params["DC"] + params["scale"] * drive  # pA, at 60 Hz

def upsample_zoh_chunk_repeat(I_frame_one: jnp.ndarray, 
                              t0_f: int, 
                              t1_f: int,
                              samples_per_frame: int) -> jnp.ndarray:
    """
    I_frame_one: (T_f,) current in nA at frame rate.
    Returns: (T_chunk_cell,) current in nA at dt_cell (ZOH).
    """
    I_chunk = I_frame_one[t0_f:t1_f]                  # (F,)
    return jnp.repeat(I_chunk, samples_per_frame)     # (F*samples_per_frame,)

def choose_checkpoint_lengths(T, chunk=2048):
    # ensures prod(checkpoint_lengths) >= T
    # single-level is simplest
    return [int(((T + chunk - 1) // chunk) * chunk)]

def make_rgc_forward_with_states(cell, dt_cell, warmup_ms=200.0):
    _, init_states = jx.integrate(cell, t_max=warmup_ms, delta_t=dt_cell, return_states=True)

    def simulate_one(I_one):
        data_stimuli = cell.data_stimulate(current=I_one, data_stimuli=None, verbose=False)
        T = I_one.shape[0]
        ckpt = choose_checkpoint_lengths(T, chunk=2048)

        v = jx.integrate(
            cell,
            delta_t=dt_cell,
            data_stimuli=data_stimuli,
            all_states=stop_gradient(init_states),
            checkpoint_lengths=ckpt,
        )
        return v[0, :-1]

    def forward(I_syn):
        if I_syn.ndim == 1:
            I_syn_b = I_syn[None, :]
        else:
            I_syn_b = I_syn
        return jax.vmap(simulate_one)(I_syn_b)

    return forward
