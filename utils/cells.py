import jax.numpy as jnp
import jax

def softplus(x, nl_gain, shift, beta):
    z = beta * (x - shift)
    sp = (jnp.log1p(jnp.exp(-jnp.abs(z))) + jnp.maximum(z, 0.0)) / beta
    return nl_gain * sp

def K_func(tau_j, tau_h, n, alpha_j, alpha_h, dur, dt, eps=1e-12):
    """
    filter paramtreization from Jun, Field, Pearson Neurips 2022 
    K(t) = alpha_j * t^n * exp(-t/tau_j) - alpha_h * t^n * exp(-t/tau_h) if t>=0 else 0
    Added parameter decoupling 
    """
    t = jnp.arange(0.0, dur, dt)         # <-- define t here
    x1 = t / (tau_j + eps)
    x2 = t / (tau_h + eps)
    K_j = alpha_j * (x1**n) * jnp.exp(-n*(x1 - 1.0))
    K_h = alpha_h * (x2**n) * jnp.exp(-n*(x2 - 1.0))
    return K_j - K_h


class LN_cell:
    def __init__(self,
                 n=6,
                 tau_j=0.06,
                 tau_h=0.15,
                 alpha_j=1.0,
                 alpha_h=1.0,
                 filt_dur=0.6,
                 dt=1/60,
                 norm="L2",          # None, "L2", or "var"
                 nl_gain=1.0,
                 nl_shift=0.0,
                 nl_beta=0.25,):        # alias for nl_shift

        self.n = n
        self.tau_j = tau_j
        self.tau_h = tau_h
        self.alpha_j = alpha_j
        self.alpha_h = alpha_h
        self.filt_dur = filt_dur
        self.dt = dt
        self.norm = norm

        self.nl_gain = nl_gain
        self.nl_shift = nl_shift 
        self.nl_beta = nl_beta

    def make_stim_filter(self, dt=None):
        if dt is None:
            dt = self.dt
        K = K_func(self.tau_j, self.tau_h, self.n,
                   self.alpha_j, self.alpha_h,
                   self.filt_dur, dt)
        if self.norm == "L2":
            print("Normalizing to unit norm")
            K = K / (jnp.sqrt(jnp.sum(K**2) * dt) + 1e-12) #so its time invariant
            
        if self.norm == "var":
            print("Normalizing to unit variance")
            K = K / (jnp.std(K) + 1e-12)
        return K

    def simulate(self, stim, dt=None, rng=None, return_components=False):
        """
        stim: (T,) or (N,T)
        returns: rate, L shaped like stim; k is (K_len,)
        """
        if dt is None:
            dt = self.dt

        stim = jnp.asarray(stim)
        single = (stim.ndim == 1)
        if single:
            stim_b = stim[None, :]      # (1,T)
        elif stim.ndim == 2:
            stim_b = stim               # (N,T)
        else:
            raise ValueError(f"stim must be 1D or 2D, got shape {stim.shape}")

        K = self.make_stim_filter(dt)

        # 1D conv for one trial
        def conv1(tr):
            return jnp.convolve(tr, K, mode="full")[: tr.shape[0]]

        # batch over trials
        L_b = jax.vmap(conv1, in_axes=0)(stim_b)  # (N,T)
        rate_b = softplus(L_b, self.nl_gain, self.nl_shift, self.nl_beta)

        if single:
            out = dict(rate=rate_b[0], L=L_b[0], k=K)
        else:
            out = dict(rate=rate_b, L=L_b, k=K)

        return out
