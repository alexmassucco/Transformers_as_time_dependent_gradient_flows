"""
Core simulation engine for attention gradient flows on the sphere.

Ported from simulate.py and reorganized into a reusable module.
"""

import math
import numpy as np
import torch

# =============================================================================
#  Core batched forward pass
#  x  : [P, n, d]
#  Ds : [P, H, d, d]
# =============================================================================


@torch.no_grad()
def _forward_metrics(x, Ds, drift_fn, noise_var, t):
    P, n, d = x.shape
    H = Ds.shape[1]
    H_n2_inv = 1.0 / (H * n * n)
    noise_coef = noise_var / 4.0

    f_drift = drift_fn(Ds, t)

    # Dx: [P, H, n, d]
    Dx = torch.einsum("phde,pne->phnd", Ds, x)

    # logits / softmax weights: [P, H, n, n]
    logits = torch.einsum("pnd,phmd->phnm", x, Dx)
    U = torch.exp(logits)
    N = U.sum(dim=-1, keepdim=True)  # [P, H, n, 1]
    A = U / N  # [P, H, n, n]

    # Energy per path: [P]
    E_now = 0.5 * U.sum(dim=(1, 2, 3)) * H_n2_inv

    # dE/dt per path: [P]
    fDx = torch.einsum("phde,pne->phnd", f_drift, x)
    drift_term = torch.einsum("pnd,phmd->phnm", x, fDx)
    xxT = torch.matmul(x, x.transpose(-1, -2))  # [P, n, n]
    ito_term = noise_coef * (xxT.pow(2) + 1.0)[:, None, :, :]  # [P,1,n,n]
    dEdt = 0.5 * (U * (drift_term + ito_term)).sum(dim=(1, 2, 3)) * H_n2_inv

    # Velocity field and metric slope G^2: [P]
    weighted = torch.einsum("phnm,phmd->phnd", A, Dx)  # [P, H, n, d]
    v_dot_x = (weighted * x[:, None]).sum(dim=-1, keepdim=True)  # [P, H, n, 1]
    proj = weighted - v_dot_x * x[:, None]  # tangent projection
    v_flow = proj.mean(dim=1)  # [P, n, d]

    sq_norms = (v_flow * v_flow).sum(dim=-1)  # [P, n]
    denom = (n / N.squeeze(-1)).mean(dim=1)  # [P, n]
    G2_now = (sq_norms / denom).mean(dim=1)  # [P]

    return f_drift, U, E_now, dEdt, G2_now, v_flow


# =============================================================================
#  Batched simulation kernel
#  Returns arrays with a leading P axis (one entry per parallel path).
# =============================================================================


@torch.no_grad()
def simulate_batch(
    x0_np,
    Ds_init,
    drift_fn,
    noise_var,
    device,
    T,
    dt,
    rec,
    n_paths=1,
    seed=0,
    store_traj=False,
):
    P = n_paths
    x0 = torch.as_tensor(x0_np, dtype=torch.float64, device=device)
    Ds0 = Ds_init.to(device=device, dtype=torch.float64)

    x = x0.unsqueeze(0).expand(P, -1, -1).clone()  # [P, n, d]
    Ds = Ds0.unsqueeze(0).expand(P, -1, -1, -1).clone()  # [P, H, d, d]

    g = float(np.sqrt(noise_var))
    sqrt_dt = math.sqrt(dt)
    n_steps = int(round(T / dt))

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Accumulators
    dE_int = torch.zeros(P, dtype=torch.float64, device=device)
    G2_int = torch.zeros(P, dtype=torch.float64, device=device)
    add_int = torch.zeros(P, dtype=torch.float64, device=device)

    # Recording lists
    rec_t = []
    rec_E = []
    rec_G2 = []
    rec_dE_int = []
    rec_G2_int = []
    rec_add_int = []
    rec_mean_x = []  # expected trajectory per token [P, n, d]
    rec_traj = [] if store_traj else None

    # Forward pass state needed inside record() closure
    E_now = torch.zeros(P, dtype=torch.float64, device=device)
    G2_now = torch.zeros(P, dtype=torch.float64, device=device)

    def record(t_val):
        rec_t.append(t_val)
        rec_E.append(E_now.detach().cpu().numpy().copy())
        rec_G2.append(G2_now.detach().cpu().numpy().copy())
        rec_dE_int.append(dE_int.detach().cpu().numpy().copy())
        rec_G2_int.append(G2_int.detach().cpu().numpy().copy())
        rec_add_int.append(add_int.detach().cpu().numpy().copy())
        rec_mean_x.append(x.detach().cpu().numpy().copy())  # [P, n, d]
        if store_traj:
            rec_traj.append(x.detach().cpu().numpy().copy())  # [P, n, d]

    for k in range(n_steps):
        t_now = k * dt

        f_drift, U, E_now, dEdt, G2_now, v_flow = _forward_metrics(
            x, Ds, drift_fn, noise_var, t_now
        )

        if k % rec == 0:
            record(t_now)

        dE_int += dEdt * dt
        G2_int += G2_now * dt

        # Euler step for x, then project back to sphere
        x_new = x - dt * v_flow
        x = x_new / torch.norm(x_new, dim=-1, keepdim=True)

        # Euler-Maruyama step for Ds
        Ds = Ds + f_drift * dt
        if noise_var > 0.0:
            Z = torch.randn(Ds.shape, dtype=Ds.dtype, device=device, generator=gen)
            dW_sym = sqrt_dt * 0.5 * (Z + Z.transpose(-2, -1))
            Ds = Ds + g * dW_sym

            gWx = torch.einsum("phde,pne->phnd", g * dW_sym, x)
            stoch_term = torch.einsum("pnd,phmd->phnm", x, gWx)
            H_ = Ds.shape[1]
            n_tok = x.shape[1]
            add_int += (
                0.5 * (U * stoch_term).sum(dim=(1, 2, 3)) / (H_ * n_tok * n_tok)
            )

    # Final snapshot
    f_drift, U, E_now, dEdt, G2_now, v_flow = _forward_metrics(
        x, Ds, drift_fn, noise_var, T
    )
    if len(rec_t) == 0 or rec_t[-1] < T - 1e-12:
        record(T)

    out = dict(
        t=np.array(rec_t),
        E=np.stack(rec_E, axis=1),  # [P, m]
        G2=np.stack(rec_G2, axis=1),  # [P, m]
        dE_int=np.stack(rec_dE_int, axis=1),  # [P, m]
        G2_int=np.stack(rec_G2_int, axis=1),  # [P, m]
        add_int=np.stack(rec_add_int, axis=1),  # [P, m]
        mean_traj=np.stack(rec_mean_x, axis=1),  # [P, m, n, d]
    )
    if store_traj:
        out["traj"] = np.stack(rec_traj, axis=1)  # [P, m, n, d]
    return out


# =============================================================================
#  Reduction helpers
# =============================================================================


def _reduce_single_path(raw):
    """Strip the P=1 leading axis and compute balance identity arrays."""
    out = dict(
        t=raw["t"],
        E=raw["E"][0],
        G2=raw["G2"][0],
        dE_int=raw["dE_int"][0],
        G2_int=raw["G2_int"][0],
        add_int=raw["add_int"][0],
        mean_traj=raw["mean_traj"][0],  # [m, n, d]
    )
    if "traj" in raw:
        out["traj"] = raw["traj"][0]  # [m, n, d]
    out["balance_lhs"] = out["E"] - out["E"][0]
    out["balance_rhs"] = out["dE_int"] - out["G2_int"] + out["add_int"]
    return out


def _reduce_mc_batches(batches, total_paths):
    """
    Online mean + variance over all MC batches.
    Returns mean arrays plus *_std for E, G2, dE_int, G2_int, add_int.
    mean_traj is averaged (no std needed for the sphere plots).
    """
    scalar_keys = ["E", "G2", "dE_int", "G2_int", "add_int"]
    sums = {k: None for k in scalar_keys}
    sums_sq = {k: None for k in scalar_keys}
    mt_sum = None  # mean_traj accumulator  [m, n, d]
    t_ref = None

    for raw in batches:
        if t_ref is None:
            t_ref = raw["t"]
            mt_sum = raw["mean_traj"].sum(axis=0)  # [m, n, d]
        else:
            mt_sum += raw["mean_traj"].sum(axis=0)

        for k in scalar_keys:
            arr = raw[k]  # [P, m]
            s = arr.sum(axis=0)
            ss = (arr * arr).sum(axis=0)
            if sums[k] is None:
                sums[k] = s
                sums_sq[k] = ss
            else:
                sums[k] += s
                sums_sq[k] += ss

    out = {"t": t_ref}
    for k in scalar_keys:
        mean = sums[k] / total_paths
        var = np.maximum(sums_sq[k] / total_paths - mean**2, 0.0)
        out[k] = mean
        out[f"{k}_std"] = np.sqrt(var)

    out["mean_traj"] = mt_sum / total_paths  # [m, n, d]
    out["balance_lhs"] = out["E"] - out["E"][0]
    out["balance_rhs"] = out["dE_int"] - out["G2_int"] + out["add_int"]
    return out


# =============================================================================
#  Public simulate() dispatcher
# =============================================================================


def simulate(
    x0_np,
    Ds_init,
    drift_fn,
    noise_var,
    device,
    T,
    dt,
    rec,
    seed=None,
    n_paths=1,
    mc_batch=10,
    store_traj=False,
):
    """
    Run simulation and return a result dict.

    Deterministic (noise_var == 0) or explicitly single-path: one trajectory.
    Noisy + n_paths > 1: mini-batched Monte Carlo; returns MC-mean scalars
    with *_std arrays, and averaged mean_traj.
    """
    base_seed = 0 if seed is None else seed

    if noise_var == 0.0 or n_paths <= 1:
        raw = simulate_batch(
            x0_np,
            Ds_init,
            drift_fn,
            noise_var,
            device,
            T,
            dt,
            rec,
            n_paths=1,
            seed=base_seed,
            store_traj=store_traj,
        )
        return _reduce_single_path(raw)

    # Noisy MC run
    batches = []
    done = 0
    while done < n_paths:
        bsz = min(mc_batch, n_paths - done)
        raw = simulate_batch(
            x0_np,
            Ds_init,
            drift_fn,
            noise_var,
            device,
            T,
            dt,
            rec,
            n_paths=bsz,
            seed=base_seed + done,
            store_traj=False,
        )
        batches.append(raw)
        done += bsz

    return _reduce_mc_batches(batches, total_paths=n_paths)
