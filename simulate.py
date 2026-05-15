"""
simulate.py
===========
GPU simulation stage.

Case 1  – deterministic oscillating D(t), H=1, run on GPU 0.
Case 2  – symmetric OU weights, H in {1, 10, 100}, each on its own GPU (1-3).
           Noisy runs use batched Monte-Carlo: N_MC trajectories are simulated
           in mini-batches of MC_BATCH on a single GPU, then averaged.

Run this first, then run plot.py.
"""

import os
import math
import numpy as np
import torch
import torch.multiprocessing as mp

RESULTS_DIR = '/store/CIA/am3270/Projects/TransformerDynamics/results/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Hyper-parameters --------------------------------------------------------
T_MAX        = 20.0
dt_sim       = 5e-3
rec_interval = 5

T_OU         = 20.0
dt_ou        = 5e-3
rec_ou       = 5

n1 = 100
n2 = 200

# Monte Carlo settings (only used when noise_var > 0)
N_MC     = 10   # total number of noisy trajectories to average
MC_BATCH = 10   # number simulated simultaneously on one GPU

# =============================================================================
#  Initial tokens
# =============================================================================

def make_initial_tokens():
    rng = np.random.default_rng(42)
    base_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a1    = rng.choice(base_angles, n1) + rng.normal(0, 0.05, n1)
    x0_S1 = np.stack([np.cos(a1), np.sin(a1)], axis=-1).astype(np.float32)

    verts  = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    pts    = verts[rng.choice(len(verts), n2)] + rng.normal(0, 0.05, (n2, 3)).astype(np.float32)
    x0_S2  = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
    return x0_S1, x0_S2

# =============================================================================
#  Core batched forward pass
#  x  : [P, n, d]
#  Ds : [P, H, d, d]
# =============================================================================

@torch.no_grad()
def _forward_metrics(x, Ds, drift_fn, noise_var, t):
    P, n, d = x.shape
    H = Ds.shape[1]
    H_n2_inv   = 1.0 / (H * n * n)
    noise_coef = noise_var / 4.0

    f_drift = drift_fn(Ds, t)

    # Dx: [P, H, n, d]
    Dx = torch.einsum('phde,pne->phnd', Ds, x)

    # logits / softmax weights: [P, H, n, n]
    logits = torch.einsum('pnd,phmd->phnm', x, Dx)
    U      = torch.exp(logits)
    N      = U.sum(dim=-1, keepdim=True)        # [P, H, n, 1]
    A      = U / N                              # [P, H, n, n]

    # Energy per path: [P]
    E_now = 0.5 * U.sum(dim=(1, 2, 3)) * H_n2_inv

    # dE/dt per path: [P]
    fDx        = torch.einsum('phde,pne->phnd', f_drift, x)
    drift_term = torch.einsum('pnd,phmd->phnm', x, fDx)
    xxT        = torch.matmul(x, x.transpose(-1, -2))              # [P, n, n]
    ito_term   = noise_coef * (xxT.pow(2) + 1.0)[:, None, :, :]   # [P,1,n,n]
    dEdt       = 0.5 * (U * (drift_term + ito_term)).sum(dim=(1, 2, 3)) * H_n2_inv

    # Velocity field and metric slope G^2: [P]
    weighted  = torch.einsum('phnm,phmd->phnd', A, Dx)             # [P, H, n, d]
    v_dot_x   = (weighted * x[:, None]).sum(dim=-1, keepdim=True)  # [P, H, n, 1]
    proj      = weighted - v_dot_x * x[:, None]                    # tangent projection
    v_flow    = proj.mean(dim=1)                                    # [P, n, d]

    sq_norms = (v_flow * v_flow).sum(dim=-1)                       # [P, n]
    denom    = (n / N.squeeze(-1)).mean(dim=1)                     # [P, n]
    G2_now   = (sq_norms / denom).mean(dim=1)                      # [P]

    return f_drift, U, E_now, dEdt, G2_now, v_flow

# =============================================================================
#  Batched simulation kernel
#  Returns arrays with a leading P axis (one entry per parallel path).
# =============================================================================

@torch.no_grad()
def simulate_batch(
    x0_np, Ds_init, drift_fn, noise_var, device, T, dt, rec,
    n_paths=1, seed=0, store_traj=False
):
    P      = n_paths
    x0     = torch.as_tensor(x0_np, dtype=torch.float64, device=device)
    Ds0    = Ds_init.to(device=device, dtype=torch.float64)

    x      = x0.unsqueeze(0).expand(P, -1, -1).clone()           # [P, n, d]
    Ds     = Ds0.unsqueeze(0).expand(P, -1, -1, -1).clone()      # [P, H, d, d]

    g       = float(np.sqrt(noise_var))
    sqrt_dt = math.sqrt(dt)
    n_steps = int(round(T / dt))

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Accumulators
    dE_int  = torch.zeros(P, dtype=torch.float64, device=device)
    G2_int  = torch.zeros(P, dtype=torch.float64, device=device)
    add_int = torch.zeros(P, dtype=torch.float64, device=device)

    # Recording lists
    rec_t       = []
    rec_E       = []
    rec_G2      = []
    rec_dE_int  = []
    rec_G2_int  = []
    rec_add_int = []
    rec_mean_x  = []                        # expected trajectory per token [P, n, d]
    rec_traj    = [] if store_traj else None

    # Forward pass state needed inside record() closure
    E_now  = torch.zeros(P, dtype=torch.float64, device=device)
    G2_now = torch.zeros(P, dtype=torch.float64, device=device)

    def record(t_val):
        rec_t.append(t_val)
        rec_E.append(      E_now.detach().cpu().numpy().copy())
        rec_G2.append(     G2_now.detach().cpu().numpy().copy())
        rec_dE_int.append( dE_int.detach().cpu().numpy().copy())
        rec_G2_int.append( G2_int.detach().cpu().numpy().copy())
        rec_add_int.append(add_int.detach().cpu().numpy().copy())
        rec_mean_x.append( x.detach().cpu().numpy().copy())              # [P, n, d]
        if store_traj:
            rec_traj.append(x.detach().cpu().numpy().copy())             # [P, n, d]

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
        x     = x_new / torch.norm(x_new, dim=-1, keepdim=True)

        # Euler-Maruyama step for Ds
        Ds = Ds + f_drift * dt
        if noise_var > 0.0:
            Z      = torch.randn(Ds.shape, dtype=Ds.dtype, device=device, generator=gen)
            dW_sym = sqrt_dt * 0.5 * (Z + Z.transpose(-2, -1))
            Ds     = Ds + g * dW_sym

            gWx        = torch.einsum('phde,pne->phnd', g * dW_sym, x)
            stoch_term = torch.einsum('pnd,phmd->phnm', x, gWx)
            H_     = Ds.shape[1]
            n_tok  = x.shape[1]
            add_int += 0.5 * (U * stoch_term).sum(dim=(1, 2, 3)) / (H_ * n_tok * n_tok)

    # Final snapshot
    f_drift, U, E_now, dEdt, G2_now, v_flow = _forward_metrics(
        x, Ds, drift_fn, noise_var, T
    )
    if len(rec_t) == 0 or rec_t[-1] < T - 1e-12:
        record(T)

    out = dict(
        t         = np.array(rec_t),
        E         = np.stack(rec_E,        axis=1),    # [P, m]
        G2        = np.stack(rec_G2,       axis=1),    # [P, m]
        dE_int    = np.stack(rec_dE_int,   axis=1),    # [P, m]
        G2_int    = np.stack(rec_G2_int,   axis=1),    # [P, m]
        add_int   = np.stack(rec_add_int,  axis=1),    # [P, m]
        mean_traj = np.stack(rec_mean_x,   axis=1),    # [P, m, n, d]
    )
    if store_traj:
        out['traj'] = np.stack(rec_traj, axis=1)       # [P, m, n, d]
    return out

# =============================================================================
#  Reduction helpers
# =============================================================================

def _reduce_single_path(raw):
    """Strip the P=1 leading axis and compute balance identity arrays."""
    out = dict(
        t         = raw['t'],
        E         = raw['E'][0],
        G2        = raw['G2'][0],
        dE_int    = raw['dE_int'][0],
        G2_int    = raw['G2_int'][0],
        add_int   = raw['add_int'][0],
        mean_traj = raw['mean_traj'][0],   # [m, n, d]
    )
    if 'traj' in raw:
        out['traj'] = raw['traj'][0]       # [m, n, d]
    out['balance_lhs'] = out['E'] - out['E'][0]
    out['balance_rhs'] = out['dE_int'] - out['G2_int'] + out['add_int']
    return out

def _reduce_mc_batches(batches, total_paths):
    """
    Online mean + variance over all MC batches.
    Returns mean arrays plus *_std for E, G2, dE_int, G2_int, add_int.
    mean_traj is averaged (no std needed for the sphere plots).
    """
    scalar_keys = ['E', 'G2', 'dE_int', 'G2_int', 'add_int']
    sums    = {k: None for k in scalar_keys}
    sums_sq = {k: None for k in scalar_keys}
    mt_sum  = None          # mean_traj accumulator  [m, n, d]
    t_ref   = None

    for raw in batches:
        if t_ref is None:
            t_ref  = raw['t']
            mt_sum = raw['mean_traj'].sum(axis=0)   # [m, n, d]
        else:
            mt_sum += raw['mean_traj'].sum(axis=0)

        for k in scalar_keys:
            arr = raw[k]            # [P, m]
            s   = arr.sum(axis=0)
            ss  = (arr * arr).sum(axis=0)
            if sums[k] is None:
                sums[k]    = s
                sums_sq[k] = ss
            else:
                sums[k]    += s
                sums_sq[k] += ss

    out = {'t': t_ref}
    for k in scalar_keys:
        mean            = sums[k] / total_paths
        var             = np.maximum(sums_sq[k] / total_paths - mean**2, 0.0)
        out[k]          = mean
        out[f'{k}_std'] = np.sqrt(var)

    out['mean_traj']   = mt_sum / total_paths    # [m, n, d]
    out['balance_lhs'] = out['E'] - out['E'][0]
    out['balance_rhs'] = out['dE_int'] - out['G2_int'] + out['add_int']
    return out

# =============================================================================
#  Public simulate() dispatcher
# =============================================================================

def simulate(
    x0_np, Ds_init, drift_fn, noise_var, device, T, dt, rec,
    seed=None, n_paths=1, mc_batch=MC_BATCH, store_traj=False
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
            x0_np, Ds_init, drift_fn, noise_var, device, T, dt, rec,
            n_paths=1, seed=base_seed, store_traj=store_traj
        )
        return _reduce_single_path(raw)

    # Noisy MC run
    batches = []
    done    = 0
    while done < n_paths:
        bsz = min(mc_batch, n_paths - done)
        raw = simulate_batch(
            x0_np, Ds_init, drift_fn, noise_var, device, T, dt, rec,
            n_paths=bsz, seed=base_seed + done, store_traj=False
        )
        batches.append(raw)
        done += bsz

    return _reduce_mc_batches(batches, total_paths=n_paths)

# =============================================================================
#  Case 1 - GPU 0  (deterministic oscillating D(t), H=1)
# =============================================================================

def run_case1(gpu_id, H_heads, x0_S1,x0_S2):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    print(f'[GPU {gpu_id}] Case 1 H={H_heads}: starting...', flush=True)

    # D2_0 = torch.tensor([[[5., 0.], [0., 1.]]], dtype=torch.float64, device=device)
    # D3_0 = torch.diag(torch.tensor([3., 1., 2.5], dtype=torch.float64)).unsqueeze(0).to(device)

    def drift2(D, t):
        H = D.shape[0]
        c  = 1.5 * math.cos(t)
        s2 = math.sin(2 * t)

        if H == 1:
            # Original behaviour
            f = D.new_tensor([[[2. + c, s2], [s2, 2. - c]]])
            return f.expand_as(D) - D
        else:
            # One F per head, each parameterised by phase φ_h = 2π·h/H
            frames = []
            for h in range(H):
                phi = 2 * math.pi * h / H
                c_h  = 1.5 * math.cos(t + phi)
                s2_h = math.sin(2 * t + phi)
                frames.append([[2. + c_h, s2_h], [s2_h, 2. - c_h]])
            f = D.new_tensor(frames)   # (H, 2, 2)
            return f - D

    def drift3(D, t):
        H = D.shape[0]
        c     = 1.5 * math.cos(t)
        s2    = math.sin(2 * t)
        s     = math.sin(t)
        c2    = math.cos(2 * t)
        c_pi4 = 1.5 * math.cos(t + math.pi / 4)

        if H == 1:
            # Original behaviour
            f = D.new_tensor([[[2. + c,  s2, s ],
                               [s2, 2. + 1.5 * s, c2],
                               [s,  c2, 2. + c_pi4]]])
            return f.expand_as(D) - D
        else:
            # One F per head, parameterised by phase φ_h = 2π·h/H
            frames = []
            for h in range(H):
                phi    = 2 * math.pi * h / H
                c_h    = 1.5 * math.cos(t + phi)
                s2_h   = math.sin(2 * t + phi)
                s_h    = math.sin(t + phi)
                c2_h   = math.cos(2 * t + phi)
                cp4_h  = 1.5 * math.cos(t + phi + math.pi / 4)
                frames.append([[2. + c_h,        s2_h,              s_h ],
                               [s2_h,   2. + 1.5 * s_h,            c2_h],
                               [s_h,             c2_h,   2. + cp4_h    ]])
            f = D.new_tensor(frames)   # (H, 3, 3)
            return f - D

    def init_weights(H, d, seed):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        Z = torch.randn((H, d, d), dtype=torch.float64, device=device, generator=gen)
        return Z + Z.transpose(-2, -1)

    # res_s1 = simulate(
    #     x0_S1, D2_0, drift2, noise_var=0.0, device=device,
    #     T=T_MAX, dt=dt_sim, rec=rec_interval, store_traj=True
    # )
    # print(f'[GPU {gpu_id}] Case 1: S1 done', flush=True)

    D3_0 = init_weights(H_heads, d=3, seed=42)
    res_s2 = simulate(
        x0_S2, D3_0, drift3, noise_var=0.0, device=device,
        T=T_MAX, dt=dt_sim, rec=rec_interval, store_traj=True
    )
    print(f'[GPU {gpu_id}] Case 1: S2 done', flush=True)

    torch.save({'s2': res_s2, 'H': H_heads}, RESULTS_DIR + f'case1_H{H_heads}.pt')
    print(f'[GPU {gpu_id}] Case 1 H={H_heads}: saved', flush=True)

# =============================================================================
#  Case 2 - GPUs 1-3  (symmetric OU weights)
# =============================================================================

def run_case2(gpu_id, H_heads, x0_S1, x0_S2):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    print(f'[GPU {gpu_id}] Case 2 H={H_heads}: starting...', flush=True)

    noise_var = 2.

    def ou_drift(D, t):
        return torch.eye(D.shape[-1], device=D.device) - D

    def init_weights(H, d, seed):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        Z = torch.randn((H, d, d), dtype=torch.float64, device=device, generator=gen)
        return Z + Z.transpose(-2, -1)

    # Ds_S1 = init_weights(H_heads, d=2, seed=42 + H_heads)
    # res_s1 = simulate(
    #     x0_S1, Ds_S1, ou_drift, noise_var=noise_var, device=device,
    #     T=T_OU, dt=dt_ou, rec=rec_ou,
    #     seed=1000 + H_heads, n_paths=N_MC, mc_batch=MC_BATCH, store_traj=False
    # )
    # print(f'[GPU {gpu_id}] Case 2 H={H_heads}: S1 done', flush=True)

    Ds_S2 = init_weights(H_heads, d=3, seed=42)
    res_s2 = simulate(
        x0_S2, Ds_S2, ou_drift, noise_var=noise_var, device=device,
        T=T_OU, dt=dt_ou, rec=rec_ou,
        seed=2000 + H_heads, n_paths=N_MC, mc_batch=MC_BATCH, store_traj=False
    )
    print(f'[GPU {gpu_id}] Case 2 H={H_heads}: S2 done', flush=True)

    torch.save(
        {'s2': res_s2, 'H': H_heads, 'n_mc': N_MC, 'mc_batch': MC_BATCH},
        RESULTS_DIR + f'case2_H{H_heads}.pt'
    )
    print(f'[GPU {gpu_id}] Case 2 H={H_heads}: saved', flush=True)

# =============================================================================
#  Multi-processing entry point
# =============================================================================

def worker(args):
    kind, gpu_id, kwargs = args
    if kind == 'case1':
        run_case1(gpu_id, **kwargs)
    else:
        run_case2(gpu_id, **kwargs)

if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    print(f'Found {n_gpus} GPU(s)')
    assert n_gpus >= 4, f'Need >= 4 GPUs, found {n_gpus}.'

    x0_S1, x0_S2 = make_initial_tokens()

    jobs = [
        ('case1', 0, dict(H_heads=1, x0_S1=x0_S1, x0_S2=x0_S2)),
        ('case1', 1, dict(H_heads=100, x0_S1=x0_S1, x0_S2=x0_S2)),
        ('case2', 2, dict(H_heads=10, x0_S1=x0_S1, x0_S2=x0_S2)),
        ('case2', 3, dict(H_heads=100, x0_S1=x0_S1, x0_S2=x0_S2)),
        ('case1', 0, dict(H_heads=10, x0_S1=x0_S1, x0_S2=x0_S2)),
        ('case2', 1, dict(H_heads=1, x0_S1=x0_S1, x0_S2=x0_S2)),
    ]

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=4) as pool:
        pool.map(worker, jobs)

    print('\nAll simulations complete.')