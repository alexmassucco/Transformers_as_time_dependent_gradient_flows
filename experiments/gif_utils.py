"""
GIF animation utilities for S¹ and S² token trajectories.
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
from PIL import Image

__all__ = ["save_s1_gif", "save_s2_gif"]

_U, _V = np.mgrid[0:2*np.pi:20j, 0:np.pi:12j]


def _frames_to_gif(frames, path, duration=80):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 loop=0, duration=duration)


def _fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def save_s1_gif(traj, path, color="#c0392b", title="", n_frames=60, duration=80):
    """
    traj : (m, n, 2)  or  (m, P, n, 2)  token positions on S¹
    """
    if traj is None:
        print(f"  [gif] traj is None, skipping {path}"); return
    if traj.ndim == 4: traj = traj[:, 0]       # take first MC path
    m = traj.shape[0]
    idx = np.round(np.linspace(0, m-1, n_frames)).astype(int)

    theta = np.linspace(0, 2*np.pi, 300)
    frames = []
    for i in idx:
        pts = traj[i]                           # (n, 2)
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax.plot(np.cos(theta), np.sin(theta), color="#ccc", lw=0.8)
        angles_all = np.linspace(0, 2*np.pi, pts.shape[0], endpoint=False)
        ax.scatter(pts[:,0], pts[:,1], c=angles_all, cmap="hsv",
                   s=22, edgecolors="white", linewidths=0.3, zorder=5)
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
        ax.set_aspect("equal"); ax.axis("off")
        t_frac = i/(m-1)
        ax.set_title(f"{title}   t={t_frac:.2f}T", fontsize=9)
        frames.append(_fig_to_array(fig))
        plt.close(fig)

    _frames_to_gif(frames, path, duration=duration)
    print(f"  [gif] S¹ GIF ({n_frames} frames) → {path}")


def save_s2_gif(traj, path, color="#27ae60", title="", n_frames=60, duration=80):
    """
    traj : (m, n, 3)  or  (m, P, n, 3)  token positions on S²
    """
    if traj is None:
        print(f"  [gif] traj is None, skipping {path}"); return
    if traj.ndim == 4: traj = traj[:, 0]
    m = traj.shape[0]
    idx = np.round(np.linspace(0, m-1, n_frames)).astype(int)

    # colour tokens by their initial angle (azimuth) for visual continuity
    az0 = np.arctan2(traj[0,:,1], traj[0,:,0])   # (n,)

    frames = []
    for i in idx:
        pts  = traj[i]
        fig  = plt.figure(figsize=(3.5,3.5))
        ax   = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(np.cos(_U)*np.sin(_V),
                          np.sin(_U)*np.sin(_V),
                          np.cos(_V),
                          color="#ddd", linewidth=0.2, alpha=0.4)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   c=az0, cmap="hsv", s=20,
                   depthshade=False, edgecolors="white", linewidths=0.3, zorder=5)
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2); ax.set_zlim(-1.2,1.2)
        ax.set_box_aspect([1,1,1]); ax.axis("off")
        ax.view_init(elev=18, azim= -55 + 180*i/max(m-1,1))   # slow rotation
        t_frac = i/(m-1)
        ax.set_title(f"{title}   t={t_frac:.2f}T", fontsize=8, pad=0)
        frames.append(_fig_to_array(fig))
        plt.close(fig)

    _frames_to_gif(frames, path, duration=duration)
    print(f"  [gif] S² GIF ({n_frames} frames) → {path}")
