"""
Distributed Multi-Hop Node Localization using DNN-based CSI Estimates

Algorithm (from prompt.txt):
  1. Each node gets an IP address and a flag (0 = unlocalized, 1 = localized).
  2. AP receives 500 packets from each node, extracts CSI, estimates
     (distance, angle, confidence) using the trained DNN.
  3. AP picks the node with the highest confidence, sends it the estimated
     (distance, angle).  That node computes its absolute position and
     sets its flag to 1.
  4. The newly localized node then processes CSI from packets sent by its
     neighbours, estimates (d, θ, conf), picks the unflagged node with the
     highest confidence, and sends its own absolute position + estimated
     (d, θ) so the neighbour can compute its position.
  5. Repeat until all nodes have flag = 1.

Position computation:
  - The CSI model estimates angle θ from TX → RX (CCW from +x axis).
  - When an already-localized node (RX) receives from an unlocalized
    node (TX) at distance d and angle θ:
        TX_pos = RX_pos - d * (cos θ,  sin θ)
  - (Because the direction from RX to TX is θ + 180°.)

Usage:
    python localization_dnn.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import time
import os
from collections import defaultdict
from scipy.optimize import minimize

from train_dnn_dist_angle_conf import DistAngleConfEstimator


# ================================================================
# 1. Model Loading (DNN only)
# ================================================================

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a trained DNN checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    best_str = ""
    if 'best_val_loss' in ckpt:
        best_str = f" (best val_loss: {ckpt['best_val_loss']:.4f})"
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs{best_str}")

    model = DistAngleConfEstimator(
        ckpt['phase_dim'], ckpt['sincos_dim'],
        ckpt['hidden_dims_dist'],
        ckpt['hidden_dims_angle'],
        ckpt['hidden_dims_conf'],
    ).to(device)

    state_dict = ckpt['model_state_dict']
    if ckpt.get('model_version', 1) < 2:
        print("  Migrating v1 checkpoint → v2 ...")
        for key_prefix in ['conf_net.0']:
            w_key = f'{key_prefix}.weight'
            if w_key in state_dict:
                old_w = state_dict[w_key]
                new_w = torch.cat([old_w, torch.zeros(old_w.shape[0], 1)], dim=1)
                state_dict[w_key] = new_w

    model.load_state_dict(state_dict)
    model.eval()

    n_phase  = ckpt['phase_dim']
    n_sincos = ckpt['sincos_dim']
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Type: DNN  |  {n_params:,} params  |  "
          f"phase_dim={n_phase}, sincos_dim={n_sincos}")

    return (model, n_phase, n_sincos,
            ckpt['scaler_dist'], ckpt['scaler_sincos'], ckpt['scaler_y'])


# ================================================================
# 2. CSI Featurization (same as training preprocessing)
# ================================================================

def featurize_csi(csi_block, n_phase, n_sincos):
    """
    Convert raw CSI [num_packet, Nrx, Nfft] → feature array [num_packet, n_features].
    """
    num_packet, num_rx, num_fft = csi_block.shape
    features = np.zeros((num_packet, n_phase + n_sincos), dtype=np.float32)

    for pkt_idx in range(num_packet):
        csi = np.array(csi_block[pkt_idx, :, :])

        phase_raw = np.angle(csi)
        phase_unw = np.zeros((num_rx, num_fft), dtype=np.float32)
        for rx in range(num_rx):
            phase_unw[rx, :] = np.unwrap(phase_raw[rx, :])

        csi_ant0 = csi[0, :]
        dist_feat = np.concatenate([csi_ant0.real, csi_ant0.imag]).astype(np.float32)

        phase_diff = np.zeros((num_rx - 1, num_fft), dtype=np.float32)
        phase_diff[0, :] = phase_unw[1, :] - phase_unw[0, :]
        phase_diff[1, :] = phase_unw[2, :] - phase_unw[0, :]
        phase_diff[2, :] = phase_unw[3, :] - phase_unw[0, :]
        sincos = np.concatenate([np.sin(phase_diff), np.cos(phase_diff)], axis=0)

        features[pkt_idx] = np.concatenate([dist_feat, sincos.flatten()])

    return features


# ================================================================
# 3. DNN Inference
# ================================================================

@torch.no_grad()
def estimate_link(features, model, n_phase, n_sincos,
                  scaler_dist, scaler_sincos, scaler_y, device):
    """
    Run inference on preprocessed features → per-packet (d, θ, conf).

    Returns:
        dists, angles, confs  — each ndarray [N]
    """
    batch_size = 256
    all_preds = []

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        batch = features[start:end]

        X_phase  = scaler_dist.transform(batch[:, :n_phase]).astype(np.float32)
        X_sincos = scaler_sincos.transform(batch[:, n_phase:]).astype(np.float32)

        xb = torch.from_numpy(np.concatenate([X_phase, X_sincos], axis=1)).to(device).float()

        pred = model(xb[:, :n_phase], xb[:, n_phase:])  # DNN split input

        all_preds.append(pred.cpu().numpy())

    preds_norm = np.concatenate(all_preds)
    preds = np.zeros_like(preds_norm)
    preds[:, :2] = scaler_y.inverse_transform(preds_norm[:, :2])
    preds[:, 2]  = preds_norm[:, 2]

    return preds[:, 0], preds[:, 1], preds[:, 2]


# ================================================================
# 4. Position Computation
# ================================================================

def compute_position(anchor_pos, distance, angle_deg):
    """
    Anchor transmits → target receives.  CSI gives (d, θ) where θ is the
    angle from anchor to target.  Compute target's absolute position.
    """
    theta = np.deg2rad(angle_deg)
    tx = anchor_pos[0] + distance * np.cos(theta)
    ty = anchor_pos[1] + distance * np.sin(theta)
    return np.array([tx, ty])


# ================================================================
# 5. Localization Algorithm
# ================================================================

def localize_nodes(scene_data, model, n_phase, n_sincos,
                   scaler_dist, scaler_sincos, scaler_y,
                   nodes_true, links_meta, device,
                   n_packets=500):
    """
    Run the distributed multi-hop localization algorithm.

    Returns:
        estimated_positions:  dict  node_name → np.array([x, y])
        localization_log:     list of (step, localized_node, anchor_node, conf)
    """
    node_names = list(nodes_true.keys())

    # ── State ──
    flags = {name: (name == 'AP') for name in node_names}          # AP pre-localized
    positions = {name: None for name in node_names}
    positions['AP'] = nodes_true['AP'].copy()                      # AP knows its own position
    ip = {name: f"192.168.1.{i+1}" for i, name in enumerate(node_names)}

    # ── Build link lookup: (tx, rx) → CSI field name ──
    link_csi = {}
    for field_name in scene_data.files:
        parts = field_name.split('_to_')
        if len(parts) == 2:
            tx, rx = parts
            link_csi[(tx, rx)] = field_name

    log = []

    print(f"\n{'='*70}")
    print(f"Distributed Localization — {len(node_names)} nodes  ({n_packets} pkts/link)")
    print(f"{'='*70}")
    print(f"\nInitial state: AP @ ({positions['AP'][0]:.1f}, {positions['AP'][1]:.1f}), "
          f"IP={ip['AP']}  [flag=1]\n")

    iteration = 0
    while not all(flags.values()):
        iteration += 1
        best_conf   = -1.0
        best_anchor = None
        best_target = None
        best_d_mean = None
        best_a_mean = None

        # ── For each localized node, scan its unflagged neighbours ──
        for anchor in node_names:
            if not flags[anchor]:
                continue

            for target in node_names:
                if flags[target]:
                    continue
                if anchor == target:
                    continue

                # CSI is from anchor (TX) → target (RX)
                csi_key = (anchor, target)
                if csi_key not in link_csi:
                    continue

                field = link_csi[csi_key]
                csi_block = scene_data[field][:n_packets, :, :]

                # Featurize + estimate
                feat = featurize_csi(csi_block, n_phase, n_sincos)
                d_est, a_est, c_est = estimate_link(
                    feat, model, n_phase, n_sincos,
                    scaler_dist, scaler_sincos, scaler_y, device,
                )

                c_mean = float(np.mean(c_est))
                d_mean = float(np.mean(d_est))
                a_mean = float(np.mean(a_est))

                if c_mean > best_conf:
                    best_conf   = c_mean
                    best_anchor = anchor
                    best_target = target
                    best_d_mean = d_mean
                    best_a_mean = a_mean

        if best_target is None:
            print("  No reachable unflagged node — stopping.")
            break

        # ── Localize the chosen target ──
        anchor_pos = positions[best_anchor]
        target_pos = compute_position(anchor_pos, best_d_mean, best_a_mean)
        positions[best_target] = target_pos
        flags[best_target] = True

        true_pos = nodes_true[best_target]
        err = np.linalg.norm(target_pos - true_pos)

        log.append((iteration, best_target, best_anchor, best_conf,
                    best_d_mean, best_a_mean, err))

        print(f"  Step {iteration}:  {best_anchor} ──→ {best_target}  "
              f"(conf={best_conf:.3f}, d={best_d_mean:.2f}m, θ={best_a_mean:.0f}°)")
        print(f"           Est pos: ({target_pos[0]:.2f}, {target_pos[1]:.2f})  "
              f"|  True: ({true_pos[0]:.2f}, {true_pos[1]:.2f})  "
              f"|  Err: {err:.2f} m  [flag=1]")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"Localization complete after {iteration} step(s).")
    print(f"{'='*70}")
    print(f"\n{'Node':<10} {'Flag':<6} {'IP':<16} {'Est (x,y)':<20} {'True (x,y)':<20} {'Err (m)':<10}")
    print("-" * 70)
    total_err = 0.0
    for name in node_names:
        flag = flags[name]
        est  = positions[name]
        true = nodes_true[name]
        if est is not None:
            err = np.linalg.norm(est - true)
            total_err += err
            print(f"{name:<10} {flag!s:<6} {ip[name]:<16} "
                  f"({est[0]:.1f}, {est[1]:.1f}){'':<10} "
                  f"({true[0]:.1f}, {true[1]:.1f}){'':<10} "
                  f"{err:.2f}")
        else:
            print(f"{name:<10} {flag!s:<6} {ip[name]:<16} "
                  f"{'—':<20} ({true[0]:.1f}, {true[1]:.1f}){'':<10} {'N/A'}")
    n_located = sum(flags.values())
    print(f"\n  Total nodes localized: {n_located}/{len(node_names)}")
    if n_located > 1:
        print(f"  Mean position error:   {total_err / (n_located - 1):.2f} m  "
              f"(excluding AP)")

    return positions, log


# ================================================================
# 6. Graph Optimization (refine positions to eliminate drift)
# ================================================================

def graph_optimize(init_positions, link_dists, link_confs,
                   link_angles=None, angle_weight=1.0,
                   anchor='AP', verbose=True):
    """
    Refine node positions by minimizing discrepancy between DNN-estimated
    (distance, angle) and those computed from positions.

    Cost = Σ w·(||pi-pj|| - d_est)² + w·wθ·(d_computed · angle_err)²
    where angle_err is the wrapped angular error in radians.
    d_computed scales so 1° error costs equally at any distance.
    """
    node_names = sorted(init_positions.keys())
    movable = [n for n in node_names if n != anchor]
    anchor_pos = init_positions[anchor]

    x0 = np.array([init_positions[n][0] for n in movable]
                  + [init_positions[n][1] for n in movable])

    index = {n: i for i, n in enumerate(movable)}
    n = len(movable)
    use_angles = link_angles is not None and len(link_angles) > 0

    pairs = []
    weights = []
    d_est = []
    a_est_rad = []
    for (a, b), d in link_dists.items():
        w = link_confs.get((a, b), 0.5) ** 2
        ia, ib = None, None
        if a == anchor:
            ia = -1; ib = index[b]
        elif b == anchor:
            ia = index[a]; ib = -1
        else:
            ia = index[a]; ib = index[b]
        if ia is None or ib is None:
            continue

        pairs.append((ia, ib))
        weights.append(w)
        d_est.append(d)
        if use_angles:
            ang = link_angles.get((a, b), None)
            a_est_rad.append(np.deg2rad(ang) if ang is not None else np.nan)

    pairs = np.array(pairs)
    weights = np.array(weights, dtype=np.float64)
    d_est = np.array(d_est, dtype=np.float64)
    if use_angles:
        a_est_rad = np.array(a_est_rad, dtype=np.float64)

    def cost(x):
        pts = np.zeros((n, 2), dtype=np.float64)
        pts[:, 0] = x[:n]
        pts[:, 1] = x[n:2*n]

        total = 0.0
        for k, (ia, ib) in enumerate(pairs):
            pa = anchor_pos if ia < 0 else pts[ia]
            pb = anchor_pos if ib < 0 else pts[ib]
            vec = pb - pa
            d_comp = np.linalg.norm(vec)
            err_d = d_comp - d_est[k]
            total += weights[k] * err_d * err_d

            if use_angles and not np.isnan(a_est_rad[k]):
                theta_c = np.arctan2(vec[1], vec[0])
                delta = theta_c - a_est_rad[k]
                angle_err = np.arctan2(np.sin(delta), np.cos(delta))
                lateral = d_comp * angle_err
                total += weights[k] * angle_weight * lateral * lateral

        return total

    if verbose:
        before = cost(x0)
        extra = " + angle" if use_angles else ""
        print(f"\n  Graph optimization: refining {n} positions "
              f"over {len(pairs)} links (distance{extra}) ...")
        print(f"  Initial cost: {before:.4f}")

    res = minimize(cost, x0, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-12})

    opt = np.zeros((n, 2), dtype=np.float64)
    opt[:, 0] = res.x[:n]
    opt[:, 1] = res.x[n:2*n]

    opt_positions = {anchor: anchor_pos.copy()}
    for i, name in enumerate(movable):
        opt_positions[name] = opt[i].copy()

    if verbose:
        after = cost(res.x)
        print(f"  Final cost:   {after:.4f}  "
              f"(reduced by {(1 - after/before)*100:.1f}%)\n")

    return opt_positions


# ================================================================
# 7. Visualization
# ================================================================

def visualize_localization(nodes_true, positions, log, walls, BW,
                           atten_coeff, fc, suffix=""):
    """Plot true vs estimated node positions with localization order."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Walls ──
    for w in walls:
        p1, p2 = np.array(w['p1']), np.array(w['p2'])
        t = w['thickness']
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        c1 = p1 + normal * t / 2
        c2 = p1 - normal * t / 2
        c3 = p2 - normal * t / 2
        c4 = p2 + normal * t / 2
        corners = np.array([c1, c2, c3, c4])
        alpha = min(0.3 + 0.5 * (t / 0.50), 1.0)
        ax.fill(corners[:, 0], corners[:, 1],
                facecolor='gray', edgecolor='black',
                linewidth=1.5, alpha=alpha)

    # ── Localization steps (arrows from anchor → target) ──
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(log), 1)))
    for i, (step, target, anchor, conf, d, a, err) in enumerate(log):
        p_anchor = positions[anchor]
        p_target = positions[target]
        ax.annotate("", xy=(p_target[0], p_target[1]),
                    xytext=(p_anchor[0], p_anchor[1]),
                    arrowprops=dict(arrowstyle="->", color=colors[i],
                                    lw=2.0, alpha=0.8))
        # Step number at midpoint
        mid = (p_anchor + p_target) / 2
        ax.text(mid[0], mid[1], str(step), fontsize=8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white',
                          edgecolor=colors[i], alpha=0.9))

    # ── True positions (hollow markers) ──
    for name, pos in nodes_true.items():
        ax.scatter(pos[0], pos[1], marker='o', s=120,
                   facecolors='none', edgecolors='gray',
                   linewidths=2, linestyle='--', zorder=4)

    # ── Estimated positions (filled markers) ──
    for name, pos in positions.items():
        if pos is not None:
            if name == 'AP':
                marker, size, color = 's', 180, '#d62728'
            else:
                marker, size, color = 'o', 120, '#1f77b4'
            ax.scatter(pos[0], pos[1], marker=marker, s=size,
                       c=color, edgecolors='black', linewidths=1.5, zorder=5)
            ax.annotate(name, xy=(pos[0], pos[1]),
                        xytext=(6, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold')

    # ── Connect true → estimated with dashed line ──
    for name in nodes_true:
        if positions.get(name) is not None and name != 'AP':
            t = nodes_true[name]
            e = positions[name]
            ax.plot([t[0], e[0]], [t[1], e[1]], 'r:', linewidth=1.0, alpha=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Distributed Node Localization ({BW} MHz)\n'
                 f'Solid = estimated,  Hollow = true,  '
                 f'Dotted red = error vector')
    ax.grid(True, alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color='gray', alpha=0.5, label='Wall'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
                   markersize=10, label='AP (known)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                   markersize=10, label='Node (estimated)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='gray', markeredgewidth=2, markersize=10,
                   label='Node (true)'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

    plt.tight_layout()
    os.makedirs('scene_figures', exist_ok=True)
    plt.savefig(f'scene_figures/localization_dnn_{BW}MHz{suffix}.png', dpi=200)
    plt.savefig(f'scene_figures/localization_dnn_{BW}MHz{suffix}.svg')
    plt.show()
    print(f"\nLocalization plot saved → scene_figures/localization_dnn_{BW}MHz{suffix}.png")


# ================================================================
# 8. Main
# ================================================================

def main():
    BW = 20
    scene_path = f"CSI_Scene_{BW}MHz_4Rx.npz"
    meta_path  = f"metadata_CSI_Scene_{BW}MHz_4Rx.json"
    ckpt_path  = f"dnn_checkpoint_dist_angle_conf_{BW}MHz.pth"

    for path, name in [(scene_path, "scene data"),
                       (meta_path, "metadata"),
                       (ckpt_path, "checkpoint")]:
        if not os.path.exists(path):
            print(f"{name} not found: {path}")
            return

    # ── Load metadata ──
    with open(meta_path) as f:
        meta = json.load(f)

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    (model, n_phase, n_sincos,
     scaler_dist, scaler_sincos, scaler_y) = load_checkpoint(ckpt_path, device)

    # ── Load scene data ──
    scene_data = np.load(scene_path)
    nodes_true = {k: np.array(v) for k, v in meta['nodes'].items()}
    links_meta = meta['links']
    walls      = meta.get('walls', [])
    atten_coeff = meta.get('atten_coeff_dBpm', 100.0)
    fc          = meta.get('fc', 5.8e9)

    # ── Run localization ──
    positions, log = localize_nodes(
        scene_data, model, n_phase, n_sincos,
        scaler_dist, scaler_sincos, scaler_y,
        nodes_true, links_meta, device,
        n_packets=500,
    )

    # ── Collect all link distance + confidence estimates for graph opt ──
    print(f"\n{'='*70}")
    print(f"Graph Optimization — using LOS-only links")
    print(f"{'='*70}")

    link_dists = {}
    link_confs = {}
    link_angles = {}
    for field_name in scene_data.files:
        parts = field_name.split('_to_')
        if len(parts) != 2:
            continue
        tx, rx = parts
        csi = scene_data[field_name][:500]
        feat = featurize_csi(csi, n_phase, n_sincos)
        d_est, a_est, c_est = estimate_link(
            feat, model, n_phase, n_sincos,
            scaler_dist, scaler_sincos, scaler_y, device,
        )
        link_dists[(tx, rx)] = float(np.mean(d_est))
        link_confs[(tx, rx)] = float(np.mean(c_est))
        link_angles[(tx, rx)] = float(np.mean(a_est))

    # ── Filter: keep only high-confidence (LOS) links ──
    conf_thresh = 0.8
    link_dists_los = {k: v for k, v in link_dists.items()
                      if link_confs[k] >= conf_thresh}
    link_confs_los  = {k: v for k, v in link_confs.items()
                       if k in link_dists_los}
    link_angles_los = {k: v for k, v in link_angles.items()
                       if k in link_dists_los}

    # ── Graph optimization ──
    opt_positions = graph_optimize(positions, link_dists_los, link_confs_los,
                                    link_angles=link_angles_los, anchor="AP")

    # ── Compare: sequential vs optimized ──
    node_names = sorted(nodes_true.keys())
    print(f"{'Node':<10} {'True (x,y)':<16} {'Sequential':<16} "
          f"{'Err(m)':<8} {'Optimized':<16} {'Err(m)':<8} {'Δ(m)':<8}")
    print("-" * 85)
    seq_errs = []
    opt_errs = []
    for name in node_names:
        true = nodes_true[name]
        seq_pos = positions[name]
        opt_pos = opt_positions[name]
        if seq_pos is not None:
            se = np.linalg.norm(seq_pos - true)
            oe = np.linalg.norm(opt_pos - true)
            seq_errs.append(se)
            opt_errs.append(oe)
            delta = se - oe
            print(f"{name:<10} ({true[0]:.1f}, {true[1]:.1f}){'':<4} "
                  f"({seq_pos[0]:.2f}, {seq_pos[1]:.2f}){'':<4} "
                  f"{se:<8.2f} "
                  f"({opt_pos[0]:.2f}, {opt_pos[1]:.2f}){'':<4} "
                  f"{oe:<8.2f} "
                  f"{delta:<+.2f}")
    mean_seq = np.mean([e for e in seq_errs if e > 0])
    mean_opt = np.mean([e for e in opt_errs if e > 0])
    print("-" * 85)
    print(f"{'OVERALL':<10} {'':16} {'':16} {mean_seq:<8.2f} {'':16} "
          f"{mean_opt:<8.2f} {mean_seq-mean_opt:<+.2f}")

    # ── Visualize: sequential (before optimization) ──
    visualize_localization(nodes_true, positions, log, walls, BW,
                           atten_coeff, fc, suffix='_sequential')

    # ── Visualize: optimized (after graph optimization) ──
    visualize_localization(nodes_true, opt_positions, log, walls, BW,
                           atten_coeff, fc, suffix='_optimized')


if __name__ == '__main__':
    main()
