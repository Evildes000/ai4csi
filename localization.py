"""
Distributed Multi-Hop Node Localization using CSI-based DNN Estimates

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
    python localization.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import time
import os
from collections import defaultdict

from train_dnn_dist_angle_conf import DistAngleConfEstimator


# ================================================================
# 1. Model Loading
# ================================================================

def load_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
    return (model,
            ckpt['phase_dim'], ckpt['sincos_dim'],
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
        pred = model(xb[:, :n_phase], xb[:, n_phase:])
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
                   nodes_true, links_meta, device, n_packets=500):
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
                    scaler_dist, scaler_sincos, scaler_y, device
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
# 6. Visualization
# ================================================================

def visualize_localization(nodes_true, positions, log, walls, BW,
                           atten_coeff, fc):
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
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    os.makedirs('scene_figures', exist_ok=True)
    plt.savefig(f'scene_figures/localization_{BW}MHz.png', dpi=200)
    plt.savefig(f'scene_figures/localization_{BW}MHz.svg')
    plt.show()
    print(f"\nLocalization plot saved → scene_figures/localization_{BW}MHz.png")


# ================================================================
# 7. Main
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
    model, n_phase, n_sincos, scaler_dist, scaler_sincos, scaler_y = \
        load_checkpoint(ckpt_path, device)

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
        nodes_true, links_meta, device, n_packets=500
    )

    # ── Visualize ──
    visualize_localization(nodes_true, positions, log, walls, BW,
                           atten_coeff, fc)


if __name__ == '__main__':
    main()
