"""
Direct AP-to-Node Localization (no multi-hop)

Each node's position is estimated independently using only the CSI
from AP to that node.  No relay, no propagation of errors across hops.

Usage:
    python localization_direct.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import time

from train_dnn_dist_angle_conf import DistAngleConfEstimator


# ================================================================
# Model loader
# ================================================================

def load_checkpoint(checkpoint_path, device):
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
# Featurization + Inference
# ================================================================

def featurize_csi(csi_block, n_phase, n_sincos):
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


@torch.no_grad()
def estimate_link(features, model, n_phase, n_sincos,
                  scaler_dist, scaler_sincos, scaler_y, device):
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
# Main
# ================================================================

def main():
    BW = 20
    scene_path = f"CSI_Scene_{BW}MHz_4Rx.npz"
    meta_path  = f"metadata_CSI_Scene_{BW}MHz_4Rx.json"
    ckpt_path  = f"dnn_checkpoint_dist_angle_conf_{BW}MHz.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, n_phase, n_sincos, scaler_dist, scaler_sincos, scaler_y = \
        load_checkpoint(ckpt_path, device)

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    scene_data = np.load(scene_path)
    nodes_true = {k: np.array(v) for k, v in meta['nodes'].items()}
    walls      = meta.get('walls', [])

    AP_pos = nodes_true['AP']
    n_packets = 500

    print(f"\n{'='*65}")
    print(f"Direct AP-to-Node Localization  ({n_packets} packets/link)")
    print(f"{'='*65}")
    print(f"\nAP position: ({AP_pos[0]:.1f}, {AP_pos[1]:.1f})\n")

    print(f"{'Node':<10} {'True (x,y)':<16} {'Est (x,y)':<16} "
          f"{'d_true':<8} {'d_est':<8} {'θ_true':<8} {'θ_est':<8} "
          f"{'Err(m)':<8} {'Conf':<8}")
    print("-" * 90)

    errors = {}
    for node_name in nodes_true:
        if node_name == 'AP':
            continue

        field = f'AP_to_{node_name}'
        if field not in scene_data.files:
            print(f"  {node_name:<8}  SKIP — no CSI field {field}")
            continue

        true_pos = nodes_true[node_name]
        true_dist = np.linalg.norm(true_pos - AP_pos)
        dx, dy = true_pos - AP_pos
        true_angle = np.rad2deg(np.arctan2(dy, dx)) % 360

        csi = scene_data[field][:n_packets]
        feat = featurize_csi(csi, n_phase, n_sincos)
        d_est, a_est, c_est = estimate_link(
            feat, model, n_phase, n_sincos,
            scaler_dist, scaler_sincos, scaler_y, device
        )

        d_mean = np.mean(d_est)
        a_mean = np.mean(a_est)
        c_mean = np.mean(c_est)

        est_pos = AP_pos + d_mean * np.array([np.cos(np.deg2rad(a_mean)),
                                               np.sin(np.deg2rad(a_mean))])
        err = np.linalg.norm(est_pos - true_pos)
        errors[node_name] = {'err': err, 'conf': c_mean,
                             'd_true': true_dist, 'd_est': d_mean,
                             'a_true': true_angle, 'a_est': a_mean}

        print(f"{node_name:<10} ({true_pos[0]:.1f}, {true_pos[1]:.1f}){'':<4} "
              f"({est_pos[0]:.2f}, {est_pos[1]:.2f}){'':<4} "
              f"{true_dist:<8.2f} {d_mean:<8.2f} "
              f"{true_angle:<8.0f} {a_mean:<8.0f} "
              f"{err:<8.2f} {c_mean:<8.3f}")

    mean_err = np.mean([v['err'] for v in errors.values()])
    print("-" * 90)
    print(f"{'OVERALL':<10} {'':16} {'':16} {'':8} {'':8} "
          f"{'':8} {'':8} {mean_err:<8.2f}")
    print(f"\n  Mean position error: {mean_err:.2f} m  "
          f"(all nodes via AP only, no multi-hop)")

    # ── Visualize ──
    fig, ax = plt.subplots(figsize=(9, 8))
    for w in walls:
        p1, p2 = np.array(w['p1']), np.array(w['p2'])
        t = w['thickness']
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        corners = np.array([
            p1 + normal * t / 2,
            p1 - normal * t / 2,
            p2 - normal * t / 2,
            p2 + normal * t / 2,
        ])
        alpha = min(0.3 + 0.5 * (t / 0.50), 1.0)
        ax.fill(corners[:, 0], corners[:, 1],
                facecolor='gray', edgecolor='black', linewidth=1.5, alpha=alpha)

    # Draw AP→Node arrows
    for node_name, info in errors.items():
        est_pos = AP_pos + info['d_est'] * np.array([
            np.cos(np.deg2rad(info['a_est'])),
            np.sin(np.deg2rad(info['a_est']))
        ])
        ax.annotate("", xy=(est_pos[0], est_pos[1]),
                    xytext=(AP_pos[0], AP_pos[1]),
                    arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=2, alpha=0.7))
        # Label at midpoint
        mid = (AP_pos + est_pos) / 2
        ax.text(mid[0], mid[1], f"{info['err']:.2f}m", fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

    # AP
    ax.scatter(AP_pos[0], AP_pos[1], marker='s', s=200, c='#d62728',
               edgecolors='black', linewidths=2, zorder=5)
    ax.annotate('AP', xy=AP_pos, xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=11, fontweight='bold')

    # Nodes: true (hollow) vs estimated (filled)
    for node_name, pos in nodes_true.items():
        if node_name == 'AP':
            continue
        # True
        ax.scatter(pos[0], pos[1], marker='o', s=100,
                   facecolors='none', edgecolors='gray', linewidths=2,
                   linestyle='--', zorder=4)
        # Estimated
        info = errors[node_name]
        est_pos = AP_pos + info['d_est'] * np.array([
            np.cos(np.deg2rad(info['a_est'])),
            np.sin(np.deg2rad(info['a_est']))
        ])
        ax.scatter(est_pos[0], est_pos[1], marker='o', s=80,
                   c='#1f77b4', edgecolors='black', linewidths=1.5, zorder=5)
        # Error line
        ax.plot([pos[0], est_pos[0]], [pos[1], est_pos[1]], 'r:', lw=1.0, alpha=0.6)
        ax.annotate(node_name, xy=est_pos, xytext=(6, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Direct AP → Node Localization ({BW} MHz)\n'
                 f'Mean error = {mean_err:.2f} m')
    ax.grid(True, alpha=0.3)
    legend_patches = [
        mpatches.Patch(color='gray', alpha=0.5, label='Wall'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
                   markersize=10, label='AP'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                   markersize=8, label='Est. position'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='gray', markeredgewidth=2, markersize=8,
                   label='True position'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    plt.tight_layout()
    os.makedirs('scene_figures', exist_ok=True)
    plt.savefig(f'scene_figures/localization_direct_{BW}MHz.png', dpi=200)
    plt.savefig(f'scene_figures/localization_direct_{BW}MHz.svg')
    plt.show()
    print(f"\nPlot saved → scene_figures/localization_direct_{BW}MHz.png")


if __name__ == '__main__':
    main()
