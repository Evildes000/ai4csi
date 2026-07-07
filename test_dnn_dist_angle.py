"""
Test script for the SplitDistAngleEstimator model trained by train_dnn_dist_angle.py.

Loads a saved checkpoint and runs inference on the full dataset,
then visualizes distance & angle prediction performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import scienceplots
import time
from train_dnn_dist_angle import SplitDistAngleEstimator, pre_processing_dist_angle


def count_model_params(model: torch.nn.Module) -> dict:
    """
    Count parameters and estimate model size.

    Returns dict with keys:
        total_params, trainable_params, size_mb
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 * 1024)          # float32 = 4 bytes
    return {
        'total_params': total,
        'trainable_params': trainable,
        'size_mb': size_mb,
    }


def DNN_predict_dist_angle(processed_csi, checkpoint_path: str):
    """
    Load a trained SplitDistAngleEstimator and predict [distance, angle].

    Processes data in chunks to avoid OOM for large datasets (e.g. 100 MHz).

    Args:
        processed_csi:  feature vector, shape [N, features] (ndarray or memmap)
                        (antenna-0 real+imag + sin/cos of inter-antenna phase diffs)
        checkpoint_path: path to saved .pth checkpoint

    Returns:
        predictions:       shape [N, 2] = [distance, angle] in original physical units
        avg_inference_us:  average model forward time per sample (µs)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── Retrieve model config & scalers from checkpoint ──
    n_phase  = checkpoint['phase_dim']
    n_sincos = checkpoint['sincos_dim']
    scaler_dist   = checkpoint.get('scaler_dist')
    scaler_sincos = checkpoint.get('scaler_sincos')
    scaler_y      = checkpoint.get('scaler_y')

    best_str = f" (best val_loss: {checkpoint['best_val_loss']:.4f})" if 'best_val_loss' in checkpoint else ""
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs{best_str}")

    # ── Build model on CPU first, then try GPU ──
    model = SplitDistAngleEstimator(
        n_phase, n_sincos,
        checkpoint['hidden_dims_dist'],
        checkpoint['hidden_dims_angle']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    try:
        model = model.to(device)
    except RuntimeError:
        print("  GPU unavailable, falling back to CPU")
        device = torch.device("cpu")
        model = model.to(device)

    # ── Model size ──
    info = count_model_params(model)
    print(f"  Model size: {info['total_params']:,} params  ({info['size_mb']:.2f} MB)")

    # ── Chunked inference to avoid loading full dataset into RAM ──
    n_total = processed_csi.shape[0]
    # Chunk size: process ~5000 samples at a time; adjust based on available RAM
    chunk_size = min(5000, n_total)
    batch_size = 64
    all_preds = []
    total_model_time = 0.0  # cumulative model forward time (seconds)

    with torch.no_grad():
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk = processed_csi[start:end]  # memmap slicing is efficient, no full load

            # ── Apply saved normalization ──
            if scaler_dist is not None:
                X_phase = scaler_dist.transform(chunk[:, :n_phase]).astype(np.float32)
            else:
                X_phase = chunk[:, :n_phase].astype(np.float32)

            if scaler_sincos is not None:
                X_sincos = scaler_sincos.transform(chunk[:, n_phase:]).astype(np.float32)
            else:
                X_sincos = chunk[:, n_phase:].astype(np.float32)

            X_chunk = np.concatenate([X_phase, X_sincos], axis=1)
            del X_phase, X_sincos  # free memory

            # ── Batch inference on this chunk ──
            dataset = TensorDataset(torch.from_numpy(X_chunk))
            loader = DataLoader(dataset, batch_size=batch_size)
            del X_chunk

            for (xb,) in loader:
                xb = xb.to(device).float()
                x_phase_b  = xb[:, :n_phase]
                x_sincos_b = xb[:, n_phase:]
                t0 = time.perf_counter()
                pred = model(x_phase_b, x_sincos_b)
                total_model_time += time.perf_counter() - t0
                all_preds.append(pred.cpu().numpy())

    preds_norm = np.concatenate(all_preds)          # [N, 2] in normalized space

    # ── Inverse-transform to original units ──
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(preds_norm)
    else:
        predictions = preds_norm

    # Per-sample average inference time (µs)
    avg_inference_us = total_model_time / n_total * 1e6

    return predictions, avg_inference_us


def visualize_dist_angle(label_array: np.ndarray, predictions: np.ndarray, meta: dict):
    """
    Plot distance & angle prediction results.

    Args:
        label_array:  true [distance, angle], shape [N, 2]
        predictions:  predicted [distance, angle], shape [N, 2]
        meta:         metadata dict (must contain 'BW')
    """
    bw = meta['BW']
    labels_dist  = label_array[:, 0]
    labels_angle = label_array[:, 1]
    preds_dist   = predictions[:, 0]
    preds_angle  = predictions[:, 1]

    # ── Metrics ──
    err_dist  = preds_dist - labels_dist
    err_angle = preds_angle - labels_angle

    mae_dist   = float(np.mean(np.abs(err_dist)))
    rmse_dist  = float(np.sqrt(np.mean(err_dist ** 2)))
    mae_angle  = float(np.mean(np.abs(err_angle)))
    rmse_angle = float(np.sqrt(np.mean(err_angle ** 2)))

    print(f"\n{'='*60}")
    print(f"Test Results — {bw} MHz")
    print(f"  Distance — MAE: {mae_dist:.4f} m,  RMSE: {rmse_dist:.4f} m")
    print(f"  Angle    — MAE: {mae_angle:.2f}°,  RMSE: {rmse_angle:.2f}°")
    print(f"{'='*60}")

    # ── Per-label averages ──
    unique_distances = sorted(set(labels_dist))
    unique_angles    = sorted(set(labels_angle))

    dist_label_to_preds = defaultdict(list)
    angle_label_to_preds = defaultdict(list)
    for pd_, pa_, td, ta in zip(preds_dist, preds_angle, labels_dist, labels_angle):
        dist_label_to_preds[int(td)].append(float(pd_))
        angle_label_to_preds[int(ta)].append(float(pa_))

    dist_avg  = np.array([np.mean(dist_label_to_preds[int(d)]) for d in unique_distances])
    angle_avg = np.array([np.mean(angle_label_to_preds[int(a)]) for a in unique_angles])

    # ── Plotting ──

    # 'ieee' 会自动设置双栏宽度、Times New Roman 字体以及合理的默认字号
    plt.style.use(['science', 'ieee'])

    # 2. 针对 8pt 规范进行手动微调（SciencePlots 默认可能稍大，这里强制对齐 8pt）
    plt.rcParams.update({
        'font.family': 'serif',          # 使用衬线字体 (Times New Roman)
        'font.serif': ['Times New Roman'],
        'font.size': 7,                  # 【关键】全局基础字号设为 8pt
        'axes.labelsize': 7,             # 坐标轴标签大小
        'axes.titlesize': 7,             # 标题稍微大一点点，但也别超过 9pt
        'xtick.labelsize': 7,            # X轴刻度数字大小
        'ytick.labelsize': 7,            # Y轴刻度数字大小
        'legend.fontsize': 7,            # 图例文字大小
        'lines.linewidth': 1.2,          # 线宽适中，太粗会显得乱
        'figure.dpi': 300                # 导出清晰度
    })
    plt.rcParams['text.usetex'] = False

    # ------------distance------------------
    fig1, ax1 = plt.subplots(figsize=(6, 3))

    # (1) Distance: Predicted vs True
    # ax.scatter(labels_dist, preds_dist, alpha=0.3, s=8, label="Predictions")
    # ax.plot(unique_distances, dist_avg, 'g-', alpha=0.7, linewidth=2, label="Avg prediction")
    # d_min, d_max = min(unique_distances), max(unique_distances)
    # ax.plot([d_min, d_max], [d_min, d_max], 'r--', linewidth=1.5, label="Perfect")
    # ax.set_xlabel("True Distance (m)", fontsize=10)
    # ax.set_ylabel("Predicted Distance (m)", fontsize=10)
    # ax.set_title(f"Distance: Predicted vs True  (MAE={mae_dist:.2f} m)", fontsize=12)
    # ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.3)
    # ax.tick_params(labelsize=9)

    # ax1.scatter(labels_dist, preds_dist, alpha=0.3, s=8, label="Predictions")
    ax1.plot(unique_distances, dist_avg, 'g-', alpha=0.7, label="Avg prediction")
    d_min, d_max = min(unique_distances), max(unique_distances)
    ax1.plot([d_min, d_max], [d_min, d_max], 'r--', label="Perfect")
    ax1.set_xlabel("True Distance (m)")
    ax1.set_ylabel("Predicted Distance (m)")
    # axis='y': 仅修改Y轴
    # style='sci': 启用科学计数法
    # scilimits=(0,0): 强制对所有数量级都使用科学计数法（默认可能只对极大/极小值生效）
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.set_title(f"Distance: Predicted vs True  (MAE={mae_dist:.2f} m)", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    plt.savefig(f"dnn_figures/csi_dist_results_{bw}MHz_test.svg")
    # ax.tick_params(labelsize=9)




    # --------------------(2) Angle: Predicted vs True
    fig1, ax2 = plt.subplots(figsize=(6, 3))
    # ax2.scatter(labels_angle, preds_angle, alpha=0.3, s=8, label="Predictions")
    ax2.plot(unique_angles, angle_avg, 'g-', alpha=0.7, label="Avg prediction")
    a_min, a_max = min(unique_angles), max(unique_angles)
    ax2.plot([a_min, a_max], [a_min, a_max], 'r--', label="Perfect")
    ax2.set_xlabel("True Angle (deg)")
    ax2.set_ylabel("Predicted Angle (deg)")
    # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.set_title(f"Angle: Predicted vs True  (MAE={mae_angle:.1f}°)")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    # ax.tick_params(labelsize=9)
    plt.savefig(f"dnn_figures/csi_angle_results_{bw}MHz_test.svg")

    # ------------------------(3) Distance Error Histogram
    fig1, ax3 = plt.subplots(figsize=(6, 3))
    ax3.hist(err_dist, bins=40, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--')
    ax3.set_xlabel("Distance Error (m)")
    ax3.set_ylabel("Count")
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax3.set_title(f"Distance Error Distribution  (std={np.std(err_dist):.2f} m)")
    plt.savefig(f"dnn_figures/csi_dist_histogram{bw}MHz_test.svg")
    ax3.grid(True, alpha=0.3)
    # ax.tick_params(labelsize=9)

    # ------------------------(4) Angle Error Histogram
    fig1, ax4 = plt.subplots(figsize=(6, 3))
    ax4.hist(err_angle, bins=40, edgecolor='black', alpha=0.7, color='orange')
    ax4.axvline(0, color='r', linestyle='--')
    ax4.set_xlabel("Angle Error (deg)")
    ax4.set_ylabel("Count")
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax4.set_title(f"Angle Error Distribution  (std={np.std(err_angle):.1f}°)")
    plt.savefig(f"dnn_figures/csi_angle_histogram{bw}MHz_test.svg")
    ax4.grid(True, alpha=0.3)
    # ax.tick_params(labelsize=9)

    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    import os
    os.makedirs("dnn_figures", exist_ok=True)
    # plt.savefig(f"dnn_figures/csi_dist_angle_results_{bw}MHz_test.png")
    plt.savefig(f"dnn_figures/csi_dist_angle_results_{bw}MHz_test.svg")
    plt.show()


if __name__ == "__main__":
    bw_list = [20,40,80]

    for bw in bw_list:
        print(f"\n{'#'*60}")
        print(f"#  Testing — Bandwidth = {bw} MHz")
        print(f"{'#'*60}")

        path_to_data   = f"CSI_Angle_Dataset_{bw}MHz_4Rx_test.npz"
        path_to_meta   = f"metadata_CSI_Angle_Dataset_{bw}MHz_4Rx_test.json"
        path_checkpoint = f"dnn_checkpoint_dist_angle_{bw}MHz.pth"

        # ── Preprocessing (same as training) ──
        phase_data, label_array, meta = pre_processing_dist_angle(
            path_to_data=path_to_data,
            path_to_meta=path_to_meta
        )

        # ── Inference ──
        predictions, avg_inference_us = DNN_predict_dist_angle(phase_data, path_checkpoint)
        print(f"  Average inference time: {avg_inference_us:.1f} µs/sample")

        # ── Visualization ──
        visualize_dist_angle(label_array, predictions, meta)
