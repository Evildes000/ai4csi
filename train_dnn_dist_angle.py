"""
DNN for Joint Distance & Angle Estimation from CSI Unwrapped Phase

Based on train_dnn.py, extended to:
  - Input:  unwrapped CSI phase (not real+imag parts)
  - Output: [distance, angle]  (2-D regression)

Architecture (SPLIT dual-network):
  - Distance Net: per-antenna unwrapped phase → distance
    Phase slope across subcarriers encodes ToF → distance.
  - Angle Net:    sin/cos of inter-antenna phase diffs → angle
    Inter-antenna phase differences encode cos(θ) & sin(θ) via 2×2 array.
  - Two independent FNNs — no shared layers, each learns from its
    physically-relevant input subset.

NOTE: 2×2 rectangular array resolves the sin(θ) ambiguity inherent in ULAs
  by simultaneously capturing both cos(θ) and sin(θ).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import json
from collections import defaultdict


# ============================================================
# Weighted Loss — forces the model to pay attention to angle
# ============================================================
class WeightedMSELoss(nn.Module):
    """MSE with higher weight on angle to balance distance-vs-angle learning."""
    def __init__(self, angle_weight=50.0):
        super().__init__()
        self.w = angle_weight

    def forward(self, pred, target):
        # pred, target: [batch, 2] = [dist_norm, angle_norm]
        d = (pred[:, 0] - target[:, 0]) ** 2
        a = (pred[:, 1] - target[:, 1]) ** 2
        return d.mean() + self.w * a.mean()


# ============================================================
# Model Definition
# ============================================================
class SplitDistAngleEstimator(nn.Module):
    """
    Two separate FNNs decoupled by physical meaning:
      - Distance Net: antenna-0 CSI real+imag → distance (scalar)
                       Real+imag preserves full complex CSI (same as train_dnn.py).
      - Angle Net:     sin/cos of inter-antenna phase diffs → angle (scalar)
                       Inter-antenna phase differences encode AoA.

    The two sub-networks do NOT share any hidden layers — each learns
    from its physically-relevant input subset independently.
    """
    def __init__(self, phase_dim, sincos_dim,
                 hidden_dims_dist=(512, 256, 128, 64),
                 hidden_dims_angle=(512, 256, 128, 64), # 1024
                 dropout=0.2):
        super().__init__()

        def _build_mlp(in_dim, hidden_dims, out_dim):
            layers = []
            d = in_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(d, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                d = h
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        self.dist_net  = _build_mlp(phase_dim,  hidden_dims_dist,  1)   # → distance
        self.angle_net = _build_mlp(sincos_dim, hidden_dims_angle, 1)   # → angle

    def forward(self, x_phase, x_sincos):
        """
        Args:
            x_phase:   [batch, phase_dim]   — antenna-0 CSI real+imag
            x_sincos:  [batch, sincos_dim]  — sin/cos of inter-antenna phase diffs
        Returns:
            [batch, 2] = [distance, angle]
        """
        d = self.dist_net(x_phase)      # [batch, 1]
        a = self.angle_net(x_sincos)    # [batch, 1]
        return torch.cat([d, a], dim=1) # [batch, 2]


# ============================================================
# Training Helpers
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, n_phase):
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()                   # shape: [batch, 2]

        # Split concatenated input into phase & sincos portions
        x_phase  = xb[:, :n_phase]                   # [batch, n_phase]
        x_sincos = xb[:, n_phase:]                   # [batch, sincos_dim]

        optimizer.zero_grad()
        pred = model(x_phase, x_sincos)              # shape: [batch, 2]
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_phase, label_scaler=None):
    """
    Returns:
        avg_loss, mae_dist, rmse_dist, mae_angle, rmse_angle,
        all_preds_raw, all_labels_raw  (both [N, 2] in original scale)
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_preds_norm = []
    all_labels_norm = []

    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        # Split concatenated input into phase & sincos portions
        x_phase  = xb[:, :n_phase]
        x_sincos = xb[:, n_phase:]

        pred = model(x_phase, x_sincos)
        loss = criterion(pred, yb)
        total_loss += loss.item() * len(xb)
        n += len(xb)

        # transfer tensor to numpy and move it to cpu
        all_preds_norm.append(pred.cpu().numpy())
        all_labels_norm.append(yb.cpu().numpy())

    preds_norm = np.concatenate(all_preds_norm)     # [N, 2] normalized
    labels_norm = np.concatenate(all_labels_norm)   # [N, 2] normalized

    # Inverse-transform to original scales
    if label_scaler is not None:
        preds_raw = label_scaler.inverse_transform(preds_norm)
        labels_raw = label_scaler.inverse_transform(labels_norm)
    else:
        preds_raw = preds_norm
        labels_raw = labels_norm

    # Per-metric breakdown
    err_dist = preds_raw[:, 0] - labels_raw[:, 0]
    err_angle = preds_raw[:, 1] - labels_raw[:, 1]

    mae_dist  = np.mean(np.abs(err_dist))
    rmse_dist = np.sqrt(np.mean(err_dist ** 2))
    mae_angle  = np.mean(np.abs(err_angle))
    rmse_angle = np.sqrt(np.mean(err_angle ** 2))

    return (total_loss / n,
            mae_dist, rmse_dist, mae_angle, rmse_angle,
            preds_raw, labels_raw)


# ============================================================
# Main Training Routine
# ============================================================
def DNN_training_dist_angle(processed_csi: np.ndarray,
                             label_array: np.ndarray,
                             path_checkpoint: str,
                             meta: dict,
                             train_or_test: bool = True):
    """
    Args:
        processed_csi:  feature vector (phase + inter-ant diffs), shape [N, features]
        label_array:    labels, shape [N, 2] = [distance, angle]
        meta:           metadata dict from CSI generator
        path_checkpoint: checkpoint save/load path
        train_or_test:  True → train from scratch; False → load checkpoint
    """
    # ─────────────────────────────────────────────────────
    # Hyperparameters
    # ─────────────────────────────────────────────────────
    test_size    = 0.2
    val_size     = 0.1                # from training portion
    batch_size   = 128
    epochs       = 300
    lr           = 5e-4             # lower LR for stability (especially 100 MHz)
    hidden_dims_dist  = [512, 256, 128, 64]
    hidden_dims_angle = [512, 256, 128, 64]
    seed         = 42

    num_rx = meta['Nrx']
    num_subcarriers = meta['subcarriers']
    bw = meta['BW']

    num_samples = processed_csi.shape[0]

    # Features: [N, 2*Nfft + 2*(Nrx-1)*Nfft]  (antenna-0 real+imag + sincos diffs)
    print(f"Input shape: {processed_csi.shape}")
    print(f"Label shape: {label_array.shape}")

    # ─────────────────────────────────────────────────────
    # Train / Val / Test split
    # ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        processed_csi, label_array, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), random_state=seed
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ─────────────────────────────────────────────────────
    # Common setup
    # ─────────────────────────────────────────────────────

    # These variables are used to split the processed_csi for each samples
    n_phase   = 2 * num_subcarriers                       # antenna-0 real+imag (like train_dnn.py)
    n_sincos  = 2 * (num_rx - 1) * num_subcarriers         # sin+cos of 3 antenna-pair diffs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if train_or_test:
        # ═══════════════════════════════════════════════════════
        # Normalization (fit on training data)
        # ═══════════════════════════════════════════════════════
        # Real+imag features: per-feature StandardScaler (same as train_dnn.py)
        # Real/imag values have different scales per subcarrier; StandardScaler
        # normalizes each feature dimension independently.
        scaler_dist = StandardScaler()
        X_train_phase = scaler_dist.fit_transform(X_train[:, :n_phase]).astype(np.float32)
        X_val_phase   = scaler_dist.transform(X_val[:, :n_phase]).astype(np.float32)
        X_test_phase  = scaler_dist.transform(X_test[:, :n_phase]).astype(np.float32)

        # Sin/cos features: per-feature StandardScaler
        # These are already in [-1, 1]; StandardScaler is fine here.
        scaler_sincos = StandardScaler()
        # use X_train data as standardscaler
        X_train_sincos = scaler_sincos.fit_transform(X_train[:, n_phase:]).astype(np.float32)
        X_val_sincos   = scaler_sincos.transform(X_val[:, n_phase:]).astype(np.float32)
        X_test_sincos  = scaler_sincos.transform(X_test[:, n_phase:]).astype(np.float32)

        X_train = np.concatenate([X_train_phase, X_train_sincos], axis=1)
        X_val   = np.concatenate([X_val_phase,   X_val_sincos],   axis=1)
        X_test  = np.concatenate([X_test_phase,  X_test_sincos],  axis=1)

        # Label scaler
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train).astype(np.float32)
        y_val   = scaler_y.transform(y_val).astype(np.float32)
        y_test  = scaler_y.transform(y_test).astype(np.float32)

        # DataLoaders
        train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_set   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
        test_set  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size)
        test_loader  = DataLoader(test_set,  batch_size=batch_size)

        # Model, optimizer, scheduler
        model = SplitDistAngleEstimator(
            n_phase, n_sincos, hidden_dims_dist, hidden_dims_angle
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )

        print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"  Dist net:  {sum(p.numel() for p in model.dist_net.parameters()):,} params")
        print(f"  Angle net: {sum(p.numel() for p in model.angle_net.parameters()):,} params")
        print(model)

        # ═══════════════════════════════════════════════════════
        # Training Loop
        # ═══════════════════════════════════════════════════════
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae_dist": [],
            "val_rmse_dist": [],
            "val_mae_angle": [],
            "val_rmse_angle": [],
        }

        best_val_loss = float('inf')
        early_stop_patience = 300  # effectively no early stop — let model train fully
        early_stop_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, n_phase)
            val_loss, val_mae_d, val_rmse_d, val_mae_a, val_rmse_a, *_ = \
                evaluate(model, val_loader, criterion, device, n_phase, label_scaler=scaler_y)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae_dist"].append(val_mae_d)
            history["val_rmse_dist"].append(val_rmse_d)
            history["val_mae_angle"].append(val_mae_a)
            history["val_rmse_angle"].append(val_rmse_a)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'phase_dim': n_phase,
                    'sincos_dim': n_sincos,
                    'hidden_dims_dist': hidden_dims_dist,
                    'hidden_dims_angle': hidden_dims_angle,
                    'scaler_dist': scaler_dist,
                    'scaler_sincos': scaler_sincos,
                    'scaler_y': scaler_y,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'history': history,
                }
                torch.save(checkpoint, path_checkpoint)
            else:
                early_stop_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | train loss: {train_loss:.4f} | "
                      f"val loss: {val_loss:.4f} | "
                      f"dist MAE: {val_mae_d:.3f}m | angle MAE: {val_mae_a:.2f}°")

            if early_stop_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs)")
                break

    else:
        # ═══════════════════════════════════════════════════════
        # Load pre-trained model & apply its normalization
        # ═══════════════════════════════════════════════════════
        print(f"\nLoading pre-trained model from {path_checkpoint}")
        checkpoint = torch.load(path_checkpoint, map_location=device, weights_only=False)

        model = SplitDistAngleEstimator(
            checkpoint['phase_dim'], checkpoint['sincos_dim'],
            checkpoint['hidden_dims_dist'], checkpoint['hidden_dims_angle']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler_y = checkpoint.get('scaler_y')

        # Apply saved normalization to X_test
        scaler_dist = checkpoint.get('scaler_dist')
        scaler_sincos = checkpoint.get('scaler_sincos')

        if scaler_dist is not None:
            X_test_phase = scaler_dist.transform(X_test[:, :n_phase]).astype(np.float32)
        else:
            # backward compat: old checkpoints used phase_mean/phase_std
            phase_mean = checkpoint.get('phase_mean', 0.0)
            phase_std  = checkpoint.get('phase_std', 1.0)
            X_test_phase = ((X_test[:, :n_phase] - phase_mean) / phase_std).astype(np.float32)
        if scaler_sincos is not None:
            X_test_sincos = scaler_sincos.transform(X_test[:, n_phase:]).astype(np.float32)
        else:
            X_test_sincos = X_test[:, n_phase:].astype(np.float32)
        X_test = np.concatenate([X_test_phase, X_test_sincos], axis=1)

        # Standardize labels using saved scaler (model was trained on normalized y)
        if scaler_y is not None:
            y_test = scaler_y.transform(y_test).astype(np.float32)

        test_set  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(test_set, batch_size=batch_size)

        n_epochs = checkpoint.get('epoch', '?')
        best_val = checkpoint.get('best_val_loss')
        if best_val is not None:
            print(f"  Trained for {n_epochs} epochs (best val_loss: {best_val:.4f})")
        history = checkpoint.get('history')

    # criterion for final evaluation
    criterion = nn.MSELoss()

    # ─────────────────────────────────────────────────────
    # Final Evaluation on Test Set
    # ─────────────────────────────────────────────────────
    test_loss, test_mae_d, test_rmse_d, test_mae_a, test_rmse_a, \
        test_preds, test_labels = evaluate(
            model, test_loader, criterion, device, n_phase, label_scaler=scaler_y
        )

    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  MSE:        {test_loss:.4f}")
    print(f"  Distance — MAE: {test_mae_d:.4f} m, RMSE: {test_rmse_d:.4f} m")
    print(f"  Angle    — MAE: {test_mae_a:.2f}°,  RMSE: {test_rmse_a:.2f}°")
    print(f"{'='*60}")

    # ─────────────────────────────────────────────────────
    # Per-label statistics (average prediction per distance/angle)
    # ─────────────────────────────────────────────────────
    unique_distances = sorted(set(test_labels[:, 0]))
    unique_angles = sorted(set(test_labels[:, 1]))

    # Distance-wise averages
    dist_label_to_preds = defaultdict(list)
    for pred, label in zip(test_preds, test_labels):
        dist_label_to_preds[int(label[0])].append(pred[0])

    dist_avg = []
    for d_val in unique_distances:
        preds_for_d = dist_label_to_preds[int(d_val)]
        dist_avg.append(np.mean(preds_for_d) if preds_for_d else np.nan)
    dist_avg = np.array(dist_avg)

    # Angle-wise averages
    angle_label_to_preds = defaultdict(list)
    for pred, label in zip(test_preds, test_labels):
        angle_label_to_preds[int(label[1])].append(pred[1])

    angle_avg = []
    for a_val in unique_angles:
        preds_for_a = angle_label_to_preds[int(a_val)]
        angle_avg.append(np.mean(preds_for_a) if preds_for_a else np.nan)
    angle_avg = np.array(angle_avg)

    # ─────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) Training loss curves
    ax = axes[0, 0]
    if history is not None:
        ax.plot(history["train_loss"], label="Train MSE", alpha=0.7)
        ax.plot(history["val_loss"],   label="Val MSE",   alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Distance error curves
    ax = axes[0, 1]
    if history is not None:
        ax.plot(history["val_mae_dist"],  label="Dist MAE",  alpha=0.7)
        ax.plot(history["val_rmse_dist"], label="Dist RMSE", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (m)")
    ax.set_title("Distance Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) Angle error curves
    ax = axes[0, 2]
    if history is not None:
        ax.plot(history["val_mae_angle"],  label="Angle MAE",  alpha=0.7)
        ax.plot(history["val_rmse_angle"], label="Angle RMSE", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (deg)")
    ax.set_title("Angle Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) Distance: Predicted vs True
    ax = axes[1, 0]
    ax.scatter(test_labels[:, 0], test_preds[:, 0], alpha=0.3, s=8, label="Predictions")
    ax.plot(unique_distances, dist_avg, 'k-', alpha=0.7, linewidth=2, label="Avg prediction")
    d_min, d_max = min(unique_distances), max(unique_distances)
    ax.plot([d_min, d_max], [d_min, d_max], 'r--', linewidth=1.5, label="Perfect")
    ax.set_xlabel("True Distance (m)")
    ax.set_ylabel("Predicted Distance (m)")
    ax.set_title(f"Distance: Predicted vs True (MAE={test_mae_d:.2f}m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (5) Angle: Predicted vs True
    ax = axes[1, 1]
    ax.scatter(test_labels[:, 1], test_preds[:, 1], alpha=0.3, s=8, label="Predictions")
    ax.plot(unique_angles, angle_avg, 'k-', alpha=0.7, linewidth=2, label="Avg prediction")
    a_min, a_max = min(unique_angles), max(unique_angles)
    ax.plot([a_min, a_max], [a_min, a_max], 'r--', linewidth=1.5, label="Perfect")
    ax.set_xlabel("True Angle (deg)")
    ax.set_ylabel("Predicted Angle (deg)")
    ax.set_title(f"Angle: Predicted vs True (MAE={test_mae_a:.1f}°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (6) Error histograms
    ax = axes[1, 2]
    err_dist = test_preds[:, 0] - test_labels[:, 0]
    ax.hist(err_dist, bins=40, alpha=0.6, label=f'Dist err (σ={np.std(err_dist):.2f}m)',
            edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=1.5)
    # Second y-axis for angle errors
    ax2 = ax.twiny()
    err_angle = test_preds[:, 1] - test_labels[:, 1]
    ax2.hist(err_angle, bins=40, alpha=0.4, label=f'Angle err (σ={np.std(err_angle):.1f}°)',
             edgecolor='blue', color='orange')
    ax.set_xlabel("Distance Error (m)")
    ax2.set_xlabel("Angle Error (deg)")
    ax.set_title(f"Error Distributions")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out_dir = "dnn_figures"
    import os
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/csi_dist_angle_results_{bw}MHz.png", dpi=150)
    plt.savefig(f"{out_dir}/csi_dist_angle_results_{bw}MHz.svg", dpi=150)
    plt.show()


# ============================================================
# Data Preprocessing
# ============================================================
def pre_processing_dist_angle(path_to_data: str, path_to_meta: str):
    """
    Load CSI angle dataset and extract:
      - Antenna-0 real+imag CSI + inter-antenna phase differences as input
      - [distance, angle] as label (shape: [N, 2])

    Returns:
        phase_array:  feature vector, shape [N_total, 2*Nfft + 2*(Nrx-1)*Nfft]
                      (antenna-0 real+imag + sin/cos of inter-antenna phase diffs)
        label_array:  [distance, angle] for each sample, shape [N_total, 2]
        meta:         metadata dict
    """

    # laod the raw dataset
    data = np.load(path_to_data)

    with open(path_to_meta) as f:
        meta = json.load(f)

    num_rx = meta['Nrx']
    num_subcarriers = meta['subcarriers']
    num_packets = meta['num_packet']
    distance_list = meta['distance_list']
    angle_list = meta['angle_list']


    # load all name of files in the data
    name_data_arr = list(data.files)
    print(f"Number of (distance, angle) combinations: {len(name_data_arr)}")
    print(f"Distances: {distance_list}")
    print(f"Angles:    {angle_list}")

    # Pre-allocate memory-mapped array to avoid OOM for large datasets (e.g. 100 MHz)
    num_features = 2 * num_subcarriers + 2 * (num_rx - 1) * num_subcarriers
    total_samples = len(name_data_arr) * num_packets

    phase_path = path_to_data.replace('.npz', '_processed_phase.dat')
    print(f"Allocating memmap: {phase_path}  shape=({total_samples}, {num_features})")
    phase_array = np.memmap(phase_path, dtype='float32', mode='w+',
                            shape=(total_samples, num_features))
    label_array = np.zeros((total_samples, 2), dtype=np.float32)

    idx = 0
    for field_name in name_data_arr:
        # Parse field name: "dist_X_angle_Y"
        parts = field_name.split('_')
        d_val = int(parts[1])        # distance value
        a_val = int(parts[3])        # angle value

        csi_block = data[field_name]   # shape: [num_packets, Nrx, Nfft]


        # extract all csi samples out of all packets
        for pkt in range(num_packets):
            csi = np.array(csi_block[pkt, :, :])      # [Nrx, Nfft]

            # --- Extract unwrapped phase per antenna ---
            phase_raw = np.angle(csi)  # [Nrx, Nfft]
            phase_unwrapped = np.zeros((num_rx, num_subcarriers), dtype=np.float32)
            for rx in range(num_rx):
                phase_unwrapped[rx, :] = np.unwrap(phase_raw[rx, :])

            # --- Distance feature: real+imag of antenna at (0,0) ---
            # Same approach as train_dnn.py: real+imag preserves full complex CSI
            # information. The subcarrier-dependent phase rotation is encoded in
            # the joint real/imag pattern across subcarriers.
            # Antenna layout: 0(0,0), 1(d,0), 2(0,d), 3(d,d)
            csi_ant0 = csi[0, :]  # [Nfft] complex
            # csi_ant0 = phase_unwrapped[0,:]
            dist_features = np.concatenate([csi_ant0.real, csi_ant0.imag]).astype(np.float32)
            # shape: [2*Nfft]

            # --- Inter-antenna phase differences (2×2 array) ---
            #   Δφ_x = φ₁−φ₀ → encodes cos(θ), Δφ_y = φ₂−φ₀ → encodes sin(θ)
            # Use sin/cos of diffs to eliminate ±π wrapping ambiguity
            phase_diff = np.zeros((num_rx - 1, num_subcarriers), dtype=np.float32)  # [Nrx-1, Nfft]
            phase_diff[0, :] = phase_unwrapped[1, :] - phase_unwrapped[0, :]  # x-axis
            phase_diff[1, :] = phase_unwrapped[2, :] - phase_unwrapped[0, :]  # y-axis
            phase_diff[2, :] = phase_unwrapped[3, :] - phase_unwrapped[0, :]  # diagonal
            # sin/cos wrapping handles ±π → same value (cos(±π)=−1, sin(±π)=0)
            phase_diff_sincos = np.concatenate([np.sin(phase_diff), np.cos(phase_diff)], axis=0)
            # shape: [2*(Nrx-1), Nfft]

            # Concatenate antenna-0 real+imag + sin/cos of inter-antenna diffs
            features = np.concatenate([dist_features, phase_diff_sincos.flatten()])
            # shape: [2*Nfft + 2*(Nrx-1)*Nfft]

            phase_array[idx] = features
            label_array[idx] = [d_val, a_val]
            idx += 1

    phase_array.flush()  # ensure all data is written to disk



    print(f"\nPreprocessing complete:")
    print(f"  Phase array shape:  {phase_array.shape}")
    print(f"  Label array shape:  {label_array.shape}")
    print(f"  Label ranges — distance: [{label_array[:, 0].min()}, {label_array[:, 0].max()}] m")
    print(f"                angle:    [{label_array[:, 1].min()}, {label_array[:, 1].max()}]°")

    return phase_array, label_array, meta


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    # Use the angle-aware dataset (generated by csi_generator_with_angle.py)
    # For each bandwidth, train a joint distance+angle estimator

    bw_list = [20, 40, 80, 100]
    for bw in bw_list:
        print(f"\n{'#'*60}")
        print(f"#  Bandwidth = {bw} MHz")
        print(f"{'#'*60}")

        path_to_data = f"CSI_Angle_Dataset_{bw}MHz_4Rx.npz"
        path_to_meta = f"metadata_CSI_Angle_Dataset_{bw}MHz_4Rx.json"
        path_checkpoint = f"dnn_checkpoint_dist_angle_{bw}MHz.pth"

        phase_data, labels, metadata = pre_processing_dist_angle(
            path_to_data=path_to_data,
            path_to_meta=path_to_meta
        )

        DNN_training_dist_angle(
            processed_csi=phase_data,
            label_array=labels,
            path_checkpoint=path_checkpoint,
            meta=metadata,
            train_or_test=True
        )
