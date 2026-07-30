"""
Transformer for Joint Distance, Angle & Confidence Estimation from CSI

Replaces the split-tri-network DNN (train_dnn_dist_angle_conf.py) with a
single-tower Transformer architecture:

Architecture (SINGLE-TOWER with three task heads):
  - CSI features reshaped into per-subcarrier tokens
  - Learnable positional embedding on subcarrier index
  - CLS token appended before Transformer Encoder
  - Transformer Encoder (multi-head self-attention × N layers)
  - Three separate MLP heads on the CLS output:
      → distance, angle, confidence ∈ [0,1]

Same training pipeline, loss function, and data preprocessing as the DNN version.
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
import os
from collections import defaultdict


# ============================================================
# Loss: per-task MSE with configurable weights  (same as DNN)
# ============================================================
class MultiTaskMSELoss(nn.Module):
    """MSE over [distance, angle, confidence] with per-task weights."""
    def __init__(self, angle_weight=50.0, conf_weight=10.0):
        super().__init__()
        self.w_angle = angle_weight
        self.w_conf  = conf_weight

    def forward(self, pred, target):
        d = (pred[:, 0] - target[:, 0]) ** 2
        a = (pred[:, 1] - target[:, 1]) ** 2
        c = (pred[:, 2] - target[:, 2]) ** 2
        return d.mean() + self.w_angle * a.mean() + self.w_conf * c.mean()


# ============================================================
# Model Definition
# ============================================================
class TransformerDistAngleConfEstimator(nn.Module):
    """
    Single-tower Transformer with three regression heads.

    Input: flat feature vector [batch, n_phase + n_sincos]
           → reshaped to per-subcarrier tokens [batch, N_sc, d_per_sc]
           → projected to d_model → positional embedding → prepend CLS
           → Transformer Encoder → CLS output → 3 task MLP heads

    d_per_sc = 2 (ant-0 real+imag) + 2*(Nrx-1) (phase-diff sin/cos)
    """

    def __init__(self, num_subcarriers, num_rx,
                 d_model=64, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1,
                 hidden_dims_head=(128, 64)):
        super().__init__()

        d_per_sc = 2 + 2 * (num_rx - 1)

        # Input projection: per-subcarrier feature → d_model
        self.input_proj = nn.Linear(d_per_sc, d_model)

        # Learnable positional embedding for each subcarrier index
        self.pos_embedding = nn.Parameter(torch.randn(1, num_subcarriers, d_model) * 0.02)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Three task heads (small MLPs on CLS output)
        def _build_head(in_dim, hidden_dims, out_dim, final_activation=None):
            layers = []
            d = in_dim
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(d, h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                d = h
            layers.append(nn.Linear(d, out_dim))
            if final_activation is not None:
                layers.append(final_activation)
            return nn.Sequential(*layers)

        self.dist_head  = _build_head(d_model, hidden_dims_head, 1)
        self.angle_head = _build_head(d_model, hidden_dims_head, 1)
        self.conf_head  = _build_head(d_model, hidden_dims_head, 1,
                                      final_activation=nn.Sigmoid())

        # Store for checkpoint / inference
        self.num_subcarriers = num_subcarriers
        self.num_rx = num_rx
        self.d_per_sc = d_per_sc

        # Cache feature-dimension info used in reshaping
        self.n_phase  = 2 * num_subcarriers
        self.n_sincos = 2 * (num_rx - 1) * num_subcarriers

    def forward(self, x):
        """
        Args:
            x: [batch, total_features]  — flat, concatenated phase + sincos
        Returns:
            [batch, 3] = [distance, angle, confidence]
        """
        batch = x.shape[0]
        N_sc = self.num_subcarriers

        # ── Reshape flat input → per-subcarrier tokens ──
        # x_phase:  [batch, 2*N_sc]  arranged as  [real_0..real_{N-1}, imag_0..imag_{N-1}]
        # x_sincos: [batch, 6*N_sc]  arranged as  [sin0_0..sin0_{N-1}, sin1_0.., sin2_0..,
        #                                           cos0_0..cos0_{N-1}, cos1_0.., cos2_0..]
        x_phase  = x[:, :self.n_phase]       # [batch, 2*N_sc]
        x_sincos = x[:, self.n_phase:]        # [batch, K*N_sc],  K = 2*(Nrx-1)

        # Each group of N_sc values corresponds to one feature channel;
        # reshape to [batch, channels, N_sc] then transpose to [batch, N_sc, channels]
        x_phase  = x_phase.view(batch, 2, N_sc).transpose(1, 2)     # [batch, N_sc, 2]
        x_sincos = x_sincos.view(batch, 2 * (self.num_rx - 1), N_sc).transpose(1, 2)
        #                                                             # [batch, N_sc, K]

        tokens = torch.cat([x_phase, x_sincos], dim=-1)  # [batch, N_sc, d_per_sc]

        # ── Project to d_model & add positional embedding ──
        tokens = self.input_proj(tokens)                 # [batch, N_sc, d_model]
        tokens = tokens + self.pos_embedding

        # ── Prepend CLS token ──
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch, 1+N_sc, d_model]

        # ── Transformer Encoder ──
        encoded = self.transformer(tokens)               # [batch, 1+N_sc, d_model]

        # ── CLS token output → three heads ──
        cls_out = encoded[:, 0, :]                       # [batch, d_model]

        dist   = self.dist_head(cls_out)
        angle  = self.angle_head(cls_out)
        conf   = self.conf_head(cls_out)

        return torch.cat([dist, angle, conf], dim=1)     # [batch, 3]


# ============================================================
# Training Helpers
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        optimizer.zero_grad()
        pred = model(xb)              # [batch, 3]
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_scaler=None):
    """
    Returns:
        avg_loss, mae_dist, rmse_dist, mae_angle, rmse_angle,
        mae_conf, rmse_conf, all_preds_raw, all_labels_raw
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_preds_norm = []
    all_labels_norm = []

    for xb, yb in loader:
        xb = xb.to(device).float()
        yb = yb.to(device).float()

        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item() * len(xb)
        n += len(xb)

        all_preds_norm.append(pred.cpu().numpy())
        all_labels_norm.append(yb.cpu().numpy())

    preds_norm = np.concatenate(all_preds_norm)
    labels_norm = np.concatenate(all_labels_norm)

    if label_scaler is not None:
        preds_raw = np.zeros_like(preds_norm)
        labels_raw = np.zeros_like(labels_norm)
        preds_raw[:, :2] = label_scaler.inverse_transform(preds_norm[:, :2])
        labels_raw[:, :2] = label_scaler.inverse_transform(labels_norm[:, :2])
        preds_raw[:, 2] = preds_norm[:, 2]
        labels_raw[:, 2] = labels_norm[:, 2]
    else:
        preds_raw = preds_norm
        labels_raw = labels_norm

    err_dist  = preds_raw[:, 0] - labels_raw[:, 0]
    err_angle = preds_raw[:, 1] - labels_raw[:, 1]
    err_conf  = preds_raw[:, 2] - labels_raw[:, 2]

    mae_dist   = np.mean(np.abs(err_dist))
    rmse_dist  = np.sqrt(np.mean(err_dist ** 2))
    mae_angle  = np.mean(np.abs(err_angle))
    rmse_angle = np.sqrt(np.mean(err_angle ** 2))
    mae_conf   = np.mean(np.abs(err_conf))
    rmse_conf  = np.sqrt(np.mean(err_conf ** 2))

    return (total_loss / n,
            mae_dist, rmse_dist, mae_angle, rmse_angle, mae_conf, rmse_conf,
            preds_raw, labels_raw)


# ============================================================
# Main Training Routine
# ============================================================
def Transformer_training_dist_angle_conf(processed_csi, label_array,
                                         path_checkpoint, meta):
    """
    Args:
        processed_csi:  feature vector, shape [N, n_phase + n_sincos]
        label_array:    labels, shape [N, 3] = [distance, angle, confidence]
        meta:           metadata dict
        path_checkpoint: checkpoint save path
    """
    # ─────────────────────────────────────────────────────
    # Hyperparameters
    # ─────────────────────────────────────────────────────
    test_size    = 0.2
    val_size     = 0.1
    batch_size   = 128
    epochs       = 100
    lr           = 3e-4
    angle_weight = 50.0
    conf_weight  = 10.0
    seed         = 42

    # ---- Transformer hyperparameters ----
    d_model         = 64
    nhead           = 4
    num_layers      = 4
    dim_feedforward = 256
    dropout         = 0.1
    hidden_dims_head = [128, 64]

    num_rx = meta['Nrx']
    num_subcarriers = meta['subcarriers']
    bw = meta['BW']

    print(f"Input shape: {processed_csi.shape}")
    print(f"Label shape: {label_array.shape}")
    print(f"Per-subcarrier token dim: {2 + 2 * (num_rx - 1)} "
          f"(2 ant0 real+imag + {2 * (num_rx - 1)} phase-diff sin/cos)")
    print(f"Sequence length (subcarriers): {num_subcarriers}")

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

    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        los_frac = np.mean(y[:, 2] > 0.5)
        print(f"  {name} LOS fraction: {los_frac:.2%}")

    # ─────────────────────────────────────────────────────
    # Common setup
    # ─────────────────────────────────────────────────────
    n_phase   = 2 * num_subcarriers
    n_sincos  = 2 * (num_rx - 1) * num_subcarriers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ═══════════════════════════════════════════════════════
    # Normalization (fit on training data)
    # ═══════════════════════════════════════════════════════
    scaler_dist = StandardScaler()
    X_train_phase = scaler_dist.fit_transform(X_train[:, :n_phase]).astype(np.float32)
    X_val_phase   = scaler_dist.transform(X_val[:, :n_phase]).astype(np.float32)
    X_test_phase  = scaler_dist.transform(X_test[:, :n_phase]).astype(np.float32)

    scaler_sincos = StandardScaler()
    X_train_sincos = scaler_sincos.fit_transform(X_train[:, n_phase:]).astype(np.float32)
    X_val_sincos   = scaler_sincos.transform(X_val[:, n_phase:]).astype(np.float32)
    X_test_sincos  = scaler_sincos.transform(X_test[:, n_phase:]).astype(np.float32)

    X_train_full = np.concatenate([X_train_phase, X_train_sincos], axis=1)
    X_val_full   = np.concatenate([X_val_phase,   X_val_sincos],   axis=1)
    X_test_full  = np.concatenate([X_test_phase,  X_test_sincos],  axis=1)

    # Label scaler: only normalize distance & angle
    scaler_y = StandardScaler()
    y_train_scaled = np.zeros_like(y_train, dtype=np.float32)
    y_val_scaled   = np.zeros_like(y_val,   dtype=np.float32)
    y_test_scaled  = np.zeros_like(y_test,  dtype=np.float32)

    y_train_scaled[:, :2] = scaler_y.fit_transform(y_train[:, :2]).astype(np.float32)
    y_val_scaled[:, :2]   = scaler_y.transform(y_val[:, :2]).astype(np.float32)
    y_test_scaled[:, :2]  = scaler_y.transform(y_test[:, :2]).astype(np.float32)
    y_train_scaled[:, 2]  = y_train[:, 2].astype(np.float32)
    y_val_scaled[:, 2]    = y_val[:, 2].astype(np.float32)
    y_test_scaled[:, 2]   = y_test[:, 2].astype(np.float32)

    y_train = y_train_scaled
    y_val   = y_val_scaled
    y_test  = y_test_scaled

    # DataLoaders
    train_set = TensorDataset(torch.from_numpy(X_train_full), torch.from_numpy(y_train))
    val_set   = TensorDataset(torch.from_numpy(X_val_full),   torch.from_numpy(y_val))
    test_set  = TensorDataset(torch.from_numpy(X_test_full),  torch.from_numpy(y_test))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    # ═══════════════════════════════════════════════════════
    # Model, optimizer, scheduler
    # ═══════════════════════════════════════════════════════
    model = TransformerDistAngleConfEstimator(
        num_subcarriers=num_subcarriers,
        num_rx=num_rx,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        hidden_dims_head=hidden_dims_head,
    ).to(device)

    criterion = MultiTaskMSELoss(angle_weight=angle_weight, conf_weight=conf_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"  d_model={d_model}, nhead={nhead}, layers={num_layers}, "
          f"ff_dim={dim_feedforward}")
    print(f"  Token dim per subcarrier: {model.d_per_sc}")
    print(f"  Sequence length:          {num_subcarriers} + 1 (CLS)")

    # ═══════════════════════════════════════════════════════
    # Training Loop
    # ═══════════════════════════════════════════════════════
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae_dist": [],   "val_rmse_dist": [],
        "val_mae_angle": [],  "val_rmse_angle": [],
        "val_mae_conf": [],   "val_rmse_conf": [],
    }

    best_val_loss = float('inf')
    early_stop_patience = 300
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, v_mae_d, v_rmse_d, v_mae_a, v_rmse_a, v_mae_c, v_rmse_c, *_ = \
            evaluate(model, val_loader, criterion, device, label_scaler=scaler_y)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_dist"].append(v_mae_d)
        history["val_rmse_dist"].append(v_rmse_d)
        history["val_mae_angle"].append(v_mae_a)
        history["val_rmse_angle"].append(v_rmse_a)
        history["val_mae_conf"].append(v_mae_c)
        history["val_rmse_conf"].append(v_rmse_c)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'num_subcarriers': num_subcarriers,
                    'num_rx': num_rx,
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                    'hidden_dims_head': hidden_dims_head,
                },
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
                  f"dist MAE: {v_mae_d:.3f}m | angle MAE: {v_mae_a:.2f}° | "
                  f"conf MAE: {v_mae_c:.4f}")

        if early_stop_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    criterion = MultiTaskMSELoss(angle_weight=angle_weight, conf_weight=conf_weight)

    # ─────────────────────────────────────────────────────
    # Final Evaluation on Test Set
    # ─────────────────────────────────────────────────────
    test_loss, test_mae_d, test_rmse_d, test_mae_a, test_rmse_a, test_mae_c, test_rmse_c, \
        test_preds, test_labels = evaluate(
            model, test_loader, criterion, device, label_scaler=scaler_y
        )

    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  MSE:         {test_loss:.4f}")
    print(f"  Distance  — MAE: {test_mae_d:.4f} m,  RMSE: {test_rmse_d:.4f} m")
    print(f"  Angle     — MAE: {test_mae_a:.2f}°,   RMSE: {test_rmse_a:.2f}°")
    print(f"  Confidence — MAE: {test_mae_c:.4f},  RMSE: {test_rmse_c:.4f}")
    print(f"{'='*60}")

    # ─────────────────────────────────────────────────────
    # Breakdown: LOS vs NLOS test performance
    # ─────────────────────────────────────────────────────
    los_mask = test_labels[:, 2] > 0.5
    nlos_mask = ~los_mask
    if los_mask.sum() > 0 and nlos_mask.sum() > 0:
        print(f"\n{'─'*60}")
        print(f"Breakdown by scenario:")
        for mask, label in [(los_mask, "LOS (no wall)"), (nlos_mask, "NLOS (wall)")]:
            err_d = test_preds[mask, 0] - test_labels[mask, 0]
            err_a = test_preds[mask, 1] - test_labels[mask, 1]
            pred_c = test_preds[mask, 2]
            print(f"  {label} ({mask.sum()} samples):")
            print(f"    Dist MAE: {np.mean(np.abs(err_d)):.4f}m, "
                  f"Angle MAE: {np.mean(np.abs(err_a)):.2f}°, "
                  f"Mean confidence: {np.mean(pred_c):.4f}")

    # ─────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────
    unique_distances = sorted(set(test_labels[:, 0]))
    unique_angles = sorted(set(test_labels[:, 1]))

    dist_label_to_preds = defaultdict(list)
    for pred, label in zip(test_preds, test_labels):
        dist_label_to_preds[int(label[0])].append(pred[0])
    dist_avg = np.array([np.mean(dist_label_to_preds[int(d)]) for d in unique_distances])

    angle_label_to_preds = defaultdict(list)
    for pred, label in zip(test_preds, test_labels):
        angle_label_to_preds[int(label[1])].append(pred[1])
    angle_avg = np.array([np.mean(angle_label_to_preds[int(a)]) for a in unique_angles])

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # (1) Training loss curves
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", alpha=0.7)
    ax.plot(history["val_loss"],   label="Val",   alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # (2) Distance error curves
    ax = axes[0, 1]
    ax.plot(history["val_mae_dist"],  label="Dist MAE",  alpha=0.7)
    ax.plot(history["val_rmse_dist"], label="Dist RMSE", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (m)")
    ax.set_title("Distance Prediction Error"); ax.legend(); ax.grid(True, alpha=0.3)

    # (3) Angle error curves
    ax = axes[0, 2]
    ax.plot(history["val_mae_angle"],  label="Angle MAE",  alpha=0.7)
    ax.plot(history["val_rmse_angle"], label="Angle RMSE", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (deg)")
    ax.set_title("Angle Prediction Error"); ax.legend(); ax.grid(True, alpha=0.3)

    # (4) Confidence error curves
    ax = axes[0, 3]
    ax.plot(history["val_mae_conf"],  label="Conf MAE",  alpha=0.7)
    ax.plot(history["val_rmse_conf"], label="Conf RMSE", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error")
    ax.set_title("Confidence Prediction Error"); ax.legend(); ax.grid(True, alpha=0.3)

    # (5) Distance: Predicted vs True (color by confidence)
    ax = axes[1, 0]
    sc = ax.scatter(test_labels[:, 0], test_preds[:, 0],
                    c=test_preds[:, 2], cmap='RdYlGn', alpha=0.3, s=8, vmin=0, vmax=1)
    ax.plot(unique_distances, dist_avg, 'k-', alpha=0.7, linewidth=2, label="Avg")
    d_min, d_max = min(unique_distances), max(unique_distances)
    ax.plot([d_min, d_max], [d_min, d_max], 'r--', linewidth=1.5, label="Perfect")
    plt.colorbar(sc, ax=ax, label="Predicted Confidence")
    ax.set_xlabel("True Distance (m)"); ax.set_ylabel("Predicted Distance (m)")
    ax.set_title(f"Distance: Predicted vs True (MAE={test_mae_d:.2f}m)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (6) Angle: Predicted vs True (color by confidence)
    ax = axes[1, 1]
    sc = ax.scatter(test_labels[:, 1], test_preds[:, 1],
                    c=test_preds[:, 2], cmap='RdYlGn', alpha=0.3, s=8, vmin=0, vmax=1)
    ax.plot(unique_angles, angle_avg, 'k-', alpha=0.7, linewidth=2, label="Avg")
    a_min, a_max = min(unique_angles), max(unique_angles)
    ax.plot([a_min, a_max], [a_min, a_max], 'r--', linewidth=1.5, label="Perfect")
    plt.colorbar(sc, ax=ax, label="Predicted Confidence")
    ax.set_xlabel("True Angle (deg)"); ax.set_ylabel("Predicted Angle (deg)")
    ax.set_title(f"Angle: Predicted vs True (MAE={test_mae_a:.1f}°)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (7) Confidence histogram (LOS vs NLOS)
    ax = axes[1, 2]
    pred_conf_los  = test_preds[los_mask, 2] if los_mask.sum() > 0 else []
    pred_conf_nlos = test_preds[nlos_mask, 2] if nlos_mask.sum() > 0 else []
    ax.hist(pred_conf_los,  bins=30, alpha=0.6, label=f'True LOS (n={len(pred_conf_los)})',
            edgecolor='black', color='green')
    ax.hist(pred_conf_nlos, bins=30, alpha=0.5, label=f'True NLOS (n={len(pred_conf_nlos)})',
            edgecolor='black', color='red')
    ax.set_xlabel("Predicted Confidence"); ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution (MAE={test_mae_c:.4f})")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (8) Distance error vs confidence
    ax = axes[1, 3]
    err_dist_abs = np.abs(test_preds[:, 0] - test_labels[:, 0])
    ax.scatter(test_preds[:, 2], err_dist_abs, alpha=0.3, s=8)
    ax.set_xlabel("Predicted Confidence"); ax.set_ylabel("Absolute Distance Error (m)")
    ax.set_title("Distance Error vs Confidence")
    ax.grid(True, alpha=0.3)
    if len(err_dist_abs) > 1:
        coeff = np.polyfit(test_preds[:, 2], err_dist_abs, 1)
        x_fit = np.linspace(0, 1, 100)
        ax.plot(x_fit, np.polyval(coeff, x_fit), 'r-', linewidth=2,
                label=f'Slope: {coeff[0]:.2f}')
        ax.legend()

    plt.tight_layout()
    out_dir = "transformer_figures"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/csi_dist_angle_conf_results_{bw}MHz.png", dpi=150)
    plt.savefig(f"{out_dir}/csi_dist_angle_conf_results_{bw}MHz.svg", dpi=150)
    plt.show()


# ============================================================
# Data Preprocessing  (identical to DNN version)
# ============================================================
def pre_processing_dist_angle_conf(path_to_data, path_to_meta, confidence_label):
    """
    Load one CSI dataset and assign confidence labels.

    Args:
        path_to_data:      path to .npz file
        path_to_meta:      path to metadata .json file
        confidence_label:  1.0 for LOS (no-wall), 0.0 for NLOS (wall-obstructed)

    Returns:
        phase_array:  feature vector, shape [N, n_features]
        label_array:  [distance, angle, confidence], shape [N, 3]
        meta:         metadata dict
    """
    data = np.load(path_to_data)

    with open(path_to_meta) as f:
        meta = json.load(f)

    num_rx = meta['Nrx']
    num_subcarriers = meta['subcarriers']
    num_packets = meta['num_packet']
    distance_list = meta['distance_list']
    angle_list = meta['angle_list']

    name_data_arr = list(data.files)
    print(f"  Dataset: {path_to_data}")
    print(f"  Combinations: {len(name_data_arr)}, "
          f"Distances: {distance_list[0]}–{distance_list[-1]}m, "
          f"Angles: {angle_list[0]}–{angle_list[-1]}°")

    num_features = 2 * num_subcarriers + 2 * (num_rx - 1) * num_subcarriers
    total_samples = len(name_data_arr) * num_packets

    memmap_path = path_to_data.replace('.npz', f'_processed_conf{int(confidence_label)}.dat')
    phase_array = np.memmap(memmap_path, dtype='float32', mode='w+',
                            shape=(total_samples, num_features))
    label_array = np.zeros((total_samples, 3), dtype=np.float32)

    idx = 0
    for field_name in name_data_arr:
        parts = field_name.split('_')
        d_val = int(parts[1])
        a_val = int(parts[3])

        csi_block = data[field_name]   # [num_packets, Nrx, Nfft]

        for pkt in range(num_packets):
            csi = np.array(csi_block[pkt, :, :])      # [Nrx, Nfft]

            # --- Unwrapped phase per antenna ---
            phase_raw = np.angle(csi)
            phase_unwrapped = np.zeros((num_rx, num_subcarriers), dtype=np.float32)
            for rx in range(num_rx):
                phase_unwrapped[rx, :] = np.unwrap(phase_raw[rx, :])

            # --- Distance feature: antenna-0 real+imag ---
            csi_ant0 = csi[0, :]
            dist_features = np.concatenate([csi_ant0.real, csi_ant0.imag]).astype(np.float32)

            # --- Inter-antenna phase differences (sin/cos) ---
            phase_diff = np.zeros((num_rx - 1, num_subcarriers), dtype=np.float32)
            phase_diff[0, :] = phase_unwrapped[1, :] - phase_unwrapped[0, :]   # x-axis
            phase_diff[1, :] = phase_unwrapped[2, :] - phase_unwrapped[0, :]   # y-axis
            phase_diff[2, :] = phase_unwrapped[3, :] - phase_unwrapped[0, :]   # diagonal
            phase_diff_sincos = np.concatenate([np.sin(phase_diff), np.cos(phase_diff)], axis=0)

            features = np.concatenate([dist_features, phase_diff_sincos.flatten()])

            phase_array[idx] = features
            label_array[idx] = [d_val, a_val, confidence_label]
            idx += 1

    phase_array.flush()
    print(f"  Samples: {total_samples}, Features: {num_features}")
    return phase_array, label_array, meta


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    bw_list = [20]

    # Confidence labels matching csi_generator_with_angle.py 10-level output:
    #   Level 0:  0 dB wall loss → conf 1.0, ..., Level 9: 90 dB → conf 0.1
    NUM_LEVELS = 10
    conf_labels = [round(1.0 - lvl * 0.1, 1) for lvl in range(NUM_LEVELS)]  # [1.0, 0.9, ..., 0.1]

    for bw in bw_list:
        print(f"\n{'#'*60}")
        print(f"#  Bandwidth = {bw} MHz  —  Transformer: Distance + Angle + Confidence")
        print(f"#  Reading {NUM_LEVELS} obstruction levels (conf 1.0 → 0.1)")
        print(f"#{'='*59}")

        path_checkpoint = f"transformer_checkpoint_dist_angle_conf_{bw}MHz.pth"

        # ── Load all 10 datasets ──
        all_phase = []
        all_labels = []
        meta = None

        for lvl in range(NUM_LEVELS):
            conf_val = conf_labels[lvl]
            path_data = f"CSI_Angle_Dataset_{bw}MHz_4Rx_conf_{conf_val}.npz"
            path_meta = f"metadata_CSI_Angle_Dataset_{bw}MHz_4Rx_conf_{conf_val}.json"

            print(f"\n[{lvl+1}/{NUM_LEVELS}] Loading level {lvl} "
                  f"(wall_loss={lvl*10}dB, confidence={conf_val})")

            phase_lvl, labels_lvl, meta_lvl = pre_processing_dist_angle_conf(
                path_data, path_meta, confidence_label=conf_val
            )

            all_phase.append(phase_lvl)
            all_labels.append(labels_lvl)
            if meta is None:
                meta = meta_lvl

        # ── Combine & shuffle ──
        phase_combined = np.concatenate(all_phase, axis=0)
        labels_combined = np.concatenate(all_labels, axis=0)

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(phase_combined))
        phase_combined = phase_combined[idx]
        labels_combined = labels_combined[idx]

        # Report confidence distribution
        for conf_tag, conf_val in [("LOS (1.0)", 1.0), ("≥0.8", 0.8), ("≥0.5", 0.5), ("<0.5", 0.0)]:
            if conf_tag.startswith("LOS"):
                frac = np.mean(np.isclose(labels_combined[:, 2], conf_val))
            elif conf_tag.startswith("≥0.8"):
                frac = np.mean(labels_combined[:, 2] >= 0.8)
            elif conf_tag.startswith("≥0.5"):
                frac = np.mean((labels_combined[:, 2] >= 0.5) & (labels_combined[:, 2] < 0.8))
            else:
                frac = np.mean(labels_combined[:, 2] < 0.5)
            print(f"  {conf_tag}: {frac:.1%}")

        print(f"\nCombined dataset: {len(phase_combined)} samples")

        # ── Train ──
        Transformer_training_dist_angle_conf(
            processed_csi=phase_combined,
            label_array=labels_combined,
            path_checkpoint=path_checkpoint,
            meta=meta
        )
