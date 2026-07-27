"""
Single-Path WiFi CSI Simulation with Angle of Arrival (Python version)

Extends the original csi_generator.py by adding Angle of Arrival (AoA)
information through per-antenna phase shifts across a Uniform Linear Array (ULA).

Key design:
  - 2×2 rectangular array: d_ant = λ/2  (~2.59 cm at 5.8 GHz)
  - Antenna positions: (0,0), (d,0), (0,d), (d,d)
  - Phase: φ_i = -2π·(x_i·cos(θ) + y_i·sin(θ)) / λ
  - x-axis diffs encode cos(θ), y-axis diffs encode sin(θ)
  - Together, (cos, sin) uniquely determine θ over full 360° — no ambiguity

Supports 10 obstruction levels with corresponding confidence labels:
  Level  0:  0 dB loss → confidence 1.0 (LOS, FSPL only)
  Level  1: 10 dB loss → confidence 0.9
  ...
  Level  9: 90 dB loss → confidence 0.1 (severe NLOS)
Each level saved to a separate file named by its confidence value.

Usage example (as described in requirement):
  distance=2m, angles 30°:10°:150°, 2000 packets each
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import time

# ==========================================
# OFDM Parameters
# ==========================================
BW = int(input('Select bandwidth (20/40/80/100 MHz): '))

if BW == 20:
    Fs   = 20e6
    Nfft = 64
    Ncp  = 16
elif BW == 40:
    Fs   = 40e6
    Nfft = 128
    Ncp  = 32
elif BW == 80:
    Fs   = 80e6
    Nfft = 256
    Ncp  = 64
elif BW == 100:
    Fs   = 100e6
    Nfft = 320
    Ncp  = 80
else:
    raise ValueError('Unsupported bandwidth. Choose 20, 40, 80, or 100 MHz.')

print(f'Bandwidth = {BW} MHz')
print(f'FFT Size  = {Nfft}')
print(f'CP Length = {Ncp}')

M = 4                       # QPSK modulation order
numSubcarrier = Nfft         # all subcarriers used (real WiFi uses subset)

# ==========================================
# Physical Constants
# ==========================================
c = 3e8                                # Speed of light (m/s)
fc = 5.8e9                             # Carrier frequency (Hz)
lambda_ = c / fc                       # Wavelength (~0.0517 m)
delta_f = Fs / Nfft                    # Subcarrier spacing (Hz)

# ==========================================
# Antenna Array Parameters (2×2 Rectangular Array)
# ==========================================
#  Layout (top-down view):
#         y ↑
#           │
#     2(0,d)┼——————──3(d,d)
#           │         │
#     ──────┼─────────┼──→ x
#     0(0,0)┼─———————─1(d,0)
#
#  Phase for antenna at (x_i, y_i):
#    φ_i = -2π · (x_i·cos(θ) + y_i·sin(θ)) / λ
#
#  2×2 array provides BOTH cos(θ) and sin(θ) → unique angle for full 360°
Nrx = 4
d_ant = lambda_ / 3                    # λ/3 spacing (~1.72 cm)
# λ/3 avoids ±π at boundaries: max diff = ±2π/3 ≈ ±120° (< 180°)
ant_positions = np.array([
    [0,       0],
    [d_ant,   0],        # x-axis neighbor of ant0 → encodes cos(θ)
    [0,       d_ant],    # y-axis neighbor of ant0 → encodes sin(θ)
    [d_ant,   d_ant],    # diagonal → encodes cos(θ)+sin(θ)
])  # shape: [Nrx, 2]
print(f'Array: 2×2 rectangular, d = λ/3 = {d_ant*100:.2f} cm (avoids ±π boundary)')

# ==========================================
# Dataset Parameters
# ==========================================
# --- Distance grid ---
# Multiple distances so the model can learn distance prediction.
# Range: 1–50 meters (0 is excluded — physically meaningless for FSPL).
distance_list = np.arange(1, 51, 2)      # [1, 3, 5, ..., 49]

# --- Angle grid ---
# 2×2 array captures both sin(θ) and cos(θ) → unique for ALL angles 0°–360°.
# Full 0°–350° in 10° steps (36 angles); 360° excluded to avoid 0°/360° duplicate.
angle_list = np.arange(0, 360, 10)       # [0, 10, 20, ..., 350]

# --- Packets per (distance, angle) combination ---
num_packet = 500                        # Change to 2000 for finer statistics

print(f'Distances: {len(distance_list)} points from {distance_list[0]}–{distance_list[-1]}m')
print(f'Angles:    {len(angle_list)} points from {angle_list[0]}°–{angle_list[-1]}°')
print(f'Packets per (d,θ): {num_packet}')
print(f'Total combinations: {len(distance_list)} × {len(angle_list)} = {len(distance_list)*len(angle_list)}')
print(f'Total samples: {len(distance_list)*len(angle_list)*num_packet}')

# ==========================================
# Generate Pilot Symbols (QPSK, unit average power)
# ==========================================
np.random.seed(42)                     # for reproducibility
pilot_bits = np.random.randint(0, 2, Nfft * 2)

# Convert bit pairs to QPSK symbols
pilot_bits_reshaped = pilot_bits.reshape(Nfft, 2)
bit_pairs = pilot_bits_reshaped[:, 0] * 2 + pilot_bits_reshaped[:, 1]
qpsk_map = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
pilot_symbol = qpsk_map[bit_pairs]     # shape: [Nfft]

# Subcarrier index vector for phase rotation
k = np.arange(Nfft)                    # [0, 1, ..., Nfft-1]

# Base SNR for the no-wall case
SNR_dB = 25


def generate_csi_dataset(distance_list, angle_list, num_packet, wall_loss_dB=0):
    """
    Generate CSI dataset for a given distance/angle grid.

    Parameters
    ----------
    distance_list : np.ndarray
        1-D array of distances in meters.
    angle_list : np.ndarray
        1-D array of angles in degrees.
    num_packet : int
        Number of packets per (distance, angle) combination.
    wall_loss_dB : float
        Additional wall penetration loss in dB (default 0 = FSPL only).

    Returns
    -------
    CSI_Dataset : dict
        Dictionary keyed by 'dist_{d}_angle_{angle}' with CSI arrays [num_packet, Nrx, Nfft].

    Notes
    -----
    - Noise power is fixed based on the no-wall signal level
      (SNR_dB = 25 dB referenced to FSPL-only path gain).
      Wall loss reduces the effective SNR accordingly.
    - Effective SNR with wall_loss_dB > 0: SNR_eff = SNR_dB - wall_loss_dB.
    """
    CSI_Dataset = {}
    t_start = time.time()

    for d in distance_list:
        # --------------------------------------------------
        # Propagation delay
        # --------------------------------------------------
        tau = d / c

        # --------------------------------------------------
        # Path loss — Free Space Path Loss (no-wall reference)
        # --------------------------------------------------
        pathLoss_dB_no_wall = 20 * np.log10(4 * np.pi * d / lambda_)
        pathGain_no_wall = 10 ** (-pathLoss_dB_no_wall / 20)

        # Actual path loss with wall attenuation
        pathLoss_dB = pathLoss_dB_no_wall + wall_loss_dB
        pathGain = 10 ** (-pathLoss_dB / 20)

        # --------------------------------------------------
        # Noise power — fixed receiver noise floor
        # Based on the no-wall SNR, so wall loss degrades effective SNR
        # --------------------------------------------------
        expected_signal_power = pathGain_no_wall ** 2
        noise_power = expected_signal_power / (10 ** (SNR_dB / 10))

        # --------------------------------------------------
        # Subcarrier-dependent phase rotation (encodes delay τ)
        # --------------------------------------------------
        phase_rotation = np.exp(-1j * 2 * np.pi * k * delta_f * tau)  # [Nfft]

        for theta_deg in angle_list:
            t_combo = time.time()
            label = f'd={d:2d}m, θ={theta_deg:3d}°'
            print(f'  {label} ...', end=' ', flush=True)

            theta_rad = np.deg2rad(theta_deg)

            # --------------------------------------------------
            # 2×2 rectangular array response:
            #   a_i(θ) = exp(-j·2π·(x_i·cos(θ) + y_i·sin(θ)) / λ)
            #   → x-axis diffs encode cos(θ), y-axis diffs encode sin(θ)
            # --------------------------------------------------
            antenna_response = np.exp(
                -1j * 2 * np.pi * (
                    ant_positions[:, 0] * np.cos(theta_rad) +
                    ant_positions[:, 1] * np.sin(theta_rad)
                ) / lambda_
            )  # shape: [Nrx]

            # Pre-allocate CSI storage: [num_packet, Nrx, Nfft]
            CSI_temp = np.zeros((num_packet, Nrx, Nfft), dtype=np.complex64)

            for pkt in range(num_packet):
                # --------------------------------------------------
                # Single-path Rayleigh fading (scalar h, same for all antennas)
                # --------------------------------------------------
                h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

                # --------------------------------------------------
                # Frequency-domain channel with AoA (2×2 array):
                #   H(rx,k) = h·exp(-j2π·k·Δf·τ) · exp(-j2π·(x_rx·cos(θ)+y_rx·sin(θ))/λ)
                #
                #   → phase_rotation[k] encodes distance (ToF)
                #   → antenna_response[rx] encodes cos(θ) & sin(θ) via 2×2 array
                # --------------------------------------------------
                H = (h * pathGain) * phase_rotation[np.newaxis, :] * antenna_response[:, np.newaxis]
                #                             [1,num_subcarriers]  * [num_rx, 1]
                # shape: [Nrx, Nfft]

                # --------------------------------------------------
                # Transmit & Receive (frequency domain)
                # --------------------------------------------------
                txFreq = pilot_symbol                                          # [Nfft]
                rxFreq = H * txFreq[np.newaxis, :]                             # [Nrx, Nfft]

                # Add AWGN
                noise = np.sqrt(noise_power / 2) * (
                    np.random.randn(Nrx, Nfft) + 1j * np.random.randn(Nrx, Nfft)
                )
                rxFreq = rxFreq + noise.astype(np.complex64)

                # --------------------------------------------------
                # CSI Estimation (Zero-Forcing)
                # --------------------------------------------------
                CSI = rxFreq / txFreq[np.newaxis, :]                           # [Nrx, Nfft]

                CSI_temp[pkt, :, :] = CSI

            # Store in dataset dict with compound key
            field_name = f'dist_{d}_angle_{theta_deg}'
            CSI_Dataset[field_name] = CSI_temp

            print(f'done ({time.time() - t_combo:.1f}s)')

    print(f'\nTotal simulation time: {time.time() - t_start:.1f}s')
    print(f'Total combinations: {len(CSI_Dataset)}, '
          f'Total samples: {len(CSI_Dataset) * num_packet}')

    return CSI_Dataset


def save_dataset(CSI_Dataset, output_prefix, metadata_extra=None):
    """Save a CSI Dataset to .npz and metadata .json files."""
    output_path_npz = f'{output_prefix}.npz'
    output_path_json = f'metadata_{output_prefix}.json'

    np.savez_compressed(output_path_npz, **CSI_Dataset)

    metadata = {
        'BW':                  BW,
        'Fs':                  Fs,
        'Nfft':                Nfft,
        'Ncp':                 Ncp,
        'Nrx':                 Nrx,
        'd_ant':               d_ant,
        'num_packet':          num_packet,
        'distance_list':       distance_list.tolist(),
        'angle_list':          angle_list.tolist(),
        'num_distances':       len(distance_list),
        'num_angles':          len(angle_list),
        'total_combinations':  len(distance_list) * len(angle_list),
        'c':                   c,
        'fc':                  fc,
        'lambda':              lambda_,
        'subcarriers':         numSubcarrier,
        'delta_f':             delta_f,
        'SNR_dB':              SNR_dB,
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    # Convert numpy scalars to native Python types for JSON serialization
    for key, val in metadata.items():
        if hasattr(val, 'item'):
            metadata[key] = val.item()

    with open(output_path_json, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'CSI Dataset Saved → {output_path_npz} + {output_path_json}')


def visualize_example(CSI_Dataset, output_prefix, example_dist=1, example_angle=45):
    """Visualize CSI amplitude & phase for one (distance, angle) example."""
    field_name = f'dist_{example_dist}_angle_{example_angle}'

    if field_name in CSI_Dataset:
        CSI_example = CSI_Dataset[field_name]      # shape: [num_packet, Nrx, Nfft]

        # Extract first packet
        CSI_one_packet = CSI_example[0, :, :]      # shape: [Nrx, Nfft]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        for rx in range(Nrx):
            axes[0].plot(np.abs(CSI_one_packet[rx, :]),
                         linewidth=1.5, label=f'Rx Antenna {rx+1}')

        axes[0].grid(True)
        axes[0].set_xlabel('Subcarrier Index')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'CSI Amplitude (d={example_dist}m, θ={example_angle}°)')
        axes[0].legend()

        for rx in range(Nrx):
            unwrapped_phase = np.unwrap(np.angle(CSI_one_packet[rx, :]))
            axes[1].plot(unwrapped_phase,
                         linewidth=1.5, label=f'Rx Antenna {rx+1}')

        axes[1].grid(True)
        axes[1].set_xlabel('Subcarrier Index')
        axes[1].set_ylabel('Unwrapped Phase (rad)')
        axes[1].set_title(f'CSI Unwrapped Phase (d={example_dist}m, θ={example_angle}°)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'{output_prefix}_example.png', dpi=150)
        plt.show()
    else:
        print(f'\nNote: (d={example_dist}m, θ={example_angle}°) not in dataset; skipping visualization.')
        print(f'Available fields: {list(CSI_Dataset.keys())[:5]}...')


# ================================================================
# Generate CSI Datasets — 10 obstruction levels → 10 confidence labels
# ================================================================
# Level 0:  0 dB extra loss → confidence 1.0  (FSPL only, LOS)
# Level 1: 10 dB extra loss → confidence 0.9
# Level 2: 20 dB extra loss → confidence 0.8
# ...
# Level 9: 90 dB extra loss → confidence 0.1  (severe NLOS)
NUM_LEVELS = 10
wall_loss_levels = [lvl * 10 for lvl in range(NUM_LEVELS)]   # [0, 10, 20, ..., 90]
conf_labels      = [round(1.0 - lvl * 0.1, 1) for lvl in range(NUM_LEVELS)]  # [1.0, 0.9, ..., 0.1]

for lvl in range(NUM_LEVELS):
    wall_dB  = wall_loss_levels[lvl]
    conf_val = conf_labels[lvl]

    print('\n' + '=' * 60)
    print(f'Level {lvl}: wall_loss = {wall_dB} dB → confidence = {conf_val}')
    print('=' * 60)

    CSI_Dataset = generate_csi_dataset(
        distance_list, angle_list, num_packet, wall_loss_dB=wall_dB
    )

    output_prefix = f'CSI_Angle_Dataset_{BW}MHz_{Nrx}Rx_conf_{conf_val}'
    save_dataset(CSI_Dataset, output_prefix,
                 metadata_extra={
                     'level':              lvl,
                     'wall_loss_dB':       wall_dB,
                     'confidence':         conf_val,
                     'effective_SNR_dB':   SNR_dB - wall_dB,
                     'scenario':           f'Wall-obstructed level {lvl} '
                                           f'({wall_dB} dB extra loss)',
                 })

    visualize_example(CSI_Dataset, output_prefix,
                      example_dist=1, example_angle=45)

print('\n' + '=' * 60)
print('All done! Generated 10 datasets:')
for lvl in range(NUM_LEVELS):
    conf_val = conf_labels[lvl]
    wall_dB  = wall_loss_levels[lvl]
    print(f'  {lvl}. CSI_Angle_Dataset_{BW}MHz_{Nrx}Rx_conf_{conf_val}.npz'
          f'  — {wall_dB} dB loss, confidence={conf_val}')
print('=' * 60)
