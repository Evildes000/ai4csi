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
# For a quick run with only distance=2m, set: distance_list = np.array([2])
# distance_list = np.array([1, 2, 3, 5, 7, 10, 15, 20])   # 8 distances
distance_list = np.arange(1, 100, 1)
# --- Angle grid ---
# 2×2 array captures both sin(θ) and cos(θ) → unique for ALL angles 0°–360°.
# Training on 0°–180° in 15° steps (13 angles); extendable to full 360°.
angle_list = np.arange(0, 181, 10)       # [0, 15, 30, ..., 165, 180]

# --- Packets per (distance, angle) combination ---
num_packet = 500                        # Change to 2000 for finer statistics

print(f'Distances: {distance_list}')
print(f'Angles:    {angle_list}')
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

# ==========================================
# Dataset Structure
# ==========================================
CSI_Dataset = {}

# Subcarrier index vector for phase rotation
k = np.arange(Nfft)                    # [0, 1, ..., Nfft-1]

# ==========================================
# Main Loop: distance × angle
# ==========================================
t_start = time.time()

for d in distance_list:
    # --------------------------------------------------
    # Propagation delay
    # --------------------------------------------------
    tau = d / c

    # --------------------------------------------------
    # Path loss — Free Space Path Loss
    # --------------------------------------------------
    pathLoss_dB = 20 * np.log10(4 * np.pi * d / lambda_)
    pathGain = 10 ** (-pathLoss_dB / 20)

    # --------------------------------------------------
    # Noise power (fixed per distance)
    # --------------------------------------------------
    SNR_dB = 25
    expected_signal_power = pathGain ** 2
    noise_power = expected_signal_power / (10 ** (SNR_dB / 10))

    # --------------------------------------------------
    # Subcarrier-dependent phase rotation (encodes delay τ)
    # --------------------------------------------------
    phase_rotation = np.exp(-1j * 2 * np.pi * k * delta_f * tau)  # [Nfft]

    for theta_deg in angle_list:
        t_combo = time.time()
        print(f'  d={d:2d}m, θ={theta_deg:3d}° ...', end=' ', flush=True)

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
            # newaxis adds a dim at where newaxis is.
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

# ==========================================
# Save Dataset
# ==========================================
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
}

output_prefix = f'CSI_Angle_Dataset_{BW}MHz_{Nrx}Rx_test'
np.savez_compressed(f'{output_prefix}.npz', **CSI_Dataset)

# Convert numpy scalars to native Python types for JSON serialization
for key, val in metadata.items():
    if hasattr(val, 'item'):
        metadata[key] = val.item()

with open(f'metadata_{output_prefix}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'CSI Angle Dataset Saved → {output_prefix}.npz + metadata_{output_prefix}.json')

# ==========================================
# Visualization Example
# ==========================================
# Show CSI for distance=2m, angle=45°
example_dist = 1
example_angle = 45
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
