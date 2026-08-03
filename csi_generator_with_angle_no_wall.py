"""
Single-Path WiFi CSI Simulation with Angle of Arrival — NO-WALL (FSPL only)

Standalone version that generates CSI data using Free Space Path Loss only
(no wall obstruction). Same antenna array, OFDM parameters, and signal model
as csi_generator_with_angle.py.

Key design:
  - 2×2 rectangular array: d_ant = λ/3
  - Distance: 1–50 m
  - Angle:    0°–350° (10° step, 36 angles)
  - SNR: 25 dB
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
Nrx = 4
d_ant = lambda_ / 3                    # λ/3 spacing (~1.72 cm)
ant_positions = np.array([
    [0,       0],
    [d_ant,   0],
    [0,       d_ant],
    [d_ant,   d_ant],
])  # shape: [Nrx, 2]
print(f'Array: 2×2 rectangular, d = λ/3 = {d_ant*100:.2f} cm')

# ==========================================
# Dataset Parameters
# ==========================================
distance_list = np.arange(1, 51, 2)      # 1, 3, 5, ..., 49 m
angle_list    = np.arange(0, 360, 10)    # 0°–350°, 36 angles
num_packet    = 500

print(f'Distances: {len(distance_list)} points from {distance_list[0]}–{distance_list[-1]}m')
print(f'Angles:    {len(angle_list)} points from {angle_list[0]}°–{angle_list[-1]}°')
print(f'Packets per (d,θ): {num_packet}')
print(f'Total combinations: {len(distance_list)} × {len(angle_list)} = {len(distance_list)*len(angle_list)}')
print(f'Total samples: {len(distance_list)*len(angle_list)*num_packet}')

# ==========================================
# Generate Pilot Symbols (QPSK)
# ==========================================
np.random.seed(42)
pilot_bits = np.random.randint(0, 2, Nfft * 2)
pilot_bits_reshaped = pilot_bits.reshape(Nfft, 2)
bit_pairs = pilot_bits_reshaped[:, 0] * 2 + pilot_bits_reshaped[:, 1]
qpsk_map = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
pilot_symbol = qpsk_map[bit_pairs]

k = np.arange(Nfft)
SNR_dB = 25

# ==========================================
# Main Loop
# ==========================================
CSI_Dataset = {}
t_start = time.time()

for d in distance_list:
    tau = d / c

    pathLoss_dB = 20 * np.log10(4 * np.pi * d / lambda_)
    pathGain = 10 ** (-pathLoss_dB / 20)

    expected_signal_power = pathGain ** 2
    noise_power = expected_signal_power / (10 ** (SNR_dB / 10))

    phase_rotation = np.exp(-1j * 2 * np.pi * k * delta_f * tau)

    for theta_deg in angle_list:
        t_combo = time.time()
        print(f'  d={d:2d}m, θ={theta_deg:3d}° ...', end=' ', flush=True)

        theta_rad = np.deg2rad(theta_deg)
        antenna_response = np.exp(
            -1j * 2 * np.pi * (
                ant_positions[:, 0] * np.cos(theta_rad) +
                ant_positions[:, 1] * np.sin(theta_rad)
            ) / lambda_
        )

        CSI_temp = np.zeros((num_packet, Nrx, Nfft), dtype=np.complex64)

        for pkt in range(num_packet):
            h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            H = (h * pathGain) * phase_rotation[np.newaxis, :] * antenna_response[:, np.newaxis]

            txFreq = pilot_symbol
            rxFreq = H * txFreq[np.newaxis, :]
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(Nrx, Nfft) + 1j * np.random.randn(Nrx, Nfft)
            )
            rxFreq = rxFreq + noise.astype(np.complex64)
            CSI = rxFreq / txFreq[np.newaxis, :]

            CSI_temp[pkt, :, :] = CSI

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
    'SNR_dB':              SNR_dB,
    'wall_loss_dB':        0,
    'scenario':            'FSPL only (no wall)',
}

output_prefix = f'CSI_Angle_Dataset_{BW}MHz_{Nrx}Rx'
np.savez_compressed(f'{output_prefix}.npz', **CSI_Dataset)

for key, val in metadata.items():
    if hasattr(val, 'item'):
        metadata[key] = val.item()

with open(f'metadata_{output_prefix}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'CSI Angle Dataset Saved → {output_prefix}.npz + metadata_{output_prefix}.json')

# ==========================================
# Visualization
# ==========================================
example_dist = 1
example_angle = 45
field_name = f'dist_{example_dist}_angle_{example_angle}'

if field_name in CSI_Dataset:
    CSI_example = CSI_Dataset[field_name]
    CSI_one_packet = CSI_example[0, :, :]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for rx in range(Nrx):
        axes[0].plot(np.abs(CSI_one_packet[rx, :]),
                     linewidth=1.5, label=f'Rx Antenna {rx+1}')
    axes[0].grid(True)
    axes[0].set_xlabel('Subcarrier Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'CSI Amplitude (d={example_dist}m, θ={example_angle}°) — No Wall')
    axes[0].legend()

    for rx in range(Nrx):
        unwrapped_phase = np.unwrap(np.angle(CSI_one_packet[rx, :]))
        axes[1].plot(unwrapped_phase,
                     linewidth=1.5, label=f'Rx Antenna {rx+1}')
    axes[1].grid(True)
    axes[1].set_xlabel('Subcarrier Index')
    axes[1].set_ylabel('Unwrapped Phase (rad)')
    axes[1].set_title(f'CSI Unwrapped Phase (d={example_dist}m, θ={example_angle}°) — No Wall')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_example.png', dpi=150)
    plt.show()
else:
    print(f'\nNote: (d={example_dist}m, θ={example_angle}°) not in dataset; skipping visualization.')
