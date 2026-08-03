"""
Single-Path WiFi CSI Simulation — TEST Dataset Generator

Generates a test dataset that does NOT overlap with data from
csi_generator_with_angle.py.  Three axes ensure independence:

  1. Random seed:     12345  (train = 42)
  2. Distance grid:   [2, 4, 6, ..., 50]  (train = [1, 3, 5, ..., 49])
  3. Angle grid:      [5, 15, 25, ..., 355]  (train = [0, 10, 20, ..., 350])

Additionally, fewer packets per combination (200 vs 500) keep test-set size
manageable while still providing enough statistical reliability.

Same 10 obstruction levels (0–90 dB loss → confidence 1.0–0.1) as the training
set to allow direct evaluation on identical confidence labels.

Usage example:
  python csi_generator_test.py
  → Select bandwidth → generates 10 .npz + 10 metadata .json files
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
print(f'Array: 2x2 rectangular, d = lambda/3 = {d_ant*100:.2f} cm (avoids +/-pi boundary)')

# ==========================================
# TEST Dataset Parameters (offset from train)
# ==========================================
# --- Distance grid ---
# TRAIN: [1, 3, 5, ..., 49]  →  TEST: [2, 4, 6, ..., 50]
# Offsetting by 1 m ensures zero overlap in spatial positions.
distance_list = np.arange(2, 51, 2)      # [2, 4, 6, ..., 50]

# --- Angle grid ---
# TRAIN: [0, 10, 20, ..., 350]  →  TEST: [5, 15, 25, ..., 355]
# Offsetting by 5° ensures zero overlap in angular positions.
angle_list = np.arange(5, 360, 10)       # [5, 15, 25, ..., 355]

# --- Packets per (distance, angle) combination ---
# TRAIN: 500  →  TEST: 200 (fewer, but sufficient for reliable metrics)
num_packet = 200

print(f'Distances: {len(distance_list)} points from {distance_list[0]}–{distance_list[-1]}m')
print(f'Angles:    {len(angle_list)} points from {angle_list[0]}°–{angle_list[-1]}°')
print(f'Packets per (d,theta): {num_packet}')
print(f'Total combinations: {len(distance_list)} x {len(angle_list)} = {len(distance_list)*len(angle_list)}')
print(f'Total samples: {len(distance_list)*len(angle_list)*num_packet}')

# ==========================================
# Generate Pilot Symbols (QPSK, unit average power)
# ==========================================
# DIFFERENT seed than training script (42) → independent random realizations
np.random.seed(12345)
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
        # Subcarrier-dependent phase rotation (encodes delay tau)
        # --------------------------------------------------
        phase_rotation = np.exp(-1j * 2 * np.pi * k * delta_f * tau)  # [Nfft]

        for theta_deg in angle_list:
            t_combo = time.time()
            label = f'd={d:2d}m, theta={theta_deg:3d}deg'
            print(f'  {label} ...', end=' ', flush=True)

            theta_rad = np.deg2rad(theta_deg)

            # --------------------------------------------------
            # 2x2 rectangular array response:
            #   a_i(theta) = exp(-j*2pi*(x_i*cos(theta) + y_i*sin(theta)) / lambda)
            #   -> x-axis diffs encode cos(theta), y-axis diffs encode sin(theta)
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
                # Frequency-domain channel with AoA (2x2 array):
                #   H(rx,k) = h * exp(-j2pi*k*delta_f*tau)
                #            * exp(-j2pi*(x_rx*cos(theta)+y_rx*sin(theta))/lambda)
                #
                #   -> phase_rotation[k] encodes distance (ToF)
                #   -> antenna_response[rx] encodes cos(theta) & sin(theta) via 2x2 array
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
        'dataset_type':        'test',
        'random_seed':         12345,
        'description':         'Test dataset: distances [2:2:50], '
                               'angles [5:10:355], seed=12345. '
                               'Independent of training set '
                               '(dist [1:2:49], angles [0:10:350], seed=42).',
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    # Convert numpy scalars to native Python types for JSON serialization
    for key, val in metadata.items():
        if hasattr(val, 'item'):
            metadata[key] = val.item()

    with open(output_path_json, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'CSI Dataset Saved -> {output_path_npz} + {output_path_json}')


def visualize_example(CSI_Dataset, output_prefix, example_dist=2, example_angle=50):
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
        axes[0].set_title(f'CSI Amplitude (d={example_dist}m, theta={example_angle}deg)')
        axes[0].legend()

        for rx in range(Nrx):
            unwrapped_phase = np.unwrap(np.angle(CSI_one_packet[rx, :]))
            axes[1].plot(unwrapped_phase,
                         linewidth=1.5, label=f'Rx Antenna {rx+1}')

        axes[1].grid(True)
        axes[1].set_xlabel('Subcarrier Index')
        axes[1].set_ylabel('Unwrapped Phase (rad)')
        axes[1].set_title(f'CSI Unwrapped Phase (d={example_dist}m, theta={example_angle}deg)')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'{output_prefix}_example.png', dpi=150)
        plt.show()
    else:
        print(f'\nNote: (d={example_dist}m, theta={example_angle}deg) not in dataset; '
              f'skipping visualization.')
        print(f'Available fields: {list(CSI_Dataset.keys())[:5]}...')


# ================================================================
# Generate TEST CSI Datasets — 10 obstruction levels -> 10 confidence labels
# ================================================================
# Level 0:  0 dB extra loss -> confidence 1.0  (FSPL only, LOS)
# Level 1: 10 dB extra loss -> confidence 0.9
# Level 2: 20 dB extra loss -> confidence 0.8
# ...
# Level 9: 90 dB extra loss -> confidence 0.1  (severe NLOS)
NUM_LEVELS = 10
wall_loss_levels = [lvl * 10 for lvl in range(NUM_LEVELS)]   # [0, 10, 20, ..., 90]
conf_labels      = [round(1.0 - lvl * 0.1, 1) for lvl in range(NUM_LEVELS)]  # [1.0, 0.9, ..., 0.1]

for lvl in range(NUM_LEVELS):
    wall_dB  = wall_loss_levels[lvl]
    conf_val = conf_labels[lvl]

    print('\n' + '=' * 60)
    print(f'Level {lvl}: wall_loss = {wall_dB} dB -> confidence = {conf_val}')
    print('=' * 60)

    CSI_Dataset = generate_csi_dataset(
        distance_list, angle_list, num_packet, wall_loss_dB=wall_dB
    )

    output_prefix = f'CSI_Test_Angle_Dataset_{BW}MHz_{Nrx}Rx_conf_{conf_val}'
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
                      example_dist=2, example_angle=50)

print('\n' + '=' * 60)
print('All done! Generated 10 TEST datasets:')
for lvl in range(NUM_LEVELS):
    conf_val = conf_labels[lvl]
    wall_dB  = wall_loss_levels[lvl]
    print(f'  {lvl}. CSI_Test_Angle_Dataset_{BW}MHz_{Nrx}Rx_conf_{conf_val}.npz'
          f'  — {wall_dB} dB loss, confidence={conf_val}')
print('=' * 60)
print('\nSummary of train/test separation:')
print(f'  TRAIN distances: [1, 3, 5, ..., 49]  |  TEST distances: [2, 4, 6, ..., 50]')
print(f'  TRAIN angles:    [0, 10, 20, ..., 350] | TEST angles:    [5, 15, 25, ..., 355]')
print(f'  TRAIN seed:      42                    | TEST seed:      12345')
print(f'  TRAIN packets:   500                   | TEST packets:   200')
