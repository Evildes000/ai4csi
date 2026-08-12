"""
CSI Simulation for Indoor Scene with Multi-Thickness Walls
============================================================

Simulates the floor plan shown in scene.png:
  - AP, Node1, Node2, Node3 placed in 4 rooms separated by walls
  - Walls have different thicknesses → different attenuation levels
  - All AP↔Node and Node↔Node links are simulated
  - Ray-wall intersection detects which walls each link crosses
  - Total wall loss = Σ (thickness × attenuation_coefficient) per crossed wall

Scene layout (top-down view, units in meters):
            y=100 ┌────────────────────────────┐
                  │                            │
                  │           Node5            │
                  │          (30,55)           │
                  │    Node4  Node6            │
             y=50 │───(45,40)─(25,48)wall_1───│
                  │                            │
                  │  Node3  Node1              │
                  │ (65,25) (60,15)            │
                  │       Node2  AP            │
             y=0  │      (55,5) (80,0)         │
                 x=0                          x=100

Signal model: single-path Rayleigh fading + FSPL + wall attenuation + AWGN
Antenna array: 2×2 rectangular, d = λ/3
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os


# ================================================================
# 1. OFDM & Physical Parameters
# ================================================================
BW = int(input('Select bandwidth (20/40/80/100 MHz): '))

if BW == 20:
    Fs, Nfft, Ncp = 20e6, 64, 16
elif BW == 40:
    Fs, Nfft, Ncp = 40e6, 128, 32
elif BW == 80:
    Fs, Nfft, Ncp = 80e6, 256, 64
elif BW == 100:
    Fs, Nfft, Ncp = 100e6, 320, 80
else:
    raise ValueError('Unsupported bandwidth. Choose 20, 40, 80, or 100 MHz.')

print(f'Bandwidth = {BW} MHz,  Nfft = {Nfft},  Ncp = {Ncp}')

c        = 3e8
fc       = 5.8e9
lambda_  = c / fc                    # ≈ 0.0517 m
delta_f  = Fs / Nfft                 # subcarrier spacing
k        = np.arange(Nfft)           # subcarrier indices
SNR_dB   = 25                        # reference SNR (no-wall baseline)

# ── Antenna array: 2×2 rectangular ──
Nrx = 4
d_ant = lambda_ / 3                  # λ/3 spacing
ant_positions = np.array([
    [0,       0],
    [d_ant,   0],                    # x-axis neighbour → cos(θ)
    [0,       d_ant],                # y-axis neighbour → sin(θ)
    [d_ant,   d_ant],                # diagonal → cos(θ)+sin(θ)
])

# ── QPSK pilots ──
np.random.seed(42)
pilot_bits = np.random.randint(0, 2, Nfft * 2)
pilot_bits_reshaped = pilot_bits.reshape(Nfft, 2)
bit_pairs = pilot_bits_reshaped[:, 0] * 2 + pilot_bits_reshaped[:, 1]
qpsk_map = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
pilot_symbol = qpsk_map[bit_pairs]

num_packet = 500

# ================================================================
# 2. Scene Geometry — Walls & Node Positions
# ================================================================



# square room: 100*100. Origin (0,0) at left-bottom corner. Units in meters.

# ── Nodes ──
nodes = {
    'BS':  np.array([50.0,  10.0]),
    'UE1': np.array([62.0,  28.0]),
    'UE2': np.array([65.0,  50.0]),
    'UE3': np.array([80.0,  55.0]),
    'UE4': np.array([90.0,  40.0]),
    'UE5': np.array([35.0,  50.0]),
    'UE6': np.array([25.0,  48.0]),
    'UE7': np.array([20.0,  30.0]),
    'UE8': np.array([50.0,  60.0]),
    'UE9': np.array([70.0,  80.0]),
    'UE10': np.array([80.0,  90.0]),
    'UE11': np.array([35.0,  70.0]),
    'UE12': np.array([30.0,  80.0]),
    'UE13': np.array([55.0,  30.0]),
    'UE14': np.array([45.0,  40.0]),
    'UE15': np.array([62.0,  62.0]),
}


# ── Walls ──
# Each wall is defined as a line segment (start, end) + thickness (m) + label.
# The line segment is the CENTRE-LINE of the wall rectangle.
# Thickness extends symmetrically to both sides of the centre-line.
#
# The wall's insertion loss at 5.8 GHz is modelled as:
#     loss_dB = attenuation_coefficient_dB_per_m × thickness_m
#
# Typical values for common materials at 5.8 GHz:
#   Brick wall (10 cm):    ~10–15 dB   →  coeff ≈ 120 dB/m
#   Concrete (15 cm):      ~20–25 dB   →  coeff ≈ 150 dB/m
#   Drywall/plasterboard:  ~3–5 dB     →  coeff ≈  50 dB/m
#   Glass:                 ~2–4 dB     →  coeff ≈  30 dB/m

ATTEN_COEFF = 100.0          # dB/m  (generic brick/concrete at 5.8 GHz)

# walls = [
#     # ── Vertical wall (left ↔ right separator) ──
#     # Upper segment: thicker (0.30 m)
#     {'label': 'V-wall-upper',  'p1': (5.5, 5.5), 'p2': (5.5, 10.0),
#      'thickness': 0.30},
#     # Lower segment: thinner (0.20 m)
#     {'label': 'V-wall-lower',  'p1': (5.5, 0.0), 'p2': (5.5, 5.5),
#      'thickness': 0.20},

#     # ── Horizontal wall (top ↔ bottom separator) ──
#     # Left segment: thinnest (0.15 m)
#     {'label': 'H-wall-left',   'p1': (0.0, 5.5), 'p2': (5.5, 5.5),
#      'thickness': 0.15},
#     # Right segment: medium (0.25 m)
#     {'label': 'H-wall-right',  'p1': (5.5, 5.5), 'p2': (12.0, 5.5),
#      'thickness': 0.25},
# ]



# P1, P2分别是一面墙中线两个断点的坐标，thickness是墙的厚度，label是墙的标签。墙的插入损耗可以通过厚度和衰减系数计算得出。 
walls = [

    {'label': 'wall_1', 'p1': (40.0, 00.0), 'p2': (40.0, 50.0), 'thickness': 0.30},  # 50dB
    {'label': 'wall_2', 'p1': (60.0, 00.0), 'p2': (60.0, 50.0), 'thickness': 0.30},  # 30dB
    {'label': 'wall_3', 'p1': (60.0, 60.0), 'p2': (100.0, 60.0), 'thickness': 0.30},  # 10dB
    {'label': 'wall_4', 'p1': (00.0, 60.0), 'p2': (40.0, 60.0), 'thickness': 0.30},  # 10dB
    {'label': 'wall_5', 'p1': (60.0, 65.0), 'p2': (60.0, 100.0), 'thickness': 0.30},  # 10dB
    {'label': 'wall_6', 'p1': (40.0, 65.0), 'p2': (40.0, 100.0), 'thickness': 0.30},  # 10dB

]

# ── Links auto-generated from nodes (bidirectional, d ≤ WIFI_RANGE) ──
WIFI_RANGE = 30.0  # metres

node_names = list(nodes.keys())
links = []
for i, a in enumerate(node_names):
    pa = nodes[a]
    for b in node_names[i + 1:]:
        pb = nodes[b]
        if np.linalg.norm(pa - pb) <= WIFI_RANGE:
            links.append((a, b))
            links.append((b, a))

print(f'  Auto-generated {len(links)} links (d <= {WIFI_RANGE}m)')





# ================================================================
# 3. Geometry Utilities — Wall Intersection & Angle
# ================================================================

def _orientation(a, b, c):
    """Cross product (b-a) × (c-a).  >0 → ccw, <0 → cw, ==0 → collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, q1, q2):
    """Return True if open segments p1p2 and q1q2 intersect (excl. endpoints)."""
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    # General case: strictly opposite signs
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False


def link_wall_loss(pt_a, pt_b, walls, atten_coeff=ATTEN_COEFF):
    """
    Compute total wall attenuation (dB) on the line-of-sight path from A to B.

    For each wall, test whether the line segment AB intersects the wall's
    centre-line segment.  If yes, add:  thickness × atten_coeff (dB).

    Returns:
        total_loss_dB:  float
        crossed_labels: list of wall labels that were crossed
    """
    total_loss = 0.0
    crossed = []

    for w in walls:
        if segments_intersect(pt_a, pt_b,
                              np.array(w['p1']), np.array(w['p2'])):
            loss = w['thickness'] * atten_coeff
            total_loss += loss
            crossed.append(w['label'])

    return total_loss, crossed


def compute_angle(pt_a, pt_b):
    """
    Angle from A to B in degrees, measured CCW from the positive x-axis.
    Returns ∈ [0, 360).
    """
    dx = pt_b[0] - pt_a[0]
    dy = pt_b[1] - pt_a[1]
    angle_rad = np.arctan2(dy, dx)       # (-π, +π]
    angle_deg = np.rad2deg(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0
    return angle_deg


def compute_distance(pt_a, pt_b):
    return np.linalg.norm(pt_b - pt_a)


# ================================================================
# 4. CSI Generation (single-path Rayleigh + FSPL + wall loss + AWGN)
# ================================================================

def generate_csi_for_link(pt_tx, pt_rx, wall_loss_dB, num_packet,
                          seed_offset=0):
    """
    Generate CSI for one link (TX → RX).

    Parameters
    ----------
    pt_tx, pt_rx :  (2,) ndarray   — positions in metres
    wall_loss_dB :  float          — total wall attenuation on this path
    num_packet   :  int
    noise_power : float        — noise power (fixed, referenced to
                                     no-wall FSPL; wall loss degrades SNR)
    seed_offset  :  int            — per-link seed offset for reproducibility

    Returns
    -------
    CSI  :  ndarray [num_packet, Nrx, Nfft], complex64
    distance : float
    angle    : float (degrees)
    """
    rng = np.random.RandomState(42 + seed_offset)

    # ── Geometry ──
    distance = compute_distance(pt_tx, pt_rx)
    angle_deg = compute_angle(pt_tx, pt_rx)
    angle_rad = np.deg2rad(angle_deg)

    # ── Propagation delay ──
    tau = distance / c

    # ── FSPL + wall loss ──
    fspl_dB = 20 * np.log10(4 * np.pi * distance / lambda_)
    total_loss_dB = fspl_dB + wall_loss_dB
    path_gain = 10 ** (-total_loss_dB / 20)
    # ── Noise power: based on no-wall FSPL at this distance (matches training) ──
    path_gain_no_wall = 10 ** (-fspl_dB / 20)
    expected_signal_power = path_gain_no_wall ** 2
    noise_power = expected_signal_power / (10 ** (SNR_dB / 10))

    # ── Phase rotation (subcarrier-dependent, encodes ToF) ──
    phase_rotation = np.exp(-1j * 2 * np.pi * k * delta_f * tau)   # [Nfft]

    # ── Antenna array response (encodes AoA) ──
    ant_response = np.exp(
        -1j * 2 * np.pi * (
            ant_positions[:, 0] * np.cos(angle_rad) +
            ant_positions[:, 1] * np.sin(angle_rad)
        ) / lambda_
    )  # [Nrx]

    # ── Per-packet CSI ──
    CSI = np.zeros((num_packet, Nrx, Nfft), dtype=np.complex64)

    for pkt in range(num_packet):
        # Rayleigh fading
        h = (rng.randn() + 1j * rng.randn()) / np.sqrt(2)

        # Frequency-domain channel
        H = (h * path_gain) * phase_rotation[np.newaxis, :] * ant_response[:, np.newaxis]

        # Transmit → Receive
        rxFreq = H * pilot_symbol[np.newaxis, :]                  # [Nrx, Nfft]

        # AWGN (noise power fixed → wall loss degrades effective SNR)
        noise = np.sqrt(noise_power / 2) * (
            rng.randn(Nrx, Nfft) + 1j * rng.randn(Nrx, Nfft)
        )
        rxFreq = rxFreq + noise.astype(np.complex64)

        # Zero-forcing CSI estimate
        CSI[pkt, :, :] = rxFreq / pilot_symbol[np.newaxis, :]

    return CSI, distance, angle_deg


# ================================================================
# 5. Main Simulation
# ================================================================

def main():
    print('\n' + '=' * 65)
    print('  Indoor Scene CSI Simulation — Multi-Thickness Walls')
    print('=' * 65)

    # ── Reference noise power (based on FSPL at d=1 m, no-wall) ──
    # This is the receiver's fixed noise floor.
    # Any wall loss reduces the signal → reduces effective SNR.
    fspl_ref_dB = 20 * np.log10(4 * np.pi * 1.0 / lambda_)
    path_gain_ref = 10 ** (-fspl_ref_dB / 20)
    noise_power = (path_gain_ref ** 2) / (10 ** (SNR_dB / 10))

    # ── Precompute link info ──
    print('\n  Link Analysis:')
    print(f'  {"Link":<18} {"Dist(m)":<10} {"Angle(°)":<10} '
          f'{"WallLoss(dB)":<14} {"EffSNR(dB)":<12} {"CrossedWalls"}')
    print(f'  {"─"*18} {"─"*10} {"─"*10} {"─"*14} {"─"*12} {"─"*30}')

    link_info = {}
    for tx_name, rx_name in links:
        pt_tx = nodes[tx_name]
        pt_rx = nodes[rx_name]
        distance = compute_distance(pt_tx, pt_rx)
        angle = compute_angle(pt_tx, pt_rx)
        wall_loss, crossed = link_wall_loss(pt_tx, pt_rx, walls)

        # Effective SNR
        fspl_dB = 20 * np.log10(4 * np.pi * distance / lambda_)
        eff_snr = SNR_dB - wall_loss

        link_info[(tx_name, rx_name)] = {
            'distance':    distance,
            'angle':       angle,
            'wall_loss':   wall_loss,
            'crossed':     crossed,
            'eff_snr':     eff_snr,
        }

        crossed_str = ', '.join(crossed) if crossed else '(none)'
        print(f'  {tx_name:>5} → {rx_name:<5}   '
              f'{distance:<10.2f} {angle:<10.1f} '
              f'{wall_loss:<14.1f} {eff_snr:<12.1f} {crossed_str}')

    # ── Generate CSI for each link ──
    CSI_Dataset = {}
    metadata_links = {}
    t_start = time.time()

    print(f'\n  Generating CSI ({num_packet} packets/link)...')
    for idx, (tx_name, rx_name) in enumerate(links):
        info = link_info[(tx_name, rx_name)]
        print(f'  [{idx+1}/{len(links)}] {tx_name} → {rx_name}  '
              f'(d={info["distance"]:.1f}m, θ={info["angle"]:.0f}°, '
              f'wall={info["wall_loss"]:.0f}dB) ...', end=' ', flush=True)

        CSI, dist, angle = generate_csi_for_link(
            nodes[tx_name], nodes[rx_name],
            info['wall_loss'], num_packet,
            seed_offset=idx
        )

        field_name = f'{tx_name}_to_{rx_name}'
        CSI_Dataset[field_name] = CSI
        metadata_links[field_name] = {
            'tx':           tx_name,
            'rx':           rx_name,
            'tx_position':  nodes[tx_name].tolist(),
            'rx_position':  nodes[rx_name].tolist(),
            'distance_m':   float(dist),
            'angle_deg':    float(angle),
            'wall_loss_dB': float(info['wall_loss']),
            'crossed_walls': info['crossed'],
            'effective_SNR_dB': float(info['eff_snr']),
        }
        print(f'done ({time.time() - t_start:.1f}s)')

    print(f'\n  Total simulation time: {time.time() - t_start:.1f}s')

    # ================================================================
    # 6. Save Dataset
    # ================================================================
    output_prefix = f'CSI_Scene_{BW}MHz_{Nrx}Rx'

    # ── Save CSI ──
    np.savez_compressed(f'{output_prefix}.npz', **CSI_Dataset)
    print(f'\n  CSI saved → {output_prefix}.npz')

    # ── Save metadata ──
    metadata = {
        'BW':               BW,
        'Fs':               Fs,
        'Nfft':             Nfft,
        'Ncp':              Ncp,
        'Nrx':              Nrx,
        'd_ant':            d_ant,
        'fc':               fc,
        'lambda':           lambda_,
        'delta_f':          delta_f,
        'c':                c,
        'SNR_dB':           SNR_dB,
        'noise_power':  noise_power,
        'num_packet':       num_packet,
        'atten_coeff_dBpm': ATTEN_COEFF,
        'walls':            walls,
        'nodes':            {k: v.tolist() for k, v in nodes.items()},
        'links':            metadata_links,
    }
    # Convert numpy scalars
    for key, val in metadata.items():
        if hasattr(val, 'item'):
            metadata[key] = val.item()

    with open(f'metadata_{output_prefix}.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f'  Metadata saved → metadata_{output_prefix}.json')

    # ================================================================
    # 7. Visualization — Scene Layout
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Draw walls as filled rectangles ──
    for w in walls:
        p1, p2 = np.array(w['p1']), np.array(w['p2'])
        t = w['thickness']
        # Direction vector (perpendicular)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])       # perpendicular

        # Rectangle corners
        c1 = p1 + normal * t / 2
        c2 = p1 - normal * t / 2
        c3 = p2 - normal * t / 2
        c4 = p2 + normal * t / 2
        corners = np.array([c1, c2, c3, c4])

        # Color by thickness
        alpha = min(0.3 + 0.5 * (t / 0.50), 1.0)    # thicker → more opaque, clamped to [0,1]
        ax.fill(corners[:, 0], corners[:, 1],
                facecolor='gray', edgecolor='black',
                linewidth=1.5, alpha=alpha)
        # Label
        mid = (p1 + p2) / 2
        ax.annotate(f'{t*100:.0f} cm', xy=mid,
                    ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))

    # ── Draw links ──
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (tx_name, rx_name) in enumerate(links):
        info = link_info[(tx_name, rx_name)]
        pt_tx = nodes[tx_name]
        pt_rx = nodes[rx_name]
        ax.plot([pt_tx[0], pt_rx[0]], [pt_tx[1], pt_rx[1]],
                color=colors[idx % len(colors)], linewidth=1.5, alpha=0.7, linestyle='--')

        # Stagger labels along the link direction to avoid overlaps
        direction = pt_rx - pt_tx
        norm = np.linalg.norm(direction)
        if norm > 0:
            along = direction / norm
            # Spread 21 labels at unique fractions along their links
            t = 0.25 + 0.5 * (idx / max(len(links) - 1, 1))      # 0.25 → 0.75
            label_pos = pt_tx + along * norm * t
        else:
            label_pos = pt_tx

        ax.annotate(f'{info["wall_loss"]:.0f} dB',
                    xy=label_pos, fontsize=6, color=colors[idx % len(colors)],
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', alpha=0.85))

    # ── Draw nodes ──
    for name, pos in nodes.items():
        if name == 'AP':
            marker, size, color = 's', 180, '#d62728'
        else:
            marker, size, color = 'o', 150, '#1f77b4'
        ax.scatter(pos[0], pos[1], marker=marker, s=size,
                   c=color, edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(name, xy=pos, xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

    # ── Styling ──
    ax.set_xlim(-5.0, 105.0)
    ax.set_ylim(-5.0, 105.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Indoor Scene — {BW} MHz CSI Simulation\n'
                 f'(wall atten. coeff. = {ATTEN_COEFF} dB/m at {fc/1e9:.1f} GHz)')
    ax.grid(True, alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color='gray', alpha=0.5, label='Wall (thickness on label)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
                   markersize=10, label='BS'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                   markersize=10, label='UE'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    os.makedirs('scene_figures', exist_ok=True)
    plt.savefig(f'scene_figures/{output_prefix}_layout.png', dpi=200,
                bbox_inches='tight', pad_inches=0)
    plt.savefig(f'scene_figures/{output_prefix}_layout.svg',
                bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f'  Scene layout saved → scene_figures/{output_prefix}_layout.png')

    print('\n' + '=' * 65)
    print('  Done.')
    print('=' * 65)


if __name__ == '__main__':
    main()
