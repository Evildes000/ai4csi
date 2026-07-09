"""
Two-Transmitter One-Receiver: QPSK + OFDM with 3-Path Multipath Channel
======================================================================
TX1 → RX: Line-of-Sight  (LOS)  — dominant direct path (Rician, high K)
TX2 → RX: Non-Line-of-Sight (NLOS) — reflected paths only (Rayleigh-like)

- OFDM: 64 subcarriers, 16-sample CP, 52 active subcarriers
- Pilot: block-type — first 2 OFDM symbols are training symbols
- Channel estimation: LS averaging over training symbols
- Equalisation: Zero-Forcing
- Output: 4 constellation diagrams (LOS raw, LOS eq, NLOS raw, NLOS eq)
         + channel frequency responses
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== Parameters ====================
N_FFT       = 64          # Total OFDM subcarriers
N_CP        = 16          # Cyclic prefix length
N_SYMBOLS   = 200         # Total OFDM symbols (training + data)
N_TRAINING  = 2           # Block-type training symbols (at the beginning)
N_DATA_SC   = 52          # Active subcarriers (same as 802.11a: 4 pilots + 48 data)
N_DATA_SYMS = N_SYMBOLS - N_TRAINING   # 198 data symbols
SNR_DB      = 18          # AWGN SNR (dB)
RNG_SEED    = 42

np.random.seed(RNG_SEED)


# ==================== QPSK Mapper ====================
def qpsk_modulate(bits):
    """Gray-coded QPSK: 00→+1+j, 01→-1+j, 11→-1-j, 10→+1-j.  Unit power."""
    bits = np.asarray(bits, dtype=int)
    syms = np.empty(len(bits) // 2, dtype=complex)
    for i in range(0, len(bits), 2):
        real = 1.0 if bits[i] == 0 else -1.0
        imag = 1.0 if bits[i + 1] == 0 else -1.0
        syms[i // 2] = (real + 1j * imag) / np.sqrt(2)
    return syms


# ==================== OFDM Modulator ====================
def ofdm_modulate(data_per_symbol):
    """
    data_per_symbol : shape (N_DATA_SC, N_SYMBOLS)
    Returns: 1-D concatenated time-domain signal with CP,
             per-symbol matrix (N_SYMBOLS, N_FFT+N_CP)
    """
    half = N_DATA_SC // 2                     # 26
    tx_per_sym = np.zeros((N_SYMBOLS, N_FFT + N_CP), dtype=complex)

    for s in range(N_SYMBOLS):
        freq = np.zeros(N_FFT, dtype=complex)
        # positive subcarriers (skip DC at index 0)
        freq[1 : half + 1]          = data_per_symbol[:half, s]
        # negative subcarriers
        freq[N_FFT - half : N_FFT]  = data_per_symbol[half:, s]

        td = np.fft.ifft(freq) * np.sqrt(N_FFT)
        tx_per_sym[s] = np.concatenate([td[-N_CP:], td])

    return tx_per_sym.ravel(), tx_per_sym


# ==================== Multipath Channel ====================
def apply_multipath(signal, taps, delays):
    """Linear convolution with sparse multipath taps."""
    out = np.zeros_like(signal)
    for tap, dly in zip(taps, delays):
        out[dly:] += tap * signal[:len(signal) - dly]
    return out


def los_channel():
    """LOS 3-path: dominant direct ray + 2 weaker reflections (high Rician K)."""
    taps = [
        1.0,                                    # direct path (0 dB)
        0.35 * np.exp(1j * 0.6 * np.pi),       # -9 dB, 108°
        0.20 * np.exp(1j * 1.4 * np.pi),       # -14 dB, 252°
    ]
    delays = [0, 4, 9]
    return taps, delays


def nlos_channel():
    """NLOS 3-path: three reflected rays through heavy obstructions — severe fading."""
    taps = [
        0.50 * np.exp(1j * 0.10 * np.pi),      # -6.0 dB, 18°
        0.46 * np.exp(1j * 0.72 * np.pi),      # -6.7 dB, 130°
        0.38 * np.exp(1j * 1.50 * np.pi),      # -8.4 dB, 270°
    ]
    delays = [4, 11, 15]                       # full CP utilisation → strong frequency selectivity
    return taps, delays


# ==================== AWGN ====================
def add_awgn(signal, snr_db):
    """Add complex AWGN to *signal* at the given SNR (dB)."""
    sig_pow = np.mean(np.abs(signal) ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_pow / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


# ==================== OFDM Demodulator ====================
def ofdm_demodulate(rx_per_sym):
    """
    rx_per_sym : (N_SYMBOLS, N_FFT+N_CP)
    Returns:
        rx_data   : (N_DATA_SC, N_SYMBOLS)  — extracted data subcarriers
        rx_freq   : (N_FFT, N_SYMBOLS)       — full frequency-domain symbols
    """
    half = N_DATA_SC // 2
    rx_data = np.zeros((N_DATA_SC, N_SYMBOLS), dtype=complex)
    rx_freq = np.zeros((N_FFT, N_SYMBOLS), dtype=complex)

    for s in range(N_SYMBOLS):
        td = rx_per_sym[s, N_CP:]                    # discard CP
        freq = np.fft.fft(td) / np.sqrt(N_FFT)
        rx_freq[:, s] = freq
        rx_data[:half, s] = freq[1 : half + 1]
        rx_data[half:, s] = freq[N_FFT - half : N_FFT]

    return rx_data, rx_freq


# ==================== Channel Estimation & Equalisation ====================
def estimate_and_equalize(rx_data, tx_training):
    """
    Block-type LS channel estimation + ZF equalisation.

    rx_data      : (N_DATA_SC, N_SYMBOLS)  — received symbols on data SCs
    tx_training  : (N_DATA_SC, N_TRAINING) — known training symbols

    Returns:
        rx_eq        : (N_DATA_SC, N_DATA_SYMS) — equalised data symbols
        H_est        : (N_DATA_SC,)             — estimated channel per SC
    """
    # ---- 1. Estimate H per subcarrier from training symbols ----
    r_training = rx_data[:, :N_TRAINING]               # (52, 2)
    H_est = np.mean(r_training / tx_training, axis=1)   # (52,)  — average over symbols

    # ---- 2. Equalise data symbols ----
    r_data = rx_data[:, N_TRAINING:]                    # (52, 198)
    rx_eq = r_data / H_est[:, np.newaxis]               # ZF equalisation

    return rx_eq, H_est


# ==================== EVM ====================
def compute_evm(syms_rx, syms_tx):
    """Error Vector Magnitude (%)."""
    err = syms_rx.ravel() - syms_tx.ravel()
    return np.sqrt(np.mean(np.abs(err)**2) / np.mean(np.abs(syms_tx.ravel())**2)) * 100


# ==================== Main ====================
def main():
    # ----- 1. Generate data -----
    n_bits_data = N_DATA_SC * N_DATA_SYMS * 2
    n_bits_train = N_DATA_SC * N_TRAINING * 2

    # Training bits (same for both TX — fixed pattern, known to RX)
    bits_train = np.random.randint(0, 2, n_bits_train)
    # Data bits (different for each TX)
    bits_data1 = np.random.randint(0, 2, n_bits_data)
    bits_data2 = np.random.randint(0, 2, n_bits_data)

    # ----- 2. QPSK modulation -----
    # Training symbols
    train_syms = qpsk_modulate(bits_train).reshape(N_DATA_SC, N_TRAINING)
    # Data symbols
    data_syms1 = qpsk_modulate(bits_data1).reshape(N_DATA_SC, N_DATA_SYMS)
    data_syms2 = qpsk_modulate(bits_data2).reshape(N_DATA_SC, N_DATA_SYMS)

    # Concatenate: training + data
    tx_all1 = np.hstack([train_syms, data_syms1])   # (52, 200)
    tx_all2 = np.hstack([train_syms, data_syms2])   # (52, 200)

    # ----- 3. OFDM modulation -----
    sig_tx1, _ = ofdm_modulate(tx_all1)
    sig_tx2, _ = ofdm_modulate(tx_all2)

    # ----- 4. Multipath channel -----
    sig_tx1_mp = apply_multipath(sig_tx1, *los_channel())
    sig_tx2_mp = apply_multipath(sig_tx2, *nlos_channel())

    # ----- 5. AWGN -----
    sig_tx1_mp = add_awgn(sig_tx1_mp, SNR_DB)
    sig_tx2_mp = add_awgn(sig_tx2_mp, SNR_DB)

    # Reshape to per-symbol view
    sym_len = N_FFT + N_CP
    syms_rx1 = sig_tx1_mp.reshape(N_SYMBOLS, sym_len)
    syms_rx2 = sig_tx2_mp.reshape(N_SYMBOLS, sym_len)

    # ----- 6. OFDM demodulation -----
    rx1_all, _ = ofdm_demodulate(syms_rx1)   # (52, 200)
    rx2_all, _ = ofdm_demodulate(syms_rx2)   # (52, 200)

    # ----- 7. Channel estimation + equalisation -----
    rx1_eq, H1_est = estimate_and_equalize(rx1_all, train_syms)
    rx2_eq, H2_est = estimate_and_equalize(rx2_all, train_syms)

    # Raw data symbols (no equalisation) — exclude training
    rx1_raw_data = rx1_all[:, N_TRAINING:]   # (52, 198)
    rx2_raw_data = rx2_all[:, N_TRAINING:]   # (52, 198)

    # ==================== Plotting ====================
    ref_pts = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"QPSK + OFDM  |  3-Path Multipath  |  SNR = {SNR_DB} dB  |  "
        f"{N_DATA_SC} SC × {N_DATA_SYMS} data symbols\n"
        f"TX1 → RX (LOS  /  3-path with dominant direct ray)    "
        f"TX2 → RX (NLOS  /  3-path, all reflected)",
        fontsize=12, fontweight="bold",
    )

    plot_specs = [
        (rx1_raw_data, rx1_eq, data_syms1, "TX1 → RX  (LOS)", axes[0, 0], axes[0, 1]),
        (rx2_raw_data, rx2_eq, data_syms2, "TX2 → RX  (NLOS)", axes[1, 0], axes[1, 1]),
    ]

    for rx_raw, rx_eq, tx_data, title, ax_raw, ax_eq in plot_specs:
        # ---- Raw received (no equalisation) ----
        ax_raw.scatter(rx_raw.real, rx_raw.imag,
                       s=1.5, alpha=0.5, color="steelblue", edgecolors="none")
        ax_raw.scatter(ref_pts.real, ref_pts.imag,
                       s=90, marker="X", color="darkred", zorder=5, label="TX reference")
        ax_raw.axhline(0, color="gray", lw=0.5, ls="--")
        ax_raw.axvline(0, color="gray", lw=0.5, ls="--")
        ax_raw.set_xlim(-1.8, 1.8)
        ax_raw.set_ylim(-1.8, 1.8)
        ax_raw.set_aspect("equal")
        ax_raw.set_title(f"{title}  —  Raw Received (No EQ)", fontsize=12)
        ax_raw.set_xlabel("In-phase (I)")
        ax_raw.set_ylabel("Quadrature (Q)")
        ax_raw.legend(loc="upper right", fontsize=8)
        ax_raw.grid(True, alpha=0.25)

        # ---- ZF-equalised ----
        ax_eq.scatter(rx_eq.real, rx_eq.imag,
                      s=1.5, alpha=0.5, color="seagreen", edgecolors="none")
        ax_eq.scatter(ref_pts.real, ref_pts.imag,
                      s=90, marker="X", color="darkred", zorder=5, label="TX reference")
        ax_eq.axhline(0, color="gray", lw=0.5, ls="--")
        ax_eq.axvline(0, color="gray", lw=0.5, ls="--")
        ax_eq.set_xlim(-1.8, 1.8)
        ax_eq.set_ylim(-1.8, 1.8)
        ax_eq.set_aspect("equal")
        ax_eq.set_title(f"{title}  —  ZF-Equalised (2 training symbols)", fontsize=12)
        ax_eq.set_xlabel("In-phase (I)")
        ax_eq.set_ylabel("Quadrature (Q)")
        ax_eq.legend(loc="upper right", fontsize=8)
        ax_eq.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out1 = "D:\\work\\wireless sensing\\wifi_csi\\constellation_los_nlos.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[1] Constellation diagram  →  {out1}")

    # ----- EVM -----
    evm_los  = compute_evm(rx1_eq, data_syms1)
    evm_nlos = compute_evm(rx2_eq, data_syms2)
    rms_los  = np.std(np.abs(H1_est))
    rms_nlos = np.std(np.abs(H2_est))
    print(f"    EVM        LOS = {evm_los:.2f}%   |  NLOS = {evm_nlos:.2f}%")
    print(f"    |H| std    LOS = {rms_los:.4f}    |  NLOS = {rms_nlos:.4f}")

    # ==================== Channel Frequency Response ====================
    fig2, (ax_fr, ax_ir) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, taps, delays, lbl, color in [
        (ax_fr, *los_channel(), "LOS", "steelblue"),
        (ax_fr, *nlos_channel(), "NLOS", "darkorange"),
    ]:
        # Build full 64-point impulse response
        h_td = np.zeros(N_FFT, dtype=complex)
        for t, d in zip(taps, delays):
            h_td[d] = t
        H_fd = np.fft.fft(h_td)
        sc_idx = np.arange(1, N_DATA_SC // 2 + 1)   # positive subcarriers 1..26
        H_sc = H_fd[sc_idx]                          # magnitude at positive data SCs
        ax.plot(sc_idx, np.abs(H_sc), "o-", ms=4, lw=1.2, color=color, label=lbl)
        ax.plot(sc_idx, np.abs(H_sc), "o-", ms=4, lw=1.2, color=color)

    ax_fr.axhline(1.0, color="gray", lw=0.5, ls="--", label="|H|=1 (flat)")
    ax_fr.set_title("Channel Magnitude Response |H(f)| on Data Subcarriers", fontsize=12)
    ax_fr.set_xlabel("Subcarrier index k")
    ax_fr.set_ylabel("|H[k]|")
    ax_fr.legend()
    ax_fr.grid(True, alpha=0.3)

    # Impulse response comparison
    colors_ir = {"LOS": "steelblue", "NLOS": "darkorange"}
    for lbl, ch_fn in [("LOS", los_channel), ("NLOS", nlos_channel)]:
        taps, delays = ch_fn()
        ax_ir.stem(delays, np.abs(taps), linefmt=colors_ir[lbl],
                   markerfmt="o", basefmt=" ", label=lbl)
    ax_ir.set_title("Channel Impulse Response (3 Taps)", fontsize=12)
    ax_ir.set_xlabel("Sample delay τ")
    ax_ir.set_ylabel("|h[τ]|")
    ax_ir.legend()
    ax_ir.grid(True, alpha=0.3)
    ax_ir.set_xlim(-1, N_CP)

    plt.tight_layout()
    out2 = "D:\\work\\wireless sensing\\wifi_csi\\channel_response.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[2] Channel response       →  {out2}")


if __name__ == "__main__":
    main()
