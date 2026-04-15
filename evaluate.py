import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from matplotlib import pyplot as plt
from data_utils import USE_H5_DATA, H5_FOLDER, MODES, N_RES, synthesize_phase, load_h5_dataset, generate_dataset
from model import JAXL2Norm, phase_invariant_mse

def phase_invariant_fidelity(c_true, c_pred):
    dot = np.sum(np.conj(c_pred) * c_true)
    return float(np.abs(dot)**2)

def align_global_phase(c_true, c_pred):
    dot = np.sum(np.conj(c_pred) * c_true)
    align = dot / (np.abs(dot) + 1e-8)
    return c_pred * align

if __name__ == "__main__":
    # 1. Load Validation Data
    if USE_H5_DATA:
        X_all, Y_all = load_h5_dataset(H5_FOLDER, MODES)
        split = int(len(X_all) * 0.9)
        X_val, Y_val = X_all[split:], Y_all[split:]
    else:
        _, _ = generate_dataset(100, N_RES) # throwaway
        X_val, Y_val = generate_dataset(10_000, N_RES)

    MODEL_TO_EVALUATE = "trained_model_custom_loss.keras" 
    
    model = keras.saving.load_model(
        MODEL_TO_EVALUATE, 
        custom_objects={
            "JAXL2Norm": JAXL2Norm, 
            "phase_invariant_mse": phase_invariant_mse
        }
    )
    
    coeffs_pred = model.predict(X_val)

    # 3. Plot Dashboard
    os.makedirs("output_images", exist_ok=True)
    mode_labels = [f"p{p},l{l}" for p, l in MODES]
    x_pos = np.arange(len(MODES))
    width = 0.35
    AMP_THRESHOLD = 0.05

    for i in range(4):
        c_true = Y_val[i, 0::2] + 1j * Y_val[i, 1::2]
        c_pred_raw = coeffs_pred[i, 0::2] + 1j * coeffs_pred[i, 1::2]
        c_pred = align_global_phase(c_true, c_pred_raw)
        
        phase_input = X_val[i, ..., 0]
        phase_true  = synthesize_phase(c_true, N_RES)
        phase_pred  = synthesize_phase(c_pred, N_RES)
        fidelity    = phase_invariant_fidelity(c_true, c_pred)

        fig = plt.figure(figsize=(20, 11))
        kw  = dict(cmap='twilight', vmin=-np.pi, vmax=np.pi)

        # Plot 1: Images
        for col, (data, title) in enumerate([(phase_input, "Input"), (phase_true, "TRUE"), (phase_pred, "PREDICTED")], start=1):
            ax = plt.subplot(2, 3, col)
            im = ax.imshow(data, **kw)
            plt.colorbar(im, ax=ax, fraction=0.046).set_label('rad')
            ax.set_title(title, fontsize=18)

        # Plot 2: Amplitude Bars
        ax_amp = plt.subplot(2, 2, 3)
        ax_amp.bar(x_pos - width/2, np.abs(c_true), width, label='True', color='navy')
        ax_amp.bar(x_pos + width/2, np.abs(c_pred), width, label='Predicted', color='darkorange')
        ax_amp.set_xticks(x_pos); ax_amp.set_xticklabels(mode_labels, rotation=45)
        ax_amp.legend()

        # Plot 3: Phase Bars
        ax_ph = plt.subplot(2, 2, 4)
        mask = np.abs(c_true) >= AMP_THRESHOLD
        ax_ph.bar(x_pos - width/2, np.where(mask, np.angle(c_true), np.nan), width, label='True Phase', color='navy')
        ax_ph.bar(x_pos + width/2, np.where(mask, np.angle(c_pred), np.nan), width, label='Predicted Phase', color='darkorange')
        ax_ph.set_xticks(x_pos); ax_ph.set_xticklabels(mode_labels, rotation=45)
        ax_ph.legend()

        plt.suptitle(f"Sample {i} — Fidelity {fidelity:.3f}", fontsize=22)
        plt.tight_layout()
        plt.savefig(f"output_images/Dashboard_{i}.png", dpi=150)
        plt.close(fig)

    print("Dashboards saved to /output_images")