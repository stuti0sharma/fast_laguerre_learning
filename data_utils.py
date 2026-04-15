import glob
import json
import os

import h5py
import numpy as np
from scipy.special import genlaguerre

ORDER = 2
N_RES = 64

def lg_mode_indices(max_order=4):
    modes = []
    for p in range(max_order + 1):
        for l in range(-max_order, max_order + 1):
            if p + abs(l) <= max_order: 
                modes.append((p, l))
    return modes

MODES = lg_mode_indices(ORDER)
N_COEFF = len(MODES)

def make_grid(N=32):
    x = np.linspace(-2, 2, N); y = np.linspace(-2, 2, N)
    X, Y = np.meshgrid(x, y)
    return np.sqrt(X**2+Y**2), np.arctan2(Y, X)

def lg_mode(p, l, r, theta):
    l_abs = abs(l)
    L   = genlaguerre(p, l_abs)(2*r**2)
    amp = ((r * np.sqrt(2))**l_abs) * np.exp(-(r**2)) * L
    mode = amp * np.exp(1j*l*theta)
    return mode / np.sqrt(np.sum(np.abs(mode)**2))

def synthesize_phase(coeffs, n_res=32):
    r, theta = make_grid(n_res)
    E = np.zeros_like(r, dtype=np.complex128)
    for c, (p, l) in zip(coeffs, MODES):
        E += c * lg_mode(p, l, r, theta)
    return np.angle(E)

def generate_dataset(N_samples=5000, n_res=32):
    coeffs = np.random.randn(N_samples, N_COEFF) + 1j*np.random.randn(N_samples, N_COEFF)
    coeffs /= np.sqrt(np.sum(np.abs(coeffs)**2, axis=1, keepdims=True))
    Y = np.zeros((N_samples, 2*N_COEFF), dtype=np.float32)
    Y[:, 0::2] = coeffs.real;  Y[:, 1::2] = coeffs.imag
    X = np.zeros((N_samples, n_res, n_res, 1), dtype=np.float32)
    for i in range(N_samples):
        X[i, ..., 0] = synthesize_phase(coeffs[i], n_res)
    return X, Y

def load_h5_dataset(folder_path, modes_list):
    mode_to_idx = {f"p{p}l{l}": i for i, (p, l) in enumerate(modes_list)}
    X_list, Y_list = [], []
    for fp in glob.glob(os.path.join(folder_path, "*.h5")):
        with h5py.File(fp, 'r') as f:
            X_list.append(f['phase'][:][..., np.newaxis])
            coeffs_c = np.zeros(len(modes_list), dtype=complex)
            mixing   = json.loads(f.attrs.get('mixing_coefficients_json', '{}'))
            for md in mixing.values():
                if md['mode'] in mode_to_idx:
                    idx = mode_to_idx[md['mode']]
                    coeffs_c[idx] += md['amplitude'] * np.exp(1j * md['phase_rad'])
            norm = np.sqrt(np.sum(np.abs(coeffs_c)**2))
            if norm > 0:
                coeffs_c /= norm

            y = np.zeros(2 * len(modes_list), dtype=np.float32)
            y[0::2] = coeffs_c.real
            y[1::2] = coeffs_c.imag
            Y_list.append(y)
            
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    print(f"Loaded {len(X)} samples — X:{X.shape}  Y:{Y.shape}")
    return X, Y