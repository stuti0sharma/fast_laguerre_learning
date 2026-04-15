import jax
import jax.numpy as jnp
import keras
from keras.models import Sequential
lay = keras.layers

class JAXL2Norm(lay.Layer):
    def __init__(self, epsilon=1e-8, **kw):
        super().__init__(**kw); self.epsilon = epsilon
    def call(self, x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + self.epsilon)
    def get_config(self):
        cfg = super().get_config(); cfg['epsilon'] = self.epsilon; return cfg


def phase_invariant_mse(y_true, y_pred):
    c_true = y_true[:, 0::2] + 1j*y_true[:, 1::2]
    c_pred = y_pred[:, 0::2] + 1j*y_pred[:, 1::2]
    
    dot   = jnp.sum(jnp.conj(c_pred) * c_true, axis=-1, keepdims=True)
    align = dot / (jnp.abs(dot) + 1e-8)
    align = jax.lax.stop_gradient(align)  
    
    diff  = c_true - c_pred * align
    return jnp.mean(jnp.abs(diff)**2)

def build_model(n_res, n_coeff):
    model = Sequential([
        lay.Input(shape=(n_res, n_res, 1)),
        lay.Conv2D(64,  3, activation="elu", padding="same"),
        lay.Conv2D(64,  3, activation="elu", padding="same"),
        lay.AvgPool2D((2, 2)),
        lay.Conv2D(128, 3, activation="elu", padding="same"),
        lay.Conv2D(128, 3, activation="elu", padding="same"),
        lay.AvgPool2D((2, 2)),
        lay.Conv2D(256, 3, activation="elu", padding="same"),
        lay.Conv2D(256, 3, activation="elu", padding="same"),
        lay.GlobalAveragePooling2D(),
        lay.Dense(256, activation="elu"),
        lay.Dropout(0.2),
        lay.Dense(2 * n_coeff),
        JAXL2Norm(name="coeffs"),
    ])
    return model