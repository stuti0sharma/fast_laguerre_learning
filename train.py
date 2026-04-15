import os
os.environ["KERAS_BACKEND"] = "jax"

from keras.optimizers import Adam
from data_utils import USE_H5_DATA, H5_FOLDER, MODES, N_RES, N_COEFF, generate_dataset, load_h5_dataset

from model import build_model, phase_invariant_mse

# Config
USE_H5_DATA = True
H5_FOLDER   = "/home/hpc/b129dc/b129dc30/simulated_dataset"
USE_CUSTOM_LOSS = True  # Set to False to use standard MSE

if __name__ == "__main__":
    # 1. Load Data
    if USE_H5_DATA:
        X_all, Y_all = load_h5_dataset(H5_FOLDER, MODES)
        split = int(len(X_all) * 0.9)
        X_train, Y_train = X_all[:split], Y_all[:split]
        X_val,   Y_val   = X_all[split:], Y_all[split:]
    else:
        X_train, Y_train = generate_dataset(100_000, N_RES)
        X_val,   Y_val   = generate_dataset(10_000,  N_RES)

    # 2. Build and Compile Model
    model = build_model(N_RES, N_COEFF)
    
    if USE_CUSTOM_LOSS:
        print("Compiling model with Custom Phase-Invariant MSE...")
        model.compile(optimizer=Adam(1e-4, amsgrad=True), loss=phase_invariant_mse)
        save_name = "trained_model_custom_loss.keras"
    else:
        print("Compiling model with Standard MSE...")
        model.compile(optimizer=Adam(1e-4, amsgrad=True), loss="mse")
        save_name = "trained_model_standard_mse.keras"

    model.summary()

    # 3. Train
    try:
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=50, batch_size=64,
        )
        # Save the trained weights using the specific name
        model.save(save_name)
        print(f"Training complete and model saved as {save_name}.")
    except KeyboardInterrupt:
        print("Training interrupted.")