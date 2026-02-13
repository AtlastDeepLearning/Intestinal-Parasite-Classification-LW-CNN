import h5py
import tensorflow as tf
import os
import sys

MODEL_PATH = "models/efficientnet_parasite_final.h5"

def inspect_h5_keys(f, indent=0):
    for key in f.keys():
        print("  " * indent + f"- {key} ({type(f[key]).__name__})")
        if isinstance(f[key], h5py.Group) and indent < 2:
            inspect_h5_keys(f[key], indent + 1)

def main():
    print(f"--- Inspecting {MODEL_PATH} ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ File not found: {MODEL_PATH}")
        sys.exit(1)

    # 1. Try generic H5 inspection
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            print("✅ Valid HDF5 file.")
            print("Keys in root:")
            inspect_h5_keys(f)
            
            # Check for Keras specific attributes
            if 'model_config' in f.attrs:
                print("\n✅ Found 'model_config' attribute (Full Keras Model)")
            elif 'layer_names' in f.attrs:
                 print("\n✅ Found 'layer_names' attribute (Keras Weights)")
            else:
                 print("\n⚠️ No Keras-specific attributes found in root.")

    except OSError:
        print("❌ Not a valid HDF5 file (Might be PyTorch .pth renamed?)")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading HDF5: {e}")

    # 2. Try loading with Keras
    print("\n--- Attempting Keras Load ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Successfully loaded as full Keras model.")
        model.summary()
    except Exception as e:
        print(f"❌ tf.keras.models.load_model failed: {e}")

if __name__ == "__main__":
    main()
