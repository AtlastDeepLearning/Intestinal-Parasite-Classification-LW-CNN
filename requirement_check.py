import sys

print("🔎 Checking environment...\n")

try:
    import numpy, pandas, matplotlib, sklearn, cv2
    print("✅ Core libraries installed")
except Exception as e:
    print("❌ Core libraries issue:", e)
    sys.exit(1)

try:
    import tensorflow as tf
    print("✅ TensorFlow version:", tf.__version__)
except Exception as e:
    print("❌ TensorFlow issue:", e)
    sys.exit(1)

try:
    import keras
    import efficientnet.tfkeras
    print("✅ Keras & EfficientNet installed")
except Exception as e:
    print("❌ Keras/EfficientNet issue:", e)
    sys.exit(1)

print("\n🎉 All checks passed! Ready to run your notebook.")
