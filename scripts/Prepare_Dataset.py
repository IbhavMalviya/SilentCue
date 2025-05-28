import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Paths
DATA_DIR = "data"
IMG_SIZE = 64  # Resize to 64x64
labels = ['yes', 'no', 'stop', 'play']

data = []
targets = []

for label_idx, label in enumerate(labels):
    folder_path = os.path.join(DATA_DIR, label)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize
            data.append(img)
            targets.append(label_idx)
            
data = np.array(data)
targets = np.array(targets)


# One-hot encode labels
targets = tf.keras.utils.to_categorical(targets, num_classes=len(labels))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data, targets, test_size=0.2, random_state=42, stratify=targets)
    
# Save the processed data (optional)    
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print(f"[✔] Dataset prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")