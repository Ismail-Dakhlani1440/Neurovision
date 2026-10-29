import os
import numpy as np
import cv2

DATA_DIR = "/kaggle/input/datasets/ismaildakhlani/neuroai/Data"
IMG_SIZE = 224
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VALID_EXT = {".jpeg", ".jpg", ".bmp", ".png"}

def scan_dataset(data_dir=DATA_DIR, classes=CLASSES, valid_ext=VALID_EXT):
    kept, rejected = [], []

    for label in classes:
        folder = os.path.join(data_dir, label) 

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.splitext(filename)[1].lower() not in valid_ext:
                rejected.append((path, "bad extension"))
                continue

            try:
                img = cv2.imread(path)
                if img is None:
                    rejected.append((path, "unreadable"))
                else:
                    kept.append((path, label))
            except Exception as e:
                rejected.append((path, str(e)))
    return kept, rejected


def load_image(path, size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_dataset(kept, classes=CLASSES, size=IMG_SIZE, seed=42):
    X = np.empty((len(kept), size, size, 3), dtype=np.uint8)
    y = np.empty(len(kept), dtype=np.int64)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for i, (path, label) in enumerate(kept):
        X[i] = load_image(path, size)
        y[i] = class_to_idx[label]

    idx = np.random.default_rng(seed).permutation(len(X))
    return X[idx], y[idx]
