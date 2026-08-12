import os
import numpy as np
import kagglehub
from PIL import Image


IMG_SIZE = 224
cnt = 1400  # cap per class — see note in load_img() re: class balance


BASE_CACHE = "data/dataset_gray.npz"
RGB_CACHE = "data/dataset_rgb.npz"

def download_db():
    """Downloads dataset via kagglehub and returns the local path.
    NOTE: kagglehub caches downloads, so calling this repeatedly is cheap
    once the dataset is already present locally.
    """
    path = kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset')
    print(f"Data source import complete. Path: {path}")
    return path


def files_in_dir(base_path):
    for current_dir, dirs, files in os.walk(base_path):
        num_images = sum(
            f.lower().endswith((".jpg", ".jpeg", ".png"))
            for f in files
        )
        if num_images > 0:
            print(f"{os.path.basename(current_dir)}: {num_images}")


def load_img(dir_path, label, model=None):
    """Loads images from `dir_path`, resizes, and applies ResNet50's
    expected preprocessing (NOT naive /255.0 rescaling — pretrained
    ImageNet backbones expect their own specific normalization).
    """
    images = []
    labels = []
    count = 0

    for file in os.listdir(dir_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(dir_path, file)
                if model == 'Base-CNN':
                    img = Image.open(img_path).convert('L')
                else:
                    img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img).astype(np.float32)

                images.append(img)
                labels.append(label)

                count += 1
                if count == cnt:
                    break

            except Exception as e:
                print("Error:", e)

    return images, labels


def _class_dirs(base):
    return {
        "Training": {
            "notumor": f"{base}/Training/notumor",
            "glioma": f"{base}/Training/glioma",
            "meningioma": f"{base}/Training/meningioma",
            "pituitary": f"{base}/Training/pituitary",
        },
        "Testing": {
            "notumor": f"{base}/Testing/notumor",
            "glioma": f"{base}/Testing/glioma",
            "meningioma": f"{base}/Testing/meningioma",
            "pituitary": f"{base}/Testing/pituitary",
        },
    }



CLASS_NAMES = ["notumor", "glioma", "meningioma", "pituitary"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _load_split(dirs_for_split,model = None):
    X_parts, y_parts = [], []
    for class_name, dir_path in dirs_for_split.items():
        imgs, labels = load_img(dir_path, LABEL_MAP[class_name], model)
        X_parts.extend(imgs)
        y_parts.extend(labels)
    return np.array(X_parts), np.array(y_parts)


def load_dataset(model=None):
    if model == 'Base-CNN':
        cache_file = BASE_CACHE      # grayscale
    else:
        cache_file = RGB_CACHE       # RGB for pretrained models

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        data = np.load(cache_file)
        return (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
        )

    print("Preparing dataset...")
    base = download_db()
    dirs = _class_dirs(base)

    X_train, y_train = _load_split(dirs["Training"], model)
    X_test, y_test = _load_split(dirs["Testing"], model)

    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    return X_train, y_train, X_test, y_test
