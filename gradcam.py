import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
import os
import tensorflow as tf

tf.config.optimizer.set_jit(False)


from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ---------------------------------------------------------------------------
# Per-model configuration — everything that differs between architectures
# lives here, so the rest of the script can stay generic.
# ---------------------------------------------------------------------------

target_layers = {
    "BaseCNN_mri_tumor": "conv2d_2",
    "resnet50_mri_tumor": "conv5_block3_out",
    "efficientNetB0_mri_tumor": "top_conv",
}

# ResNet50 needs external ImageNet-style preprocessing (RGB->BGR, mean subtraction).
# EfficientNetB0's saved model has its own internal Rescaling/Normalization layers,
# so it expects RAW 0-255 input — do NOT preprocess externally, or you'll double-process.
# BaseCNN was trained on simple /255.0 normalization, grayscale input.
PREPROCESS_FNS = {
    "resnet50_mri_tumor": resnet_preprocess,
    "efficientNetB0_mri_tumor": efficientnet_preprocess,          # identity — model handles it internally
    "BaseCNN_mri_tumor": lambda x: x ,
}

# Which cached dataset file each model reads from
DATA_FILES = {
    "resnet50_mri_tumor": "data/dataset_rgb.npz",
    "efficientNetB0_mri_tumor": "data/dataset_rgb.npz",
    "BaseCNN_mri_tumor": "data/dataset_rgb.npz",
}

CLASS_NAMES = ["notumor", "glioma", "meningioma", "pituitary"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_file(dir_path='models'):
    model_files = []
    for file in os.listdir(dir_path):
        if file.lower().endswith('.keras'):
            try:
                model_path = os.path.join(dir_path, file)
                model_files.append(model_path)
            except Exception as e:
                print("Error:", e)
    return model_files


def get_models():
    models_path = get_file()
    loaded_models = {}
    for path in models_path:
        model_name = os.path.basename(path).split('.')[0]
        model_obj = load_model(path)
        loaded_models[model_name] = model_obj
    return loaded_models


def prep_grad_model(model, model_name: str):
    target_layer_name = target_layers[model_name]

    if isinstance(model, tf.keras.Sequential):
       
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        x = inputs
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == target_layer_name:
                conv_output = x
        grad_model = Model(inputs=inputs, outputs=[conv_output, x])
    else:

        target_layer = model.get_layer(target_layer_name)
        grad_model = Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )

    return grad_model



# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

def get_gradcam_heatmap(grad_model, img_array, class_index):
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)

    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap


def resize_heatmap(heatmap, target_size=(224, 224)):
    return cv2.resize(heatmap, (target_size[1], target_size[0]))


def overlay_heatmap(heatmap, original_img_rgb, alpha=0.4):
    """original_img_rgb must be uint8, 3-channel, RGB order, same spatial size as heatmap."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)
    superimposed_bgr = cv2.addWeighted(heatmap_color_bgr, alpha, original_bgr, 1 - alpha, 0)
    return cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)


def to_display_rgb(raw_img):
    """Converts a raw stored image (grayscale or RGB) into a uint8 RGB array for display/overlay."""
    img = raw_img.astype(np.uint8)
    if img.ndim == 2 or img.shape[-1] == 1:
        img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
    return img


# ---------------------------------------------------------------------------
# Batched prediction — avoids loading the whole test set into GPU memory at once
# ---------------------------------------------------------------------------

def predict_in_batches(model, X, preprocess_fn, batch_size=8):
    all_preds = []
    for i in range(0, len(X), batch_size):
        batch = preprocess_fn(X[i:i + batch_size].copy())
        preds = model.predict(batch, verbose=0)
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Main: generate Grad-CAM overlays for every wrong prediction, for one model
# ---------------------------------------------------------------------------

def run_for_model(model_key, models, gradCam_models, batch_size=8, alpha=0.4,
                   out_dir='gradcam_outputs', max_examples=12, save=False):
    model = models[model_key]
    grad_model = gradCam_models[model_key]
    preprocess_fn = PREPROCESS_FNS[model_key]
    data_file = DATA_FILES[model_key]

    data = np.load(data_file)
    X_test, y_test = data['X_test'], data['y_test']

    predicted_classes = np.argmax(
        predict_in_batches(model, X_test, preprocess_fn, batch_size=batch_size), axis=1
    )

    all_wrong_indices = np.where(predicted_classes != y_test)[0]
    print(f"[{model_key}] {len(all_wrong_indices)} wrong predictions out of {len(y_test)} "
          f"(showing first {min(max_examples, len(all_wrong_indices))})")

    wrong_indices = all_wrong_indices[:max_examples]

    if save:
        model_out_dir = os.path.join(out_dir, model_key)
        os.makedirs(model_out_dir, exist_ok=True)

    for idx in wrong_indices:
        raw_img = to_display_rgb(X_test[idx])
        model_input = preprocess_fn(X_test[idx:idx + 1].copy())

        pred_class = predicted_classes[idx]
        true_class = y_test[idx]

        heatmap = get_gradcam_heatmap(grad_model, model_input, class_index=pred_class)
        heatmap_resized = resize_heatmap(heatmap, target_size=(224, 224))
        overlay = overlay_heatmap(heatmap_resized, raw_img, alpha=alpha)

        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original (true: {CLASS_NAMES[true_class]})",fontsize=8)
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Heatmap",fontsize=8)
        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {CLASS_NAMES[pred_class]}",fontsize=8)
        for ax in axes:
            ax.axis('off')


        if save:
            fname = os.path.join(
                model_out_dir,
                f"wrong_{idx}_true-{CLASS_NAMES[true_class]}_pred-{CLASS_NAMES[pred_class]}.png"
            )
            plt.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    if save:
        print(f"[{model_key}] Saved {len(wrong_indices)} of {len(all_wrong_indices)} "
              f"wrong-prediction Grad-CAM images to {model_out_dir}/")



def main():
    models = get_models()
    gradCam_models = {name: prep_grad_model(obj, name) for name, obj in models.items()}

    for model_key in models:
        run_for_model(model_key, models, gradCam_models, save=False)


# if __name__ == "__main__":
#     main()