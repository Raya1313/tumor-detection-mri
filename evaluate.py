"""
Evaluation utilities for the MRI tumor classification paper.
Drop into your training script after `resnet50.fit(...)` calls.

Usage:
    from evaluate import evaluate_model, plot_training_curves

    evaluate_model(resnet50, X_test, y_test, class_names=CLASS_NAMES)
    plot_training_curves(history, history_1)
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    precision_score
)

from data.data import CLASS_NAMES

def evaluate_model(model, X_test, y_test, class_names, save_prefix: str):
    """Runs predictions and prints/saves the metrics you need for the
    Results section: per-class precision/recall/F1, confusion matrix,
    and macro-averaged AUC (multi-class, one-vs-rest).
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- Classification report (precision/recall/F1 per class) ---
    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    print("Classification Report:\n", report)

    with open(f"{save_prefix}_classification_report.txt", "w") as f:
        f.write(report)

    # --- Confusion matrix (raw counts + normalized) ---
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=300)
    plt.show()

    # --- Macro AUC (multi-class one-vs-rest) — good extra metric for the paper ---
    try:
        auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="macro")
        print(f"Macro-averaged AUC (OvR): {auc:.4f}")
    except ValueError as e:
        print("AUC could not be computed:", e)
        auc = None

    return {"y_pred": y_pred, "y_pred_probs": y_pred_probs, "cm": cm, "auc": auc}


def evaluate_per_class(model, X_val, X_test, y_val, y_test, classes):
    y_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    y_pred=np.argmax(y_pred, axis=1)
    test_pred=np.argmax(test_pred, axis=1)
    # per-class precision only
    prec_val = precision_score(y_val, y_pred, average=None)
    prec_test = precision_score(y_test, test_pred, average=None)

    for i, c in enumerate(classes):
        print(f"{c} - val precision: {prec_val[i]:.4f}, test precision: {prec_test[i]:.4f}")


def plot_training_curves(history, history_finetune=None, save_prefix="resnet50"):
    """Plots accuracy/loss curves. If history_finetune is given (Phase 2),
    stitches both phases together on one timeline with a marker at the
    phase transition — this is exactly what you want as a figure showing
    training behavior across the frozen -> fine-tuned transition.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    phase1_epochs = len(acc)

    if history_finetune is not None:
        acc += history_finetune.history["accuracy"]
        val_acc += history_finetune.history["val_accuracy"]
        loss += history_finetune.history["loss"]
        val_loss += history_finetune.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, acc, label="Train Accuracy")
    axes[0].plot(epochs_range, val_acc, label="Val Accuracy")
    if history_finetune is not None:
        axes[0].axvline(x=phase1_epochs, color="gray", linestyle="--",
                        label="Fine-tuning starts")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs_range, loss, label="Train Loss")
    axes[1].plot(epochs_range, val_loss, label="Val Loss")
    if history_finetune is not None:
        axes[1].axvline(x=phase1_epochs, color="gray", linestyle="--",
                        label="Fine-tuning starts")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_training_curves.png", dpi=300)
    plt.show()


def plot_false_preds(model, X, y, model_type):
    """
    model_type: 'base_cnn', 'resnet50', or 'efficientnet'
    X: array already in the format fed to model.predict
       - base_cnn: grayscale, normalized (or raw uint8, handled below)
       - resnet50: preprocessed with resnet's preprocess_input
       - efficientnet: preprocessed with efficientnet's preprocess_input
    """
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    wrong_idx = np.where(y_pred != y)[0]
    n = min(12, len(wrong_idx))
    if n == 0:
        print("No misclassifications found!")
        return

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols)
    axes = np.array(axes).reshape(-1)

    for ax, i in zip(axes, wrong_idx[:n]):
        img = X[i].copy()

        if model_type == 'resnet50':
            img[..., 0] += 103.939
            img[..., 1] += 116.779
            img[..., 2] += 123.68
            img = img[..., ::-1]  # BGR -> RGB
            img = np.clip(img, 0, 255) / 255.0
            ax.imshow(img)


        else:   
            img = img.squeeze()
            if img.max() > 1.0:
                img = img / 255.0
            ax.imshow(img, cmap='gray')


        ax.set_title(f"True: {y[i]}, Pred: {y_pred[i]}", fontsize=9)

    for ax in axes[n:]:
        ax.axis('off')
    for ax in axes[:n]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

from gradcam import predict_in_batches
import tensorflow as tf
import json

def get_all_losses(model, X, y, preprocess_fn, batch_size=8):
    y_pred = predict_in_batches(model, X, preprocess_fn, batch_size=batch_size)
    predicted_classes = np.argmax(y_pred, axis=1)
    y_true_onehot = tf.one_hot(y, depth=y_pred.shape[-1])
    losses = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred).numpy()
    return predicted_classes, losses

def save_all_to_json(predicted_classes, losses, y_test, out_path):
    results = [
        {
            "index": int(i),
            "true_class": CLASS_NAMES[y_test[i]],
            "predicted_class": CLASS_NAMES[predicted_classes[i]],
            "correct": bool(predicted_classes[i] == y_test[i]),
            "loss": float(losses[i])
        }
        for i in range(len(y_test))
    ]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

def save_wrong_losses_to_json(wrong_indices, wrong_losses, wrong_preds, y_test, model_key, out_path):
    
    import json
    results = []
    for idx, loss, pred in zip(wrong_indices, wrong_losses, wrong_preds):
        results.append({
            "index": int(idx),
            "true_class": CLASS_NAMES[y_test[idx]],
            "predicted_class": CLASS_NAMES[pred],
            "loss": float(loss)
        })

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} wrong predictions to {out_path}")

def get_wrong_prediction_losses(model, X, y, preprocess_fn, batch_size=8):
    y_pred = predict_in_batches(model, X, preprocess_fn, batch_size=batch_size)  # (N, num_classes)
    predicted_classes = np.argmax(y_pred, axis=1)

    wrong_indices = np.where(predicted_classes != y)[0]

    y_true_onehot = tf.one_hot(y[wrong_indices], depth=y_pred.shape[-1])
    wrong_losses = tf.keras.losses.categorical_crossentropy(
        y_true_onehot, y_pred[wrong_indices]
    ).numpy()

    return wrong_indices, wrong_losses, predicted_classes[wrong_indices]