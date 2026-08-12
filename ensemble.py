
import numpy as np
from gradcam import get_models, PREPROCESS_FNS, CLASS_NAMES,DATA_FILES


def predict_ensemble(image, models, preprocess_fns, strategy="average", weights=None):
    """
    image: single raw RGB image (before preprocessing), shape (H, W, 3)
    models: dict of {model_name: keras_model}
    preprocess_fns: dict of {model_name: preprocessing_function}
    strategy: "average" | "max_confidence" | "weighted"
    weights: dict of {model_name: float}, required if strategy == "weighted"
    """
    predictions = {}
    for name, model in models.items():
        x = preprocess_fns[name](np.expand_dims(image, axis=0).copy())
        probs = model.predict(x, verbose=0)[0]
        predictions[name] = probs

    if strategy == "average":
        avg_probs = np.mean(list(predictions.values()), axis=0)
        final_class = np.argmax(avg_probs)
        confidence = avg_probs[final_class]

    elif strategy == "max_confidence":
        best_model = max(predictions, key=lambda n: np.max(predictions[n]))
        final_class = np.argmax(predictions[best_model])
        confidence = np.max(predictions[best_model])

    elif strategy == "weighted":
        if weights is None:
            raise ValueError("weights dict required for 'weighted' strategy")
        weighted_probs = sum(predictions[n] * weights[n] for n in predictions)
        final_class = np.argmax(weighted_probs)
        confidence = weighted_probs[final_class]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return CLASS_NAMES[final_class], float(confidence), predictions

def main():
    models = get_models()
    preprocess_fns = {k: v for k, v in PREPROCESS_FNS.items() if k in models}
    data = np.load(DATA_FILES["resnet50_mri_tumor"])
    X_test, y_test = data['X_test'], data['y_test']

    idx = 71
    image = X_test[idx]
    true_class = CLASS_NAMES[y_test[idx]]

    pred_class, confidence, all_model_probs = predict_ensemble(
        image, models, preprocess_fns, strategy="average"
    )

    print(f"True class:      {true_class}")
    print(f"Predicted class: {pred_class} (confidence: {confidence:.3f})")
    print(f"Correct:         {pred_class == true_class}")
    print("Per-model probabilities:")
    for name, probs in all_model_probs.items():
        top = np.argmax(probs)
        print(f"  {name}: {CLASS_NAMES[top]} ({probs[top]:.3f})")

# if __name__ =='__main__':
#     main()