import json
from gradcam import DATA_FILES,PREPROCESS_FNS,get_models,prep_grad_model
from evaluate import get_all_losses,save_all_to_json
import numpy as np
import tensorflow as tf

def compute_and_save_losses():
    models = get_models()

    data = np.load(DATA_FILES["efficientNetB0_mri_tumor"])
    X_test_eff, y_test_eff = data['X_test'], data['y_test']
    preprocess_fn_eff = PREPROCESS_FNS["efficientNetB0_mri_tumor"]
    model_eff = models["efficientNetB0_mri_tumor"]
    eff_preds, eff_losses = get_all_losses(model_eff, X_test_eff, y_test_eff, preprocess_fn_eff)
    save_all_to_json(eff_preds, eff_losses, y_test_eff, "all_predictions_efficientNetB0.json")

    data = np.load(DATA_FILES["resnet50_mri_tumor"])
    X_test_res, y_test_res = data['X_test'], data['y_test']
    preprocess_fn_res = PREPROCESS_FNS["resnet50_mri_tumor"]
    model_res = models["resnet50_mri_tumor"]
    res_preds, res_losses = get_all_losses(model_res, X_test_res, y_test_res, preprocess_fn_res)
    save_all_to_json(res_preds, res_losses, y_test_res, "all_predictions_resnet50.json")

    assert np.array_equal(y_test_eff, y_test_res), "Test sets are not aligned by index!"


def compare_divergent():
    with open("all_predictions_efficientNetB0.json") as f:
        eff_all = {item["index"]: item for item in json.load(f)}
    with open("all_predictions_resnet50.json") as f:
        res_all = {item["index"]: item for item in json.load(f)}

    interesting = []
    for idx in eff_all:
        e, r = eff_all[idx], res_all[idx]
        if e["correct"] != r["correct"]:
            interesting.append({
                "index": idx,
                "true_class": e["true_class"],
                "efficientnet_pred": e["predicted_class"],
                "efficientnet_correct": e["correct"],
                "efficientnet_loss": e["loss"],
                "resnet50_pred": r["predicted_class"],
                "resnet50_correct": r["correct"],
                "resnet50_loss": r["loss"],
            })

    with open("comparison_divergent.json", "w") as f:
        json.dump(interesting, f, indent=2)
    print(f"{len(interesting)} indices where models disagree on correctness")


if __name__ == '__main__':
    compute_and_save_losses()
    compare_divergent()