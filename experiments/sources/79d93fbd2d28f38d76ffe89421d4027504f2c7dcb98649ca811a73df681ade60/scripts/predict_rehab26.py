"""Применить сохранённую REHAB26-модель к одному подготовленному 3D-повтору."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from rehab26_features import extract_features, FEATURE_NAMES, FEATURE_VERSION


MESSAGES = {
    0: "Не обнаружены признаки двух проверяемых ошибок; это не полная оценка всей техники.",
    1: "Предсказана категория сведения коленей внутрь по разметке REHAB26.",
    2: "Предсказана категория недостаточной глубины по разметке REHAB26.",
}


def predict(model_path, poses):
    bundle = joblib.load(model_path)
    if (bundle.get("format_version") != 1 or bundle.get("feature_version") != FEATURE_VERSION
            or bundle.get("feature_names") != FEATURE_NAMES):
        raise ValueError("Модель не соответствует версии геометрии MediaPipe REHAB26")
    pipeline = bundle["pipeline"]
    if list(pipeline.classes_) != bundle["classes"]:
        raise ValueError("Классы классификатора не соответствуют метаданным")
    x = extract_features(poses)[None]
    label = int(pipeline.predict(x)[0])
    if hasattr(pipeline, "predict_proba"):
        score = float(pipeline.predict_proba(x).max())
        score_kind = "max_predict_proba_uncalibrated"
    else:
        values = np.sort(pipeline.decision_function(x)[0])
        score = float(values[-1] - values[-2])
        score_kind = "ovr_decision_margin_uncalibrated"
    return {
        "predicted_class": label, "category": bundle["label_names"][label],
        "message": MESSAGES[label], "score": score, "score_kind": score_kind,
        "confidence_calibrated": False, "abstention_configured": False,
        "model_scope": bundle["scope"], "feature_version": FEATURE_VERSION,
        "input": "one complete MediaPipe 3D repetition; RGB and EC3D arrays are not interchangeable inputs",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path, help="NPY: (кадры, 33, 4), world xyz + visibility")
    parser.add_argument("--model", type=Path, default=Path("models/rehab26_baseline/squat_geometry.joblib"))
    args = parser.parse_args()
    result = predict(args.model, np.load(args.poses, allow_pickle=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
