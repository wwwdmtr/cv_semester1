"""Применить сохранённый классификатор к одному повторению из 3D-поз EC3D."""

import argparse
import json
from pathlib import Path

import numpy as np

from ec3d_baseline import predict_saved_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path, help="Файл .npy формы (кадры, 25, 3)")
    parser.add_argument("--model", type=Path, default=Path("models/ec3d_baseline/squat_geometry.joblib"))
    args = parser.parse_args()
    result = predict_saved_model(args.model, np.load(args.poses, allow_pickle=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
