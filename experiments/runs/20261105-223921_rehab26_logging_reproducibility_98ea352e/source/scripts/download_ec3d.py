"""Скачать проверенную версию EC3D из репозитория авторов."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

COMMIT = "1c8d28ddc8a346634f9c86335e3a261f90d6b9b2"
URL = (
    "https://raw.githubusercontent.com/Jacoo-Zhao/"
    "3D-Pose-Based-Feedback-For-Physical-Exercises/"
    f"{COMMIT}/data/EC3D/data_3D.pickle"
)
SHA256 = "a05c8b4f55f2f43637d2d6cbf40722807ea780f47e84090470bd100c927ebbdf"


def verify(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SHA256:
        raise ValueError("Контрольная сумма EC3D не совпадает с проверенной версией")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/ec3d/data_3D.pickle"))
    args = parser.parse_args()
    if args.output.exists():
        verify(args.output)
        print(f"Файл уже получен и проверен: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        verify(partial)
        partial.replace(args.output)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Скачан и проверен: {args.output}")


if __name__ == "__main__":
    main()
