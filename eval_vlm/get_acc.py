"""
Accuracy evaluation for VLM / STR predictions on WordArt-Bench.

Label and prediction files share the same format:
    image_name text_label
(one entry per line, separated by whitespace)

Usage:
    python get_acc.py --label test_label.txt --pred predictions.txt
"""

import argparse
import string


def _normalize(text):
    return "".join(ch for ch in text if ch in string.digits + string.ascii_letters)


def evaluate_acc(label_path, pred_path):
    label_dict = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            label_dict[parts[0]] = _normalize(parts[1].lower())

    pred_dict = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            pred_dict[parts[0]] = _normalize(parts[1].lower())

    correct = sum(1 for k, v in label_dict.items() if pred_dict.get(k, "") == v)
    total = len(label_dict)
    acc = correct / total if total else 0.0
    print(f"ACC: {acc:.4f}  ({correct}/{total})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()
    evaluate_acc(args.label, args.pred)
