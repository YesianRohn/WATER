"""
Extract wrong cases from VLM / STR predictions for error analysis.

Prediction file format (with optional confidence score):
    image_name predicted_text [score]

Usage:
    python get_wrong.py --label test_label.txt --pred predictions.txt --output wrong_cases.txt
"""

import argparse
import string


def _normalize(text):
    return "".join(ch for ch in text if ch in string.digits + string.ascii_letters)


def evaluate_and_save_wrong(label_path, pred_path, wrong_path="wrong_cases.txt"):
    label_dict, raw_label_dict = {}, {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            raw_label_dict[parts[0]] = parts[1]
            label_dict[parts[0]] = _normalize(parts[1].lower())

    pred_dict, raw_pred_dict = {}, {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = parts[0]
            if len(parts) >= 3:
                score, pred_raw = parts[-1], " ".join(parts[1:-1])
            else:
                score, pred_raw = "", parts[1]
            pred_dict[name] = _normalize(pred_raw.lower())
            raw_pred_dict[name] = (pred_raw, score)

    correct, total, wrong_lines = 0, 0, []
    for name, label_norm in label_dict.items():
        pred_norm = pred_dict.get(name, "")
        pred_raw, score = raw_pred_dict.get(name, ("", ""))
        gt_raw = raw_label_dict.get(name, "")
        if label_norm == pred_norm:
            correct += 1
        elif all(ch in string.printable for ch in gt_raw):
            wrong_lines.append(f"{name}\tGT:{gt_raw}\tPRED:{pred_raw}\tSCORE:{score}")
        total += 1

    acc = correct / total if total else 0.0
    print(f"ACC: {acc:.4f}  ({correct}/{total})")

    if wrong_lines:
        with open(wrong_path, "w", encoding="utf-8") as f:
            f.write("\n".join(wrong_lines) + "\n")
        print(f"Wrong cases ({len(wrong_lines)}) saved to: {wrong_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--output", type=str, default="wrong_cases.txt")
    args = parser.parse_args()
    evaluate_and_save_wrong(args.label, args.pred, args.output)
