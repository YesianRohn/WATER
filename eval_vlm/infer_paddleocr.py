"""PP-OCRv5 (recognition only) inference for WordArt-Bench evaluation."""

import os
import argparse
from tqdm import tqdm
from paddleocr import TextRecognition


def parse_args():
    p = argparse.ArgumentParser(description="PP-OCRv5 inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_ppocrv5.txt")
    p.add_argument("--model_name", type=str, default="PP-OCRv5_server_rec")
    return p.parse_args()


def main():
    args = parse_args()
    model = TextRecognition(model_name=args.model_name)

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            try:
                output = model.predict(input=path, batch_size=1)
                if output and hasattr(output[0], "res") and "rec_text" in output[0].res:
                    text = str(output[0].res["rec_text"])
                else:
                    text = "[No result]"
                text = text.replace("\n", " ").replace("\r", " ").strip()
            except Exception as e:
                print(f"\nError {name}: {e}")
                text = "[Error]"
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
