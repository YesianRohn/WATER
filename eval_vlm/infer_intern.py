"""InternVL3.5-8B inference for WordArt-Bench evaluation via lmdeploy."""

import os
import argparse
from tqdm import tqdm
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image

PROMPT = (
    "Please directly output all original readable text in the image. "
    "Do not include any explanation or description. Only output the recognized text."
)


def parse_args():
    p = argparse.ArgumentParser(description="InternVL3.5-8B inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_internvl3_5_8b.txt")
    p.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-8B")
    return p.parse_args()


def main():
    args = parse_args()
    pipe = pipeline(args.model,
                    backend_config=PytorchEngineConfig(session_len=32768, tp=1))
    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            try:
                image = load_image(path)
                resp = pipe((PROMPT, image))
                text = resp.text.strip().replace("\n", " ").replace("\r", " ")
            except Exception as e:
                print(f"\nSkip {name}: {e}")
                text = "[Error]"
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
