"""PaddleOCR-VL inference for WordArt-Bench evaluation."""

import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "OCR:"


def parse_args():
    p = argparse.ArgumentParser(description="PaddleOCR-VL inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_paddleocr_vl.txt")
    p.add_argument("--model", type=str, default="PaddlePaddle/PaddleOCR-VL")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            try:
                image = Image.open(path).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT}]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=1024)
                text = processor.batch_decode(out, skip_special_tokens=True
                    )[0].strip().replace("\n", " ").replace("\r", " ")
            except Exception as e:
                print(f"\nFailed {name}: {e}")
                text = "[Error]"
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
