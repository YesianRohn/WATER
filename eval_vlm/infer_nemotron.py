"""Nemotron-Nano-VL-8B inference for WordArt-Bench evaluation."""

import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

PROMPT = (
    "Please directly output all original readable text in the image. "
    "Do not include any explanation or description. Only output the recognized text."
)


def parse_args():
    p = argparse.ArgumentParser(description="Nemotron-VL-8B inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_nemotron_vl_8b.txt")
    p.add_argument("--model", type=str, default="nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1")
    return p.parse_args()


def main():
    args = parse_args()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                       device_map="cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    image_processor = AutoImageProcessor.from_pretrained(
        args.model, trust_remote_code=True, device="cuda")

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            try:
                image = Image.open(path).convert("RGB")
                feats = image_processor([image])
                with torch.no_grad():
                    resp = model.chat(tokenizer=tokenizer, question=PROMPT,
                                      generation_config=dict(
                                          max_new_tokens=256, do_sample=False,
                                          eos_token_id=tokenizer.eos_token_id),
                                      **feats)
                text = resp.replace("\n", " ").replace("\r", " ").strip()
            except Exception as e:
                print(f"\nFailed {name}: {e}")
                text = "[Error]"
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
