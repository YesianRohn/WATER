"""HunyuanOCR inference for WordArt-Bench evaluation."""

import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

PROMPT = "Extract the text from this image."


def parse_args():
    p = argparse.ArgumentParser(description="HunyuanOCR inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_hunyuanocr.txt")
    p.add_argument("--model", type=str, default="tencent/HunyuanOCR")
    return p.parse_args()


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model, attn_implementation="eager",
        dtype=torch.bfloat16, device_map="auto",
    )

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            try:
                image = Image.open(path).convert("RGB")
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": [
                        {"type": "image", "image": path},
                        {"type": "text", "text": PROMPT}]},
                ]
                texts = [processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)]
                inputs = processor(text=texts, images=image,
                                   padding=True, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
                input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
                trimmed = [o[len(i):] for i, o in zip(input_ids, ids)]
                text = processor.batch_decode(
                    trimmed, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip().replace("\n", " ").replace("\r", " ")
            except Exception as e:
                print(f"\nFailed {name}: {e}")
                text = "[Error]"
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
