"""DeepSeek-OCR-2 inference for WordArt-Bench evaluation."""

import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROMPT = "<image>\nFree OCR. "


def parse_args():
    p = argparse.ArgumentParser(description="DeepSeek-OCR-2 inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_deepseek_ocr2.txt")
    p.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-OCR-2")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, _attn_implementation="flash_attention_2",
        trust_remote_code=True, use_safetensors=True,
    ).eval().cuda().to(torch.bfloat16)

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            try:
                model.infer(tokenizer, prompt=PROMPT, image_file=path,
                            output_path="./deepseek_tmp", base_size=1024,
                            image_size=640, crop_mode=True,
                            save_results=True, test_compress=True)
                with open("./deepseek_tmp/result.mmd", "r", encoding="utf-8") as f:
                    text = f.read().replace("\n", " ").replace("\r", " ").strip()
            except Exception:
                text = "None"
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
