"""GOT-OCR2.0 inference for WordArt-Bench evaluation."""

import os
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="GOT-OCR2.0 inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_gotocr2_0.txt")
    p.add_argument("--model", type=str, default="ucaslcl/GOT-OCR2_0")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, low_cpu_mem_usage=True,
        device_map="cuda", use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    ).eval().cuda()

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            res = model.chat(tokenizer, path, ocr_type="ocr")
            text = str(res).replace("\n", " ").replace("\r", " ").strip()
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
