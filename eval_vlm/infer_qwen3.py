"""
Inference scripts for evaluating VLMs on WordArt-Bench.

Each script loads a specific VLM, runs inference on all images in a directory,
and saves results as: image_name recognized_text (one per line).

Usage:
    python infer_qwen3.py --image_dir test_image --output results.txt
"""

import os
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

PROMPT = (
    "Please directly output all original readable text in the image. "
    "Do not include any explanation or description. Only output the recognized text."
)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-VL-8B inference")
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, default="test_qwen3_8b.txt")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    return p.parse_args()


def main():
    args = parse_args()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    names = sorted(f for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))

    with open(args.output, "w", encoding="utf-8") as fout:
        for name in tqdm(names):
            path = os.path.join(args.image_dir, name)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": path}, {"type": "text", "text": PROMPT}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            ids = model.generate(**inputs, max_new_tokens=128, top_p=0.8,
                                top_k=20, temperature=0.7, do_sample=True)
            text = processor.batch_decode(
                ids[:, len(inputs.input_ids[0]):],
                skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].replace("\n", " ").replace("\r", " ").strip()
            fout.write(f"{name} {text}\n")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
