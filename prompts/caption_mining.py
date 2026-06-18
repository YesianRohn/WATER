"""
Caption Mining for WATER-Z Pipeline (Step 1)

Uses Qwen3-VL-8B to analyze artistic text images and generate detailed captions.
Each caption uses <Text> as an editable placeholder for the specific word content.

The mined captions are released at:
    https://huggingface.co/datasets/Yesianrohn/WATER-Z_Captions

Usage:
    python caption_mining.py \
        --image_dir path/to/artistic_text_images \
        --label_file path/to/labels.txt \
        --output captions.json
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


CAPTION_PROMPT = (
    "I am providing an image of an artistic text area that has been cropped "
    "from a real photograph. "
    'The original text within the image is "{label}".'
    "Please generate a prompt template that can be used to create images in "
    "the same style as this artistic text area. In your prompt, replace the "
    "specific text content with <Text> so that I can easily substitute "
    "different text later. "
    "**Requirements:** "
    "- The prompt should be descriptive, suitable for use with large models "
    "(such as generative image models), and accurately capture the artistic "
    "and photographic style of the given text image. "
    "- Use <Text> as a placeholder for the text content within the prompt. "
    "- Output only the prompt template. "
    "- Do not include the original image, but focus on enabling the "
    "generation of a similar style. "
)


def parse_args():
    parser = argparse.ArgumentParser(description="Caption mining with Qwen3-VL")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing artistic text images")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Label file, each line: image_path<space>text_label")
    parser.add_argument("--output", type=str, default="captions.json",
                        help="Output JSON file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Qwen3-VL model name or path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    print(f"Loading {args.model} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Read label file: "image_path label"
    image_label_pairs = []
    with open(args.label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            img_rel_path, label = parts
            image_name = os.path.basename(img_rel_path)
            img_abs_path = os.path.join(args.image_dir, image_name)
            if not os.path.exists(img_abs_path):
                print(f"Warning: image not found -> {img_abs_path}")
                continue
            image_label_pairs.append({
                "image_name": image_name,
                "label": label,
                "image_path": img_abs_path,
            })

    print(f"Processing {len(image_label_pairs)} images ...")
    results = []

    for item in tqdm(image_label_pairs):
        prompt = CAPTION_PROMPT.format(label=item["label"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image_path"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            top_p=0.8,
            top_k=20,
            temperature=0.7,
            repetition_penalty=1.0,
            do_sample=True,
        )

        generated_ids_trimmed = generated_ids[:, len(inputs.input_ids[0]) :]
        caption = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        results.append({
            "image_name": item["image_name"],
            "label": item["label"],
            "caption": caption,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done! {len(results)} captions saved to: {args.output}")


if __name__ == "__main__":
    main()
