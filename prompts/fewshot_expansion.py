"""
Few-Shot Prompt Expansion for WATER-Z Pipeline (Step 2)

Given initial captions mined from artistic text images (Step 1), uses Qwen3-VL
in a few-shot manner to expand them into a large-scale prompt library.

The final prompt library (273,488 prompts) is released at:
    https://huggingface.co/datasets/Yesianrohn/WATER-Z_Captions

Usage:
    python fewshot_expansion.py \
        --input captions.txt \
        --target_count 50000 \
        --output expanded_prompts.txt
"""

import os
import argparse
import random
import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


FEWSHOT_PROMPT = (
    "You are a professional prompt engineer expert in creating an image of "
    "an artistic text area that has been cropped from a real photograph. "
    "Your task is to generate a new prompt that fully mimic the style, tone, "
    "and structural pattern of the provided reference prompts."
    "Here are three examples:\n"
    "{ref_0}\n"
    "{ref_1}\n"
    "{ref_2}\n"
    "**Requirements:** "
    "- The prompt should be descriptive, suitable for use with large models "
    "(such as generative image models), and accurately capture the artistic "
    "and photographic style. "
    "- Use <Text> as a placeholder for the text content within the prompt. "
    "- Output only the new prompt template as briefly as possible. "
)


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot prompt expansion")
    parser.add_argument("--input", type=str, required=True,
                        help="Input caption file (one caption per line)")
    parser.add_argument("--output", type=str, default="expanded_prompts.txt",
                        help="Output expanded prompts file")
    parser.add_argument("--target_count", type=int, default=50000,
                        help="Target number of unique prompts to generate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Qwen3-VL model name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_and_deduplicate(txt_path):
    """Load captions, keep only valid ones containing <Text>, and deduplicate."""
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f.readlines()]
    valid = [c for c in raw if c and "<Text>" in c]
    deduped = list(set(valid))
    print(f"Loaded {len(raw)} raw -> {len(deduped)} unique valid captions")
    return deduped


def generate_single(processor, model, ref_captions):
    """Generate one new prompt from 3 random reference captions."""
    refs = random.sample(ref_captions, 3)
    prompt = FEWSHOT_PROMPT.format(ref_0=refs[0], ref_1=refs[1], ref_2=refs[2])

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

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

    return caption if "<Text>" in caption else None


def main():
    args = parse_args()
    random.seed(args.seed)

    ref_captions = load_and_deduplicate(args.input)

    print(f"Loading {args.model} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    unique_prompts = set()
    rounds = 0
    start = time.time()

    if os.path.exists(args.output):
        os.remove(args.output)

    while len(unique_prompts) < args.target_count:
        try:
            new_prompt = generate_single(processor, model, ref_captions)
            if new_prompt and new_prompt not in unique_prompts:
                unique_prompts.add(new_prompt)
                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(new_prompt + "\n")

            rounds += 1
            elapsed = time.time() - start
            cur = len(unique_prompts)
            print(f"Round {rounds} | {cur}/{args.target_count} | "
                  f"Remaining: {args.target_count - cur} | Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"Error at round {rounds}: {e}, retrying in 1s ...")
            time.sleep(1)

    total = time.time() - start
    print(f"Done! {len(unique_prompts)} prompts saved to: {args.output}")
    print(f"Total time: {total:.1f}s | Total rounds: {rounds}")


if __name__ == "__main__":
    main()
