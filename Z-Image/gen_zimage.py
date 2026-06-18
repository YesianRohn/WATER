"""
WATER-Z Image Generation with Z-Image

Multi-GPU parallel generation of artistic text images using Z-Image-Turbo
and the prompt library from the WATER-Z pipeline.

Usage:
    python gen_zimage.py \
        --captions_file prompts.txt \
        --corpus_file corpus.txt \
        --num_images 1000000 \
        --output_dir output \
        --world_size 8
"""

import os
import json
import random
import argparse
import numpy as np
import h5py
import torch
import torch.multiprocessing as mp
from io import BytesIO
from tqdm import tqdm
from diffusers import ZImagePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="WATER-Z image generation")
    parser.add_argument("--captions_file", type=str, required=True,
                        help="Prompt template file (one per line, containing <Text>)")
    parser.add_argument("--corpus_file", type=str, required=True,
                        help="Text corpus file (one word/phrase per line)")
    parser.add_argument("--num_images", type=int, default=250000,
                        help="Total number of images to generate")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for HDF5 files")
    parser.add_argument("--world_size", type=int, default=8,
                        help="Number of GPUs for parallel generation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo",
                        help="Z-Image model name or path")
    return parser.parse_args()


def get_completed_num(hdf5_file):
    """Get number of already generated images for resume support."""
    if os.path.exists(hdf5_file):
        with h5py.File(hdf5_file, "r") as hf:
            if "images" in hf:
                return hf["images"].shape[0]
    return 0


def worker(rank, world_size, args):
    """Per-GPU worker: generate images and save to HDF5."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    pipe = ZImagePipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    pipe.to(device)

    with open(args.captions_file, "r", encoding="utf-8") as f:
        prompts_tpl = [line.strip() for line in f if line.strip()]
    with open(args.corpus_file, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    chunk = (args.num_images + world_size - 1) // world_size
    start = rank * chunk
    end = min(args.num_images, (rank + 1) * chunk)

    os.makedirs(args.output_dir, exist_ok=True)
    hdf5_file = os.path.join(args.output_dir, f"images_rank_{rank}.h5")
    completed = get_completed_num(hdf5_file)

    with h5py.File(hdf5_file, "a") as h5f:
        if "images" not in h5f:
            images_ds = h5f.create_dataset(
                "images", shape=(0,), maxshape=(None,),
                dtype=h5py.special_dtype(vlen=np.uint8),
            )
            meta_ds = h5f.create_dataset(
                "meta", shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype("utf-8"),
            )
        else:
            images_ds = h5f["images"]
            meta_ds = h5f["meta"]

        pbar = tqdm(total=end - start - completed, desc=f"GPU{rank}", position=rank)
        i = start + completed

        while i < end:
            bsize = min(args.batch_size, end - i)
            prompts, seeds, texts = [], [], []

            for j in range(bsize):
                seed = i + j
                random.seed(seed)
                text = random.choice(corpus)
                tpl = random.choice(prompts_tpl)
                prompt = tpl.replace("<Text>", text) + \
                    " Please note that the text should be rendered on a single line."
                prompts.append(prompt)
                seeds.append(seed)
                texts.append(text)

            generator = torch.Generator(device)
            generators = [generator.manual_seed(s) for s in seeds]

            images = pipe(
                prompt=prompts, height=256, width=256,
                num_inference_steps=9, guidance_scale=0.0,
                generator=generators,
            ).images

            img_bytes_list, meta_list = [], []
            for j in range(bsize):
                buf = BytesIO()
                images[j].save(buf, format="png")
                img_bytes_list.append(np.frombuffer(buf.getvalue(), dtype=np.uint8))
                meta_list.append(json.dumps({
                    "id": i + j + 1, "prompt": prompts[j], "text": texts[j],
                }, ensure_ascii=False))

            cur_len = images_ds.shape[0]
            images_ds.resize((cur_len + bsize,))
            meta_ds.resize((cur_len + bsize,))
            for j in range(bsize):
                images_ds[cur_len + j] = img_bytes_list[j]
                meta_ds[cur_len + j] = meta_list[j]
            h5f.flush()

            pbar.update(bsize)
            i += bsize
        pbar.close()


def merge_hdf5(args):
    """Merge per-rank HDF5 files into one, sorted by id."""
    all_imgs, all_meta = [], []
    for rank in range(args.world_size):
        part = os.path.join(args.output_dir, f"images_rank_{rank}.h5")
        if not os.path.exists(part):
            continue
        with h5py.File(part, "r") as hf:
            all_imgs.extend(list(hf["images"][:]))
            all_meta.extend(list(hf["meta"][:]))

    print(f"Merging {len(all_imgs)} images from {args.world_size} ranks ...")
    metas = [json.loads(m) for m in all_meta]
    order = np.argsort([m["id"] for m in metas])
    sorted_imgs = [all_imgs[i] for i in order]
    sorted_meta = [json.dumps(metas[i], ensure_ascii=False) for i in order]

    out = os.path.join(args.output_dir, "images_all.h5")
    with h5py.File(out, "w") as h5f:
        h5f.create_dataset("images", data=np.array(sorted_imgs, dtype=object),
                           dtype=h5py.special_dtype(vlen=np.uint8))
        h5f.create_dataset("meta", data=sorted_meta,
                           dtype=h5py.string_dtype("utf-8"))
    print(f"Merged HDF5 saved to: {out}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    merge_hdf5(args)


if __name__ == "__main__":
    main()
