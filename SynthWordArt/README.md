# SynthWordArt

**SynthWordArt** is an artistic-text-oriented rendering engine for generating large-scale synthetic WordArt data (**WATER-T**). It builds upon [SynthText](https://github.com/ankush-me/SynthText), [SynthTIGER](https://github.com/clovaai/synthtiger), and [UnionST](https://github.com/YesianRohn/UnionST), with key enhancements tailored for artistic text scenarios.

> **We recommend directly using our pre-generated WATER-T dataset** from [HuggingFace](https://huggingface.co/datasets/Yesianrohn/WATER-Data) for training. The engine is provided for reproducibility and custom generation.

## Key Enhancements over Standard Rendering Engines

1. **Artistic Font Library**: 11,250 open-source artistic fonts (vs. standard Google Fonts), covering styles like display, handwriting, cartoon, playful, calligraphy, etc.
2. **Enhanced Layout & Typography**: Beyond horizontal text, supports curved paths, vertical text, multi-orientation, perspective distortion, and geometric stretching.
3. **Real Text Corpus**: Uses text labels from existing real STR datasets (598,615 unique entries) instead of dictionary/news corpora, reducing distribution gap.

## Font Resources

Our curated artistic font library can be downloaded from: [artistic-fonts](https://huggingface.co/datasets/Yesianrohn/artistic-fonts)


> **Font License Disclaimer**: All fonts are collected from open-source platforms and retain their original licenses (OFL, Apache, Creative Commons, etc.). We have made our best effort to ensure licensing compliance for research use. If any font in our collection violates its license terms, please contact us and we will remove it promptly.


## Rendering Pipeline

For each sample, the engine:
1. **Randomly samples** a text string from the corpus, an artistic font, a layout mode, and a background image
2. **Renders** the text with the selected font at a random size and color
3. **Applies layout transformation**: one of horizontal / curved / vertical / multi-orientation / perspective
4. **Composites** the rendered text onto the background with optional shadow, stroke, and texture effects
5. **Saves** the image and its label

## Generation Statistics

- **Throughput**: ~23.24 samples/sec with 16 CPU workers (96-core Intel Xeon Platinum 8255C)
- **1M samples**: ~12 hours on the above hardware
- **No GPU required**: Pure CPU-based rendering

## References

- [SynthTIGER](https://github.com/clovaai/synthtiger) — Base rendering framework
- [SynthText](https://github.com/ankush-me/SynthText) — Scene text synthesis
- [UnionST](https://github.com/YesianRohn/UnionST) — Strong synthetic engine with diverse simulations
