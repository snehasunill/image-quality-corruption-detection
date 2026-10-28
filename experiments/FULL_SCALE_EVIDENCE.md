# Full-scale image-quality experiment — 27 October 2026

## Reproduced experiment

- Source: user-supplied `selected_2000.zip`, 2,000 COCO JPEGs. Input archive is **not** included in this evidence bundle.
- Generated 20,000 synthetic variants at 128×128 pixels, ten per image.
- Four labels: clean, blur, noise and JPEG compression. Distribution is not balanced (3 clean, 3 blurred, 2 noisy, 2 JPEG per source).
- Deterministic SHA-256 source-level split (seed 42): training 13,960, validation 3,110, test 2,930.
- OpenCV handcrafted image-quality features + 250-tree class-balanced Random Forest; sklearn F1 macro over all four fixed labels.
- Validation: F1 macro **0.920641**, accuracy **0.922186**.
- Unmodified held-out test: F1 macro **0.928940**, accuracy **0.930717**.
- Test per-class F1: clean **0.881618**, blur **0.997725**, noise **0.994907**, JPEG **0.841512**.

## Interpret results carefully

- These scores measure recognition of the specific *synthetic corruption recipes*, not authentic camera blur, sensor noise, or deployment-domain corruption.
- Corruption artifacts are in some cases deliberately conspicuous, and the synthetic classes are not balanced. Do not describe results as industry-standard real-world image-quality performance.
- This experiment proves the dataset count and **OpenCV baseline** score, **not** comparison with trained CNN and **pretrained** ResNet18, CPU optimization, independent image collection, or Tableau dashboard.
- The supplied pretrained ResNet18 weights were not present in the CPU-only runtime. Download or provision weights and use the provided `quality.train --model transfer --pretrained` workflow on a GPU machine.
- All source images must be split as groups; never create the split after augmentation.

## Reproduce

From `portfolio_projects_full/02_image_quality`:

```
python build_full_fast.py     # requires original ZIP at configured ZIP path; produces full_dataset
python extract_features.py 0  # repeat for chunk IDs 0–7
python fit_full_rf.py
python -m quality.train --data <full_dataset_path> --out <model_out> --model transfer --pretrained --epochs 8 --size 128 --batch 32
```

The last command requires external pretrained weights and compute, and **has not been run successfully here**. For CPU use, reduce batch size and plan for longer run time. The first three scripts contain absolute `/mnt/data/...` paths that must be adjusted on another computer.
