# Image Quality & Corruption Detection

A reproducible four-class image-quality classification experiment (`clean`, `blur`, `noise`, `jpeg`) using OpenCV/scikit-learn baselines, PyTorch CNNs, and ResNet18 transfer learning, with a Dockerized FastAPI inference service.

## Results and evidence

| Approach | Macro F1 | Evaluation caveat |
| --- | ---: | --- |
| OpenCV features + Random Forest | 0.9289 | 2,930 source-separated held-out synthetic variants |
| Custom CNN | 0.9086 | 2,980 held-out synthetic variants (separate Colab build) |
| ImageNet-pretrained ResNet18 | 0.9981 | 2,980 held-out synthetic variants (separate Colab build) |

The OpenCV and neural model runs did **not** use identical test manifests; this is not a controlled matched-test ranking. A subsequent exploratory test of different corruption recipes on 100 COCO test-source photographs gave a reported ResNet18 macro F1 of 0.6452, but preprocessing equivalence needs auditing before drawing definitive conclusions. **No separate photographic collection has yet been evaluated.**

- Original sources: 2,000 COCO photographs (not redistributed here).
- Generated: 20,000 synthetic variants with grouped source-level splits.
- OpenCV model: `artifacts/opencv/`; metrics and predictions: `experiments/opencv_metrics/`.
- Generation audit: `experiments/full_dataset/build_report.json` and `manifest.csv`.
- Error analysis: `experiments/error_analysis/`.
- CPU model-stage batch vectorization: ~23.34x speedup on 24-image prediction batches compared with repeated single-sample prediction calls, based on nine repeats. **Not** an end-to-end HTTP speedup.
- Trained neural checkpoints are separately preserved in Colab/Google Drive and are not included here.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
MODEL_DIR=artifacts/opencv uvicorn api:app --host 127.0.0.1 --port 8000
```

API documentation: `http://localhost:8000/docs`; endpoints: `/health`, `/predict`, `/batch`. Supply multipart JPEG/PNG images.

## Docker

```bash
docker build -t image-quality-api .
docker run --rm -p 8000:8000 --name image-quality-api image-quality-api
```

With the container running, use a different Terminal window to make a request:

```bash
curl -F "file=@/path/to/photo.jpg" http://localhost:8000/predict
```

The Docker image serves the saved OpenCV Random Forest model in `artifacts/opencv/`, **not** the ResNet18 checkpoint. Only load model artifacts from trusted sources.

## Regenerating the dataset

Original COCO photos are not included because of dataset size and image rights. Place permitted source photos in `my_source_photos/` and run:

```bash
python -m quality.data --sources my_source_photos --output data/generated --variants 10 --size 128
```

Other experiment scripts, training code, and the Colab notebook are included; see `quality/`, `train_cnn.py`, `notebooks/` and `experiments/`. For exact historical run evidence, use the saved manifests and metadata rather than assuming a new random split matches them.

## Notes on provenance

This was developed with AI coding assistance and verified by running local experiments, Colab GPU training, and Docker API tests. Git commits date when files were committed, not necessarily when experimentation originally started. Document project dates based on actual work rather than altering Git timestamps.
