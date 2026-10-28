# Image Quality & Corruption Detection — evidence index

## What this package includes

- `project/`: Python sources for corruption generation, OpenCV baseline, CNN/ResNet18 training, FastAPI, Dockerfile, tests and requirements. The API is from the later batch-optimization verification package.
- `evidence/full_dataset/`: 20,000-entry source-grouped manifest and generation summary; *not* the image bytes.
- `evidence/full_models/opencv/`: trained OpenCV/Random Forest model, predictions, metrics.
- `evidence/FULL_SCALE_EVIDENCE.md`: full 2,000-source experiment details including test F1 ~0.9289 on 2,930 synthetically corrupted variants.
- `notebooks/`: Google Colab notebook for custom CNN and pretrained ResNet18 training.
- `evidence/API_AND_LATENCY_RESULTS.json`, `BATCH_OPTIMIZATION_RESULTS.json`: measured CPU/API and model-only batch speed results. Batch speedup is not end-to-end API speedup.
- `evidence/error_analysis/`: confusion/errors by class and synthetic severity, 203 misclassifications in 2,930 test variants.
- `docker_validation/`: screenshots submitted by user demonstrating FastAPI Swagger docs, health, prediction, and Docker-served batch HTTP 200 with two predictions.

## Separately retained evidence — not available for bundling here

1. Original 2,000 COCO source photos: the user-uploaded `selected_2000.zip` (link separately); not duplicated here to reduce archive size.
2. Trained custom CNN weights, pretrained ResNet18 weights, train histories, original validation/test prediction CSVs: in user's Google Drive `MyDrive/portfolio_results/cnn/` and `MyDrive/portfolio_results/transfer/`; download and retain these yourself. They were NOT accessible in the current runtime and are NOT in this bundle.
3. Robustness result: in Drive `MyDrive/portfolio_results/robustness/report.json`; screenshot from Colab in `docker_validation/` shows the exploratory 0.6452 macro F1 on 400 variants, but the JSON has not been copied into this package.
4. Complete 20,000-image generated corpus: not included; regenerate using `project/` and original photos.
5. Actual Docker image layers: not included; reproducible from Dockerfile on Docker Desktop.

## Claim boundaries

- The 0.9289 F1 applies to source-separated **synthetic** test variants; not natural-world corruptions.
- The 0.9981 ResNet18 test macro F1 is a reported Colab result; its performance under altered corruption methods was much lower and the 0.9981 claim should not be presented as general-purpose robustness.
- OpenCV run and neural runs had distinct test counts (2,930 versus 2,980). Don't imply an identical test protocol between these datasets.
- `July 2026–September 2026` dates are not established by the generated files. Use accurate real project dates.
- The file bundle documents execution evidence but doesn't independently prove who wrote each line of code or performed each action.

## Preservation checklist

- Keep these sources and manifests under version control (exclude huge model weights and original image corpus, use releases/artifacts).
- Save the CNN and transfer folders from Drive, and save notebook outputs (File > Download > Download .ipynb).
- Save an environment lock, random seeds, and deployment test instructions.
- If republishing COCO images, review the applicable dataset/image licenses and attribution.
