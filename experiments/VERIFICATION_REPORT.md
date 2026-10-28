# Resume bullets 3 and 4: implementation and validation evidence

## What was actually run

- Loaded the previously trained full-scale OpenCV/Random Forest model (2,000-source, 20,000-variant experiment) from `full_scale_image_quality_evidence_2026-10-27.zip`.
- FastAPI `TestClient` validated `/health`, `/predict`, and `/batch` (two images), matching batch/single prediction outputs. Checked empty uploads (422), corrupt bytes (422), images below size requirement (422), oversized uploads (413), and batches larger than 20 (413). See `API_AND_LATENCY_RESULTS.json`.
- Corrected the original `/batch` implementation to vectorize Random Forest prediction instead of invoking the model independently for every image. It still performs per-image OpenCV feature extraction. This is an optimization of **model-call overhead**, not an end-to-end service speedup measurement.
- CPU prediction-only comparison: 24 precomputed synthetic-image feature vectors, 9 paired repetitions, 1056.473 ms median for 24 individual model calls versus 45.261 ms median for one vectorized call, approximately **23.34× speedup** in model invocation, with numerically identical probabilities. This is *not* a 23× end-to-end inference latency improvement. See `BATCH_OPTIMIZATION_RESULTS.json`.
- Repeated CPU experiments with input sizes 64, 128, 224 and the production model's trained resolution of 128. Five repetitions at each size and strategy. See `API_AND_LATENCY_RESULTS.json`. Resolution adjustments did not demonstrate a consistent latency improvement under these limited tests; previous held-out benchmarks show changes to model input resolution can reduce F1.
- Six image-quality project tests pass, including existing leakage/corruption tests. An outdated health test that assumed a missing model was changed to permit either ready/unready states depending on presence of trained artifacts.

## Limitations that still matter to the actual resume language

- Docker engine is **not installed in this runtime**. Dockerfile and requirements exist, and model weights are now packaged at the path referenced by the Dockerfile, but no real Docker image build/container-run has been completed here. Do not yet claim verified container deployment.
- Current `Dockerfile` copies all heavy training dependencies; image size and startup resource use have not been measured.
- Per-class confusion matrices and error patterns exist for the full-scale classifier, but a *severity-stratified error report* has not been completed.
- Generated input examples in the latency tests are random pixel arrays, not the real test photos. Generalization results and classification-quality comparisons should use the saved held-out predictions, not these synthetic timing inputs.
- Comparisons were made on this runtime's CPU, not on a production deployment or the user's Mac.

## Honest assessment

Bullet 3: **CPU input-size benchmarking and execution tuning are evidenced**. The strongest reproducible optimization is vectorized Random Forest inference, but the quoted 23.34× is specifically the model prediction stage, not end-to-end latency.

Bullet 4: **FastAPI service with validation, batch prediction and automated tests is evidenced**. A Dockerfile and model artifact are packaged. Real Docker image build/run and systematic severity-stratified error analysis remain to be verified.
