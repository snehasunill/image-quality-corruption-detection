# Image Quality: Error Analysis by Corruption Type and Severity

## Scope and method

OpenCV + Random Forest, full-scale held-out synthetic test set; 2,930 variants from source-separated original photographs. Predictions are aligned against the saved manifest in strict row order, with both `source_id` and `label` verified for every row. Class labels: clean, blur, noise, jpeg.

**Overall:** accuracy 0.9307; macro F1 0.9289; 203 errors / 2930 test variants.

## By corruption type

| Type | Samples | Errors | Error rate |
|---|---:|---:|---:|
| clean | 879 | 138 | 15.70% |
| blur | 879 | 2 | 0.23% |
| noise | 586 | 0 | 0.00% |
| jpeg | 586 | 63 | 10.75% |

## By corruption type and generator severity

| Type | Severity parameter | Samples | Errors | Error rate |
|---|---:|---:|---:|---:|
| blur | 0.35 | 293 | 2 | 0.68% |
| blur | 0.49 | 293 | 0 | 0.00% |
| blur | 0.91 | 293 | 0 | 0.00% |
| clean | 0.35 | 293 | 46 | 15.70% |
| clean | 0.77 | 293 | 46 | 15.70% |
| clean | 0.91 | 293 | 46 | 15.70% |
| jpeg | 0.63 | 293 | 41 | 13.99% |
| jpeg | 0.77 | 293 | 22 | 7.51% |
| noise | 0.49 | 293 | 0 | 0.00% |
| noise | 0.63 | 293 | 0 | 0.00% |

## Confusion matrix

True labels are rows, predictions are columns; order: clean, blur, noise, jpeg.

```text
[741, 0, 6, 132]
[0, 877, 0, 2]
[0, 0, 586, 0]
[61, 2, 0, 523]
```

## Interpretation and limitations

- These metrics evaluate synthetic corruption recipes, not naturally degraded photos or an external image collection.
- `severity` is the numeric parameter used by the generator, not a human-rated or calibrated quality score. Its effect differs across Gaussian blur, additive Gaussian noise, and JPEG compression.
- Clean images carry a synthetic severity field that does **not** change their pixels; do not interpret clean severity trends as degradation response.
- The generator deterministically pairs variant indexes with class and severity, so some class/severity combinations are absent. The analysis describes observed combinations only.
- Performance by severity uses repeated variants from the same original photographs, not statistically independent source images; avoid interpreting small differences as statistically significant.
- See `misclassified_images.csv` for the file paths of every error; image examples can be inspected after regenerating the image files.

## Reproduction

From a copy of `bullet34_verify`, run `python analyze_errors.py` after installing `pandas` and `scikit-learn`.
