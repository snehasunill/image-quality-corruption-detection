from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

BASE = Path(__file__).resolve().parent
OUT = BASE / 'error_analysis'
OUT.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(BASE/'full_dataset/manifest.csv')
preds = pd.read_csv(BASE/'full_models/opencv/test_predictions.csv')
test = manifest.loc[manifest['split'].eq('test')].reset_index(drop=True)
assert len(test) == len(preds) == 2930
assert (test['source_id'].values == preds['source_id'].values).all()
assert (test['label'].values == preds['truth'].values).all()
df=test.assign(prediction=preds['prediction'], correct=(preds['prediction'].values == test['label'].values))
df['severity'] = df['severity'].round(2)
labels=['clean','blur','noise','jpeg']
by_class=(df.groupby('label',sort=False).agg(samples=('correct','size'), errors=('correct',lambda x:int((~x).sum())), accuracy=('correct','mean')).reset_index())
by_class['error_rate']=1-by_class['accuracy']
by_severity=(df.groupby(['label','severity'],sort=False).agg(samples=('correct','size'),errors=('correct',lambda x:int((~x).sum())),accuracy=('correct','mean')).reset_index())
by_severity['error_rate']=1-by_severity['accuracy']
errors=df.loc[~df['correct'],['path','source_id','label','severity','prediction']]
by_class.to_csv(OUT/'by_corruption_type.csv', index=False)
by_severity.to_csv(OUT/'by_corruption_and_severity.csv', index=False)
errors.to_csv(OUT/'misclassified_images.csv',index=False)
cm=confusion_matrix(df['label'],df['prediction'],labels=labels)
metrics={'samples':int(len(df)),'correct':int(df.correct.sum()),'errors':int((~df.correct).sum()),'accuracy':float(df.correct.mean()),'macro_f1':float(f1_score(df.label,df.prediction,labels=labels,average='macro')),'labels':labels,'confusion_matrix':cm.tolist(),'classification_report':classification_report(df.label,df.prediction,labels=labels,output_dict=True,zero_division=0)}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
with (OUT/'ERROR_ANALYSIS_REPORT.md').open('w') as f:
 f.write('# Image Quality: Error Analysis by Corruption Type and Severity\n\n')
 f.write('## Scope and method\n\n')
 f.write('OpenCV + Random Forest, full-scale held-out synthetic test set; 2,930 variants from source-separated original photographs. Predictions are aligned against the saved manifest in strict row order, with both `source_id` and `label` verified for every row. Class labels: clean, blur, noise, jpeg.\n\n')
 f.write(f"**Overall:** accuracy {metrics['accuracy']:.4f}; macro F1 {metrics['macro_f1']:.4f}; {metrics['errors']} errors / {metrics['samples']} test variants.\n\n")
 f.write('## By corruption type\n\n')
 f.write('| Type | Samples | Errors | Error rate |\n|---|---:|---:|---:|\n')
 for _,r in by_class.iterrows():f.write(f"| {r.label} | {r.samples} | {r.errors} | {100*r.error_rate:.2f}% |\n")
 f.write('\n## By corruption type and generator severity\n\n')
 f.write('| Type | Severity parameter | Samples | Errors | Error rate |\n|---|---:|---:|---:|---:|\n')
 for _,r in by_severity.sort_values(['label','severity']).iterrows():f.write(f"| {r.label} | {r.severity:.2f} | {r.samples} | {r.errors} | {100*r.error_rate:.2f}% |\n")
 f.write('\n## Confusion matrix\n\nTrue labels are rows, predictions are columns; order: clean, blur, noise, jpeg.\n\n')
 f.write('```text\n'+ '\n'.join(str(list(map(int,row))) for row in cm)+'\n```\n\n')
 f.write('## Interpretation and limitations\n\n')
 f.write('- These metrics evaluate synthetic corruption recipes, not naturally degraded photos or an external image collection.\n')
 f.write('- `severity` is the numeric parameter used by the generator, not a human-rated or calibrated quality score. Its effect differs across Gaussian blur, additive Gaussian noise, and JPEG compression.\n')
 f.write('- Clean images carry a synthetic severity field that does **not** change their pixels; do not interpret clean severity trends as degradation response.\n')
 f.write('- The generator deterministically pairs variant indexes with class and severity, so some class/severity combinations are absent. The analysis describes observed combinations only.\n')
 f.write('- Performance by severity uses repeated variants from the same original photographs, not statistically independent source images; avoid interpreting small differences as statistically significant.\n')
 f.write('- See `misclassified_images.csv` for the file paths of every error; image examples can be inspected after regenerating the image files.\n')
 f.write('\n## Reproduction\n\nFrom a copy of `bullet34_verify`, run `python analyze_errors.py` after installing `pandas` and `scikit-learn`.\n')
print((OUT/'ERROR_ANALYSIS_REPORT.md').read_text()[:4500])
