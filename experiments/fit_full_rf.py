import numpy as np,json,joblib,csv
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
base=Path('/mnt/data/active_portfolio/full_features');out=Path('/mnt/data/active_portfolio/full_models/opencv');out.mkdir(parents=True,exist_ok=True)
a=[np.load(base/f'chunk_{i}.npz') for i in range(8)]
x=np.vstack([p['x'] for p in a]);y=np.concatenate([p['y'] for p in a]);splits=np.concatenate([p['splits'] for p in a]);ids=np.concatenate([p['source_ids'] for p in a]);assert len(x)==20000
assert not set(ids[splits=='train'])&set(ids[splits=='test']);assert not set(ids[splits=='train'])&set(ids[splits=='val']);assert not set(ids[splits=='val'])&set(ids[splits=='test'])
m=RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=42,n_jobs=4)
m.fit(x[splits=='train'],y[splits=='train'])
labels=['clean','blur','noise','jpeg'];metrics={}
for s in ['val','test']:
 mask=splits==s;p=m.predict(x[mask]);truth=y[mask]
 metrics[s]={'macro_f1':float(f1_score(truth,p,labels=labels,average='macro',zero_division=0)),'accuracy':float(accuracy_score(truth,p)),'confusion_matrix':confusion_matrix(truth,p,labels=labels).tolist(),'per_class':classification_report(truth,p,labels=labels,output_dict=True,zero_division=0),'n':int(mask.sum())}
 with (out/f'{s}_predictions.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['source_id','truth','prediction']);w.writerows(zip(ids[mask],truth,p))
joblib.dump(m,out/'model.joblib');(out/'metadata.json').write_text(json.dumps({'model':'opencv','seed':42,'size':128,'n_train':int((splits=='train').sum()),'n_sources':2000,'n_variants':20000,'labels':labels,'metrics':metrics},indent=2));print(json.dumps({k:{'macro_f1':v['macro_f1'],'accuracy':v['accuracy'],'n':v['n']} for k,v in metrics.items()},indent=2))
