import os,sys,time,statistics,json,io
from pathlib import Path
import numpy as np
from PIL import Image
base=Path('/mnt/data/bullet34_verify');p=base/'portfolio_projects_full/02_image_quality';sys.path.insert(0,str(p));os.environ['MODEL_DIR']=str(p/'artifacts/opencv')
from api import load
from quality.models import features
model,meta=load()
rng=np.random.default_rng(444)
imgs=[Image.fromarray(rng.integers(0,256,(128,128,3),dtype=np.uint8)) for _ in range(24)]
fs=[features(im) for im in imgs]
serial=[];batched=[]
# fixed features; isolates model invocation overhead rather than including pixel processing
for j in range(9):
 t=time.perf_counter();a=np.vstack([model.predict_proba([f])[0] for f in fs]);serial.append((time.perf_counter()-t)*1000)
 t=time.perf_counter();b=model.predict_proba(fs);batched.append((time.perf_counter()-t)*1000)
 assert np.allclose(a,b)
result={'images':len(fs),'repeats':9,'serial_median_ms':round(statistics.median(serial),3),'batched_median_ms':round(statistics.median(batched),3),'speedup':round(statistics.median(serial)/statistics.median(batched),2),'outputs_equal':True,'scope':'Model prediction only, features precomputed, random generated images, CPU environment','serial_trials_ms':[round(x,3) for x in serial],'batched_trials_ms':[round(x,3) for x in batched]}
(base/'BATCH_OPTIMIZATION_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
