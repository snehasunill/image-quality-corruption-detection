import sys,os,io,time,json,statistics,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
root=Path('/mnt/data/bullet34_verify'); p=root/'portfolio_projects_full/02_image_quality'
sys.path.insert(0,str(p)); os.environ['MODEL_DIR']=str(p/'artifacts/opencv')
from api import app,load,infer
from quality.models import features
client=TestClient(app)
res={}
def png(sz=128):
 a=np.random.default_rng(99).integers(0,256,(sz,sz,3),dtype=np.uint8)
 b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG');return b.getvalue()
data=png()
res['health']=client.get('/health').json()
r=client.post('/predict',files={'file':('x.png',data,'image/png')});res['predict_status']=r.status_code;res['predict']=r.json()
r=client.post('/batch', files=[('files',('a.png',data,'image/png')),('files',('b.png',data,'image/png'))]);res['batch_status']=r.status_code;res['batch_count']=len(r.json().get('predictions',[]));res['batch_matches_single']=all(item['label']==res['predict']['label'] and item['scores']==res['predict']['scores'] for item in r.json()['predictions'])
checks={'empty':('file',('empty.png',b'','image/png'),422),'badbytes':('file',('bad.png',b'bad input','image/png'),422),'oversize':('file',('big.jpg',b'x'*(8*1024*1024+1),'image/jpeg'),413),'tiny_image':('file',('small.png',png(16),'image/png'),422)}
res['negative']={}
for k,(field,file,expected) in checks.items():
 response=client.post('/predict',files={field:file});res['negative'][k]={'status':response.status_code,'pass':response.status_code==expected}
r=client.post('/batch',files=[('files',(str(i)+'.png',data,'image/png')) for i in range(21)]);res['negative']['batch_overlimit']={'status':r.status_code,'pass':r.status_code==413}
model,meta=load()
res['latency_ms']={}
# paired speed experiment identical model, common request path; batch model feature inference vs serial
for sz in (64,128,224):
 images=[Image.open(io.BytesIO(png(sz))).convert('RGB') for _ in range(8)]
 for optimized in (False,True):
  key='model_input_resized_128' if optimized else 'native_resolution'
  times=[]
  for _ in range(5):
   t=time.perf_counter()
   fs=[features(im.resize((128,128))) if optimized else features(im) for im in images]
   model.predict_proba(fs)
   times.append((time.perf_counter()-t)*1000/len(images))
  res['latency_ms'].setdefault(str(sz),{})[key]={'median_per_image_ms':round(statistics.median(times),3),'five_trials':[round(x,3) for x in times]}
(root/'API_AND_LATENCY_RESULTS.json').write_text(json.dumps(res,indent=2))
print(json.dumps({k:v for k,v in res.items() if k!='predict'},indent=2))
