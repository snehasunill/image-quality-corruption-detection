"""Real saved-model inference only; refuses to pretend an untrained heuristic is a model."""
import io,json,os
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
from fastapi import FastAPI,File,HTTPException,UploadFile
from quality.data import LABELS
from quality.models import features,neural,transform

app=FastAPI(title='Image corruption classifier',version='1.0')
MAX_BYTES=8*1024*1024

@lru_cache(1)
def load():
    p=Path(os.environ.get('MODEL_DIR','artifacts/opencv'))
    meta=json.loads((p/'metadata.json').read_text())
    if meta['model']=='opencv':
        import joblib
        model=joblib.load(p/'model.joblib')
    else:
        import torch
        model=neural(meta['model'],pretrained=False)
        model.load_state_dict(torch.load(p/'weights.pt',map_location='cpu',weights_only=True))
        model.eval()
    return model,meta

@app.get('/health')
def health():
    try:_,meta=load();return {'ready':True,'model':meta['model']}
    except (FileNotFoundError,OSError,ValueError) as e:
        return {'ready':False,'reason':str(e)}

def decode_image(data):
    try:
        with Image.open(io.BytesIO(data)) as raw:
            im=ImageOps.exif_transpose(raw).convert('RGB')
            if min(im.size)<32 or im.width*im.height>25_000_000:raise ValueError('invalid image dimensions')
    except (UnidentifiedImageError,OSError,ValueError) as e:raise HTTPException(422,f'invalid image: {e}')
    return im

def infer(data):
    im=decode_image(data)
    try:model,meta=load()
    except (FileNotFoundError,OSError,ValueError) as e:raise HTTPException(503,f'model unavailable: {e}')
    if meta['model']=='opencv':
        scores=model.predict_proba([features(im.resize((meta.get('size',128),meta.get('size',128))))])[0]
        confidence=dict(zip(model.classes_,map(float,scores)))
    else:
        import torch
        with torch.inference_mode():
            x=transform(meta['size'],meta['model'])(im).unsqueeze(0)
            scores=torch.softmax(model(x),dim=1)[0].tolist()
        confidence=dict(zip(LABELS,scores))
    return {'label':max(confidence,key=confidence.get),'scores':confidence,'model':meta['model']}

async def read_image(file):
    data=await file.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:raise HTTPException(413,'file too large')
    if not data:raise HTTPException(422,'empty upload')
    return data

@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    return infer(await read_image(file))

@app.post('/batch')
async def batch(files:list[UploadFile]=File(...)):
    if len(files)>20:raise HTTPException(413,'max 20 images per batch')
    contents=[await read_image(f) for f in files]
    model,meta=load()
    if meta['model']!='opencv':
        return {'predictions':[infer(data) for data in contents]}
    # Vectorized Random Forest predict_proba; matches serial predictions.
    images=[decode_image(data) for data in contents]
    size=meta.get('size',128)
    feature_matrix=[features(im.resize((size,size))) for im in images]
    probabilities=model.predict_proba(feature_matrix)
    predictions=[]
    for prob in probabilities:
        confidence=dict(zip(model.classes_,map(float,prob)))
        predictions.append({'label':max(confidence,key=confidence.get),
                            'scores':confidence,'model':meta['model']})
    return {'predictions':predictions}
