import sys,csv,numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0,'/mnt/data/active_portfolio/portfolio_projects_full/02_image_quality')
from quality.models import features
root=Path('/mnt/data/active_portfolio/full_dataset');dest=Path('/mnt/data/active_portfolio/full_features');dest.mkdir(exist_ok=True)
rows=list(csv.DictReader((root/'manifest.csv').open()))
chunk=int(sys.argv[1]);part=rows[chunk*2500:(chunk+1)*2500]
x=[]
for r in part:
 with Image.open(root/r['path']) as im:x.append(features(im))
np.savez_compressed(dest/f'chunk_{chunk}.npz',x=np.array(x),y=np.array([r['label'] for r in part]),splits=np.array([r['split'] for r in part]),source_ids=np.array([r['source_id'] for r in part]))
print('features',chunk,len(part),flush=True)
