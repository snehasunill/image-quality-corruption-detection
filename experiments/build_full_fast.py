import os,zipfile,hashlib,json,csv,io,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image,ImageOps
import sys
sys.path.insert(0,'/mnt/data/active_portfolio/portfolio_projects_full/02_image_quality')
from quality.data import corrupt,source_split,LABELS
ZIP='/mnt/data/selected_2000.zip'; OUT=Path('/mnt/data/active_portfolio/full_dataset');SIZE=128

def worker(arg):
 idx,name=arg
 with zipfile.ZipFile(ZIP) as z:
  with Image.open(io.BytesIO(z.read(name))) as img:
   img=ImageOps.fit(ImageOps.exif_transpose(img).convert('RGB'),(SIZE,SIZE),method=Image.Resampling.LANCZOS)
 split=source_split(Path(name).name); rows=[]
 for j in range(10):
  label=LABELS[j%4];sev=.35+(j%5)*.14
  rel=f'{split}/{label}/{idx:06}_{j:03}.png';target=OUT/rel
  if not target.exists():
   result=corrupt(img,label,42+idx*100003+j,sev)
   target.parent.mkdir(parents=True,exist_ok=True);result.save(target)
  rows.append((rel,Path(name).name,split,label,sev))
 return rows

def main():
 with zipfile.ZipFile(ZIP) as z:
  names=sorted(n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png')) and not n.startswith('__MACOSX/'))
 print('SOURCE COUNT',len(names),flush=True)
 assert len(names)==2000
 start=time.time();rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for k,result in enumerate(pool.map(worker,enumerate(names),chunksize=5)):
   rows.extend(result)
   if (k+1)%250==0:print('sources complete',k+1,'elapsed',round(time.time()-start,1),flush=True)
 with (OUT/'manifest.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['path','source_id','split','label','severity']);w.writerows(rows)
 report={'input_sources':len(names),'valid_sources':len(names),'variants':len(rows),'size':SIZE,'excluded':[],'split_counts':{s:sum(r[2]==s for r in rows) for s in ('train','val','test')},'synthetic':True}
 (OUT/'build_report.json').write_text(json.dumps(report,indent=2))
 print(report,flush=True)
if __name__=='__main__':main()
