import io,csv
from pathlib import Path
from PIL import Image
from quality.data import build,source_split,corrupt,LABELS
from quality.train import read_manifest
import pytest

def test_corruptions():
    im=Image.new('RGB',(48,48),color='red')
    for label in LABELS:assert corrupt(im,label,42).size==im.size

def test_split_is_stable():assert source_split('a.jpg')==source_split('a.jpg')

def test_manifest_no_leak(tmp_path):
    src=tmp_path/'sources';src.mkdir()
    for i in range(90):Image.new('RGB',(48,48),(i,80,30)).save(src/f'{i}.png')
    rows=build(src,tmp_path/'output',variants_per_source=4,size=48)
    assert len(rows)==360
    assert len(read_manifest(tmp_path/'output'))==360
    groups={}
    for row in rows:
        assert row['source_id'] not in groups or row['split']==groups[row['source_id']]
        groups[row['source_id']]=row['split']

def test_reject_leak(tmp_path):
    (tmp_path/'manifest.csv').write_text('source_id,path,label,split,severity\na,a.png,clean,train,0.5\na,b.png,noise,test,0.5\n')
    with pytest.raises(ValueError,match='LEAKAGE'):read_manifest(tmp_path)
