"""Leakage-safe illustrative baseline on generated synthetic images."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

LABELS = ['clean', 'blur', 'noise', 'jpeg']

def make_source(seed, size=96):
    rng = np.random.default_rng(seed)
    image = Image.new('RGB', (size, size), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    for _ in range(10):
        x, y = rng.integers(0, size, 2)
        radius = int(rng.integers(3, 18))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        draw.ellipse((int(x), int(y), int(x)+radius, int(y)+radius), fill=color)
    return image

def corrupt(img, kind, seed):
    if kind == 'clean': return img.copy()
    if kind == 'blur': return img.filter(ImageFilter.GaussianBlur(radius=3))
    if kind == 'noise':
        rng = np.random.default_rng(seed)
        array = np.clip(np.asarray(img).astype(np.float32) + rng.normal(0, 32, np.asarray(img).shape), 0, 255).astype('uint8')
        return Image.fromarray(array)
    if kind == 'jpeg':
        buffer = BytesIO(); img.save(buffer, format='JPEG', quality=8); buffer.seek(0)
        return Image.open(buffer).convert('RGB').copy()
    raise ValueError(kind)

def features(img):
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return [float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.std())]

def make_data(sources=80):
    x, y, groups = [], [], []
    for src_id in range(sources):
        source = make_source(src_id)
        for j, kind in enumerate(LABELS):
            x.append(features(corrupt(source, kind, src_id*4+j)))
            y.append(kind); groups.append(src_id)
    return np.array(x), np.array(y), np.array(groups)

def experiment():
    x, y, groups = make_data()
    train, test = next(GroupShuffleSplit(test_size=.25, random_state=42).split(x, y, groups))
    assert set(groups[train]).isdisjoint(groups[test])
    model = RandomForestClassifier(n_estimators=60, random_state=42).fit(x[train], y[train])
    report = classification_report(y[test], model.predict(x[test]), labels=LABELS, zero_division=0)
    print('TOY DATA ONLY — results are not resume metrics!')
    print(report)
    return model

if __name__ == '__main__': experiment()
