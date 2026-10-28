from demo_pipeline import LABELS, make_data
from sklearn.model_selection import GroupShuffleSplit
from api import app
from fastapi.testclient import TestClient

def test_label_counts_and_grouping():
    x, y, groups = make_data(12)
    assert x.shape == (48, 2)
    assert set(y) == set(LABELS)
    tr, te = next(GroupShuffleSplit(test_size=.25, random_state=42).split(x,y,groups))
    assert set(groups[tr]).isdisjoint(groups[te])

def test_api():
    c = TestClient(app)
    assert isinstance(c.get('/health').json()['ready'], bool)
    assert c.post('/predict', files={'file': ('notimage.jpg', b'bad', 'image/jpeg')}).status_code == 422
    assert c.post('/predict', files={'file': ('empty.jpg', b'', 'image/jpeg')}).status_code == 422
