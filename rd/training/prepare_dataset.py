"""Materialize the Hugging Face dataset into train/val ImageFolder directories.

The split is grouped by video_id when available to reduce frame leakage between
train and validation. The script is intentionally data-source driven: it pulls
the official dataset rather than copying the raw dataset into this repository.
"""
from pathlib import Path
import random
from collections import defaultdict
from datasets import load_dataset

DATASET_ID='adityakumarxdev/weather-whiplash'
OUT=Path(__file__).resolve().parents[1]/'dataset'
CLASSES=['dry','damp','drying','wet']
SEED=42
random.seed(SEED)

def label_name(row, features):
    # Prefer a human-readable category when present.
    for key in ('category','label'):
        if key not in row: continue
        value=row[key]
        if isinstance(value,str):
            v=value.strip().lower()
            if v in CLASSES: return v
        if key in features and hasattr(features[key],'names') and isinstance(value,int):
            names=features[key].names
            if 0<=value<len(names):
                v=names[value].strip().lower()
                if v in CLASSES: return v
    return None

ds=load_dataset(DATASET_ID, split='train')
features=ds.features
rows=list(ds)
print(f'Loaded {len(rows)} rows from {DATASET_ID}')

# Group by video_id if available.
groups=defaultdict(list)
if 'video_id' in features:
    for i,row in enumerate(rows): groups[str(row['video_id'])].append(i)
    keys=list(groups); random.shuffle(keys)
    target=max(1,int(len(rows)*0.2)); val_ids=set(); count=0
    for k in keys:
        val_ids.add(k); count += len(groups[k])
        if count>=target: break
    val_idx={i for k in val_ids for i in groups[k]}
else:
    idx=list(range(len(rows))); random.shuffle(idx); val_idx=set(idx[:int(len(idx)*0.2)])

for split in ('train','val'):
    for c in CLASSES: (OUT/split/c).mkdir(parents=True,exist_ok=True)

counts=defaultdict(int)
for i,row in enumerate(rows):
    label=label_name(row,features)
    if label is None: continue
    split='val' if i in val_idx else 'train'
    image=row['image'].convert('RGB')
    image.save(OUT/split/label/f'{i:06d}.jpg',quality=95)
    counts[(split,label)]+=1

print('Prepared:')
for split in ('train','val'):
    print(split,{c:counts[(split,c)] for c in CLASSES})
print(f'Output: {OUT}')
