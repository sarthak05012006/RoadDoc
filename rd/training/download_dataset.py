"""Download/cache the official Weather Whiplash dataset from Hugging Face."""
from datasets import load_dataset

DATASET_ID='adityakumarxdev/weather-whiplash'

ds=load_dataset(DATASET_ID)
print(ds)
for split, data in ds.items():
    print(f'{split}: {len(data)} rows')
    print('features:', data.features)
