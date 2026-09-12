# Dataset

This directory is intentionally empty in the repository. The project uses the official Hugging Face dataset:

`adityakumarxdev/weather-whiplash`

Run:

```bash
python3 -m pip install -r requirements.txt
python3 training/download_dataset.py
python3 training/prepare_dataset.py
```

`prepare_dataset.py` materializes the dataset into `dataset/train/<class>` and `dataset/val/<class>`. The split is grouped by `video_id` when that field is available, reducing leakage from adjacent frames of the same video.

The raw dataset is not committed to Git because it is much larger than a normal source repository and is maintained on Hugging Face.
