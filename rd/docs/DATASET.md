# Weather Whiplash dataset integration

Source: https://huggingface.co/datasets/adityakumarxdev/weather-whiplash

The repo does not redistribute the raw dataset. It contains reproducible code to download it from the source and convert it to ImageFolder format for training. Hugging Face Datasets supports loading image columns as PIL images, which is what `prepare_dataset.py` uses.

Pipeline:

1. `training/download_dataset.py` verifies the Hub dataset is reachable.
2. `training/prepare_dataset.py` downloads/loads the train split and writes labeled images.
3. `training/train.py` fine-tunes ResNet18.
4. `backend/model_service.py` loads the resulting checkpoint.
5. `backend/app.py` serves `/api/analyze`.
6. The premium frontend calls the API first and falls back to its original browser CV pipeline if the backend is offline.
