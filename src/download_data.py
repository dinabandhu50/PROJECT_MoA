import os
import config
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()

print('Downloading data ...')
api.competition_download_files('lish-moa',path=os.path.join(config.ROOT_DIR,'input'))

with zipfile.ZipFile(os.path.join(config.ROOT_DIR,'input','lish-moa.zip'), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(config.ROOT_DIR,'input','raw'))

print('Finished data downloading')


# # remove the zip after unziping
# os.remove(os.path.join(config.ROOT_DIR,'input','lish-moa.zip'))

# # Single file download
# api.competition_download_file(
#     'lish-moa',
#     'train_targets_scored.csv',
#     path=os.path.join(config.ROOT_DIR,'input'))

# # console line to download data
# kaggle competitions download -c lish-moa