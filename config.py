import io
import zipfile
import re
import requests
import torch


class Config:
  def __init__(self, download=True):
    self.mode = False # true if you want sequential output
    self.episode_length = 10

    self.A_channels = 64
    self.B_channels = 64
    self.C_channels = 64
    self.D_channels= 64

    self.bias = True

    # dataset
    if download: UCI_HAR_download()
    x, y = UCI_HAR_read(
      x_path='UCI_HAR_Dataset/UCI HAR Dataset/train/X_train.txt',
      y_path='UCI_HAR_Dataset/UCI HAR Dataset/train/y_train.txt',
    ) # UCI_HAR_read
    self.x, self.y = list(map(str_to_tensor, x)), list(map(lambda val: torch.tensor(float(val)), y))
  # __init__
# Config

def UCI_HAR_download(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'):
  response = requests.get(url)
  if response.status_code == 200:
    print("Download successful!")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
      zip_ref.extractall("UCI_HAR_Dataset")
      print("Extraction complete.")
  else: print("Failed to download. Status code:", response.status_code)
  return response
# get_UCI_HAR_dataset

def UCI_HAR_read(x_path, y_path):
  x, y = None, None
  with open(x_path, 'r') as file:
    content = file.read()
    x = content.strip().split('\n')
  with open(y_path, 'r') as file:
    content = file.read()
    y = content.strip().split('\n')
  return x, y
# UCI_HAR_txt_to_tensor

def str_to_tensor(val):
  val = val.strip()
  val = re.split(r'\s+', val)
  val = [float(feature) for feature in val]
  return torch.tensor(val)
# str_to_tensor

if __name__ == "__main__":
  config = Config(download=False)
  for feature, label in zip(config.x, config.y):
    print(f'feature: {feature.shape} label: {label.item()}')
# if __name__ == "__main__":