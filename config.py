import io
import zipfile
import requests
import torch


class Config:
  def __init__(self, download=True):
    self.mode = False # true if you want sequential output

    if download: UCI_HAR_download()
    x, y = UCI_HAR_read(
      x_path='UCI_HAR_Dataset/UCI HAR Dataset/train/X_train.txt',
      y_path='UCI_HAR_Dataset/UCI HAR Dataset/train/y_train.txt',
    )  # UCI_HAR_read
    self.x, self.y = list(map(str_to_tensor, x)), list(map(lambda val: torch.tensor(float(val)), y))
    print('str_to_tensor is applied to self.x\nint_to_hv is applied to self.y')
    self.episode_length = len(self.x[0])
    self.max_var_len = get_max_x_len(self.x)
    self.x = map(lambda val: zero_padding(val, self.max_var_len), self.x)
    self.x = list(self.x)
    self.output_dim = self.y[0].shape[0]

    # hippo init
    in_features, hid_features, out_features = 64, 64, 10
    self.delta = {
      "k": self.episode_length,
      "in_features": in_features,
      "hid_features": hid_features,
      "out_features": out_features
    } # self.delta
    self.A = {
      "in_features": in_features,
      "out_features": hid_features,
    } # self.A
    self.B = {
      "in_features": in_features,
      "out_features": hid_features,
    }  # self.B
    self.C = {
      "in_features": in_features,
      "out_features": out_features,
    }  # self.C
    self.D = {
      "in_features": in_features,
      "out_features": out_features,
    }  # self.D
    self.bias = True

    self.dummy = torch.zeros_like(self.x[0])
  # __init__
# Config

linear_apply = lambda config: nn.Linear(in_features=config.in_features, out_features=config.out_features)

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
  val = val.split('  ')
  val = [torch.tensor([float(feature) for feature in seq.split(' ')]) for seq in val]
  return val
# str_to_tensor

def zero_padding(sequence, max_length):
  return_sequence = list()
  for feature in sequence:
    len_feature = feature.shape[0]
    if len_feature < max_length:
      zeros = torch.zeros(max_length)
      for idx, item in enumerate(feature): zeros[idx] = item
      return_sequence.append(zeros)
    else:
      return_sequence.append(feature)
  return return_sequence
# _zero_padding

def get_max_x_len(dataset):
  sequence = dataset[0]
  length_list = list()
  for feature in sequence: length_list.append(feature.shape[0])
  return max(length_list)
# get_max_x_len

if __name__ == "__main__":
  config = Config(download=False)
  print(f'max_var_len: {config.max_var_len}')
  for feature in config.x[0]: print(f'feature: {feature.shape[0]}', end=' ')
# if __name__ == "__main__":