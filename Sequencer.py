from torch.utils.data import Dataset

class Sequencer:
  def __init__(self, config, dataset):
    self.config = config
    self.dataset = dataset
    self.dataset_len = len(self.dataset)
  # __init__

  def __len__(self): return self.dataset_len

  def __getitem__(self, idx):
    assert idx < self.dataset.__len__(), "the given idx is out of bound."
    raw_sequence = self.dataset[idx]
    sequence = Sequence(config=self.config, data=raw_sequence)
    return sequence
  # __getitem__
# Sequencer

class Sequence(Dataset):
  def __init__(self, config, data):
    super(Sequence, self).__init__()
    self.config = config
    self.data = data
    self.data_len = len(self.data)
  # __init__

  def __len__(self): return self.data_len

  def __getitem__(self, idx):
    assert idx < self.data.__len__()
    return self.data[idx]
# Sequence


if __name__ == "__main__":
  from torch.utils.data import DataLoader
  

# if __name__ == "__main__":