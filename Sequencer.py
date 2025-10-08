class Sequencer:
  def __init__(self, config, dataset):
    self.config = config
    self.dataset = self._dataset_to_sequence(dataset)
    self.dataset_len = len(self.dataset)
  # __init__

  def __len__(self): return self.dataset_len

  def _dataset_to_sequence(self, dataset):
    sequences = list()
    for sequence in dataset: sequences.append(sequence)
    return sequences
  # _dataset_to_sequence

  def __getitem__(self, idx):
    assert idx < self.dataset.__len__(), "the given idx is out of bound."
    return self.dataset[idx]
  # __getitem__
# Sequencer

if __name__ == "__main__":
  from torch.utils.data import DataLoader
  from config import Config

  features = pd.read_csv(os.path.join(DATASET_PATH, "features.txt"), sep="\s+", header=None)
  feature_names = features[1].values

  config = Config(download=False)
  sequencer = Sequencer(dataset=dataset, config=config)
  for feature, label in DataLoader(dataset=sequencer, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True):
    print(f'feature: {feature.shape} | label: {label.shape}')

# if __name__ == "__main__":