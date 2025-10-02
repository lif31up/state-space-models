class Config:
  def __init__(self):
    self.episode_length = 10

    # HIPPO initiation
    self.A_channels = 64
    self.B_channels = 64
    self.C_channels = 64
    self.D_channels= 64

    # hypo
    self.bias = True
  # __init__

# Config

if __name__ == "__main__":
  config = Config()
  print(config)
# if __name__ == "__main__":