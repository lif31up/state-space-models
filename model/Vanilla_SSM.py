from torch import nn

class Vanilla_SSM(nn.Module):
  def __init__(self, config):
    super(Vanilla_SSM, self).__init__()
    self.A = nn.Linear(in_features=config.A_channels, out_features=config.A_channels, bias=config.bias)
    self.B = nn.Linear(in_features=config.B_channels, out_features=config.B_channels, bias=config.bias)
    self.C = nn.Linear(in_features=config.C_channels, out_features=config.C_channels, bias=config.bias)
    self.D = nn.Linear(in_features=config.D_channels, out_features=config.D_channels, bias=config.bias)
  # __init__

  def process(self, h, x): return self.A(h) + self.B(x)
  def measurement(self, h, x): return self.C(h) + self.D(x)

  def forward(self, x, h):
    y = self.measurement(x, h)
    h  = self.forward(x, h)
    return y, h
  # forward

  def recurrent(self, sequence, device):
    assert sequence.__len__ is not None, "Vanilla_SSM.recurrent: the given sequence should be iterable."
    iv_state = self.config.dummy # later
    episode_length, outputs = len(sequence), list()
    state_buffer = self.forward(x=sequence[0], h=iv_state)
    for (feature, label), index in enumerate(sequence[1:]):
      feature, label = feature.to(device), label.to(device)
      if index != episode_length - 1: state_buffer = self.forward(x=feature, h=state_buffer)
      output, state_buffer = self.forward(x=feature, h=state_buffer)
      outputs.append(output)
    return outputs
  # recurrent
# Vanilla_SSM

if __name__ == "__main__":
  from config import Config
  ssm_config = Config()
  ssm = Vanilla_SSM(ssm_config)
# if __name__ == "__main__":