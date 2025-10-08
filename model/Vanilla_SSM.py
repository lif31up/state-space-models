import torch
from torch import nn

class Vanilla_SSM(nn.Module):
  def __init__(self, config):
    super(Vanilla_SSM, self).__init__()
    self.mode = self.config.mode
    self.A = nn.Linear(in_features=config.A_channels, out_features=config.A_channels, bias=config.bias)
    self.B = nn.Linear(in_features=config.B_channels, out_features=config.B_channels, bias=config.bias)
    self.C = nn.Linear(in_features=config.C_channels, out_features=config.C_channels, bias=config.bias)
    self.D = nn.Linear(in_features=config.D_channels, out_features=config.D_channels, bias=config.bias)
  # __init__

  def process(self, h, x): return self.A(h) + self.B(x)
  def measurement(self, h, x): return self.C(h) + self.D(x)

  def step(self, x, h):
    y = self.measurement(x, h)
    h  = self.step(x, h)
    return y, h
  # step

  def forward(self, sequence):
    if self.mode: return self.recurrent_sequential_output(sequence)
    return self.recurrent_no_sequential_output(sequence)
  # forward

  def recurrent_no_sequential_output(self, sequence): # do not implement parallel scan
    iv_state = self.config.dummy  # later
    episode_length, output = len(sequence), None
    state_buffer = self.step(x=sequence[0], h=iv_state)
    for x_k, index in enumerate(sequence[1:]):
      if index != episode_length - 1: state_buffer = self.step(x=x_k, h=state_buffer)
      output, state_buffer = self.step(x=x_k, h=state_buffer)
    return output
  # recurrent_no_sequential_output

  def recurrent_sequential_output(self, sequence):
    iv_state = self.config.dummy  # later
    episode_length, outputs = len(sequence), list()
    state_buffer = self.step(x=sequence[0], h=iv_state)
    for x_k, index in enumerate(sequence[1:]):
      if index != episode_length - 1: state_buffer = self.step(x=feature, h=state_buffer)
      output, state_buffer = self.step(x=x_k, h=state_buffer)
      outputs.append(output.item())
    return torch.stack(outputs)
  # recurrent_sequential_output
# Vanilla_SSM

if __name__ == "__main__":
  from config import Config
  ssm_config = Config()
  ssm = Vanilla_SSM(ssm_config)
# if __name__ == "__main__":