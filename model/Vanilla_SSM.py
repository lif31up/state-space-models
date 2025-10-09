import torch
from torch import nn


class Vanilla_SSM(nn.Module):
  def __init__(self, config):
    super(Vanilla_SSM, self).__init__()
    self.config = config
    self.mode = self.config.mode

    self.A = linear_apply(self.config.A)
    self.B = linear_apply(self.config.B)
    self.C = linear_apply(self.config.C)
    self.D = linear_apply(self.config.D)

    self.deltas = nn.ModuleList()
    self.deltas.append(nn.Linear(in_features=config.delta.in_features, out_features=config.delta.hid_features))
    for i in range(1, self.config.episode_length - 1):
      self.deltas.append(nn.Linear(in_features=config.delta.hid_features, out_features=config.delta.hid_features))
    self.deltas.append(nn.Linear(in_features=config.delta.hid_features, out_features=config.delta.out_features))
  # __init__

  def process(self, h, x, k):
    a = self.A(h)
    a_d = self.deltas[k](a)
    b_k = self.B(x)
    delta_k_b_k = self.deltas[k](b_k)
    b_d = (a_d ** -1) * (a_d - I) * delta_k_b_k
    return torch.exp(a_d) + b_d
  # process
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

  def recurrent_no_sequential_output_with_discretization(self, sequence):
    iv_state = self.config.dummy  # later
    episode_length, outputs = len(sequence), list()
    state_buffer = self.step(x=sequence[0], h=iv_state)
    for x_k, index in enumerate(sequence[1:]):
      if index != episode_length - 1: state_buffer = self.step(x=feature, h=state_buffer)
      output, state_buffer = self.step(x=x_k, h=state_buffer)
      outputs.append(output.item())
    return torch.stack(outputs)
  # discretization
# Vanilla_SSM


if __name__ == "__main__":
  from config import Config, linear_apply

  ssm_config = Config()
  ssm = Vanilla_SSM(ssm_config)
# if __name__ == "__main__":