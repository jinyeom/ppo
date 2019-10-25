import torch as pt
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class DeepMindAtari(nn.Module):
  def __init__(self, hid_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
    self.fc4 = nn.Linear(32 * 7 * 7, hid_dim)

  def forward(self, x):
    x = F.relu(self.conv1(x), inplace=True)
    x = F.relu(self.conv2(x), inplace=True)
    x = F.relu(self.conv3(x), inplace=True)
    x = pt.flatten(x, start_dim=1)
    return F.relu(self.fc4(x), inplace=True)

class ActorCritic(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim, num_layers=3):
    super().__init__()
    self.actor = self._build_mlp(obs_dim, act_dim, hid_dim, num_layers)
    self.critic = self._build_mlp(obs_dim, 1, hid_dim, num_layers)
    
  def _build_mlp(self, input_size, output_size, hidden_size, num_layers):
    if num_layers == 1:
      return nn.Linear(input_size, output_size)
    layers = [
      nn.Linear(input_size, hidden_size),
      nn.ReLU(inplace=True)
    ]
    layers += (num_layers - 2) * [
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(inplace=True)
    ]
    layers += [nn.Linear(hidden_size, output_size)]
    return nn.Sequential(*layers)

  def forward(self, obs):
    return self.act(obs), self.evaluate(obs)
    
  def act(self, obs):
    return self.actor(obs)
  
  def evaluate(self, obs):
    return self.critic(obs)

