import torch as pt
from torch import nn
from torch.distributions import Normal
from modules import ActorCritic

class ContinuousPolicyAgent(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim):
    super().__init__()
    self.ac = ActorCritic(obs_dim, act_dim, hid_dim)
    self.logstd = nn.Parameter(pt.zeros(act_dim))

  def forward(self, obs):
    return self.act(obs), self.evaluate(obs)

  def act(self, obs):
    loc = pt.tanh(self.ac.act(obs))
    scale = pt.exp(self.logstd)
    return Normal(loc, scale)

  def evaluate(self, obs):
    return self.ac.evaluate(obs)

  def rollout(self, obs, env, data):
    data.reset()

    while not data.full():
      with pt.no_grad():
        pi, vpred = self(obs)
        action = pi.sample()
        logpi = pi.log_prob(action).sum(-1, keepdim=True)
        entropy = pi.entropy().sum(-1, keepdim=True)
      
      next_obs, reward, done, _ = env.step(action)
      mask = 1 - done

      # Record the rollout data.
      data.append(
        obs=obs,
        action=action,
        reward=reward,
        mask=mask,
        vpred=vpred,
        logpi=logpi,
        entropy=entropy
      )
      
      # Update the next observation.
      obs = next_obs.to(obs.device)
    
    # Compute the last state value of the last observation.
    # This will be used to compute GAE.
    with pt.no_grad():
      vpred = self.evaluate(obs)
    data.append(vpred=vpred)
    
    # Return the next observation.
    return obs
