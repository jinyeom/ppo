from copy import deepcopy
import torch as pt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

class RolloutDataset(Dataset):
  def __init__(self, rollout, gamma, lam, alpha):
    self.obs = self._flatten_space_time(rollout.obs)
    self.action = self._flatten_space_time(rollout.action)
    self.logpi = self._flatten_space_time(rollout.logpi)
    self.vpred = self._flatten_space_time(rollout.vpred)
    self.ret = self._compute_return(rollout, gamma, lam, alpha)
    self.adv = self._compute_advantage()
  
  def _flatten_space_time(self, tensor):
    assert len(tensor.shape) > 2
    return tensor.view(-1, *tensor.shape[2:])
  
  def _compute_return(self, rollout, gamma, lam, alpha):    
    gae = 0.0
    next_vpred = rollout.appx['vpred'][0]
    returns = pt.zeros_like(rollout.reward)
    for i in reversed(range(rollout.size)):
      reward = rollout.reward[i]
      vpred = rollout.vpred[i]
      mask = rollout.mask[i]
      entropy = rollout.entropy[i]

      reward = reward + alpha * entropy
      delta = reward + mask * gamma * next_vpred - vpred
      gae = delta + mask * gamma * lam * gae
      returns[i] = gae + vpred
      next_vpred = vpred
    return self._flatten_space_time(returns)
      
  def _compute_advantage(self, normalize=True):
    adv = self.ret - self.vpred
    if normalize:
      adv = (adv - adv.mean()) / (adv.std() + 1e-5)
    return adv
    
  def __len__(self):
    return self.obs.size(0)
  
  def __getitem__(self, idx):
    return {
      'obs': self.obs[idx],
      'action': self.action[idx],
      'logpi': self.logpi[idx],
      'ret': self.ret[idx],
      'adv': self.adv[idx],
    }

class PPO:
  def __init__(
      self,
      agent,
      device,
      num_epochs=10,
      batch_size=64,
      lr_max=3e-4,
      lr_min=1e-4,
      eps=0.2,
      gamma=0.99,
      lam=0.95,
      alpha=0.2,
      value_coef=0.5,
      entropy_coef=0.0,
      max_grad_norm=0.5,
      target_kldiv=0.01,
  ):
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.eps = eps
    self.gamma = gamma
    self.lam = lam
    self.alpha = alpha
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.max_grad_norm = max_grad_norm
    self.target_kldiv = target_kldiv

    self.agent = agent
    self.optimizer = Adam(agent.parameters(), lr=lr_max)
    self.device = device

  def __repr__(self):
    return f'PPO({type(self.agent)}, {type(self.optimizer)})'
  
  def update_lr(self, step, total):
    lr = self.lr_max - ((self.lr_max - self.lr_min) * (step / total))
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr
    return lr

  def compute_losses(self, batch):
    obs = batch['obs'].to(self.device)
    action = batch['action'].to(self.device)
    old_logpi = batch['logpi'].to(self.device)
    adv = batch['adv'].to(self.device)
    ret = batch['ret'].to(self.device)
    
    pi, vpred = self.agent(obs)

    # Compute the policy loss for PPO (maximize).
    logpi = pi.log_prob(action).sum(-1, keepdim=True)
    ratio = pt.exp(logpi - old_logpi)
    term1 = ratio * adv
    term2 = pt.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv
    policy_loss = -pt.min(term1, term2).mean()
    
    # Compute the entropy regularizer (maximize).
    entropy = pi.entropy().mean()
    
    # Compute the value loss.
    value_loss = 0.5 * pt.mean((ret - vpred) ** 2)
    
    return policy_loss, entropy, value_loss

  def compute_kldiv(self, batch):
    obs = batch['obs'].to(self.device)
    action = batch['action'].to(self.device)
    old_logpi = batch['logpi'].to(self.device)
    with pt.no_grad():
      pi = self.agent.act(obs)
      logpi = pi.log_prob(action).sum(-1, keepdim=True)
    return pt.sum(old_logpi - logpi)

  def update(self, data):
    data_loader = DataLoader(
      RolloutDataset(data, self.gamma, self.lam, self.alpha),
      batch_size=self.batch_size,
      shuffle=True
    )

    ep_policy_loss = 0.0
    ep_entropy = 0.0
    ep_value_loss = 0.0
    ep_total_loss = 0.0
    
    for ep in range(self.num_epochs):
      ep_kldiv = 0.0
      for batch in data_loader:
        policy_loss, entropy, value_loss = self.compute_losses(batch)
        total_loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
      
        ep_policy_loss += policy_loss.item()
        ep_entropy += entropy.item()
        ep_value_loss += value_loss.item()
        ep_total_loss += total_loss.item()
        ep_kldiv += self.compute_kldiv(batch)

      ep_kldiv = ep_kldiv / (len(data_loader) * self.batch_size)
      if ep_kldiv.item() > self.target_kldiv:
        # Early stopping based on approximate mean KL-divergence.
        # NOTE: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        break

    num_steps = (ep + 1) * len(data_loader) * self.batch_size
    return {
      'num_epochs': (ep + 1),
      'policy_loss': ep_policy_loss / num_steps,
      'entropy': ep_entropy / num_steps,
      'value_loss': ep_value_loss / num_steps,
      'total_loss': ep_total_loss / num_steps
    }
