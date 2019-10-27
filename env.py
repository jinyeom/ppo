import numpy as np
import torch as pt
import gym

class ChannelFirst(gym.ObservationWrapper):
  def observation(self, observation):
    return np.moveaxis(observation, -1, 0)

class NormalizeObservation(gym.ObservationWrapper):
  def observation(self, observation):
    mean = np.mean(observation)
    std = np.std(observation)
    return (observation - mean) / (std + 1e-8)

class TorchObservation(gym.ObservationWrapper):
  def observation(self, observation):
    observation = observation.astype(np.float32)
    return pt.from_numpy(observation)

class TorchAction(gym.ActionWrapper):
  def action(self, action):
    action = action.squeeze()
    return action.cpu().numpy()

class TorchReward(gym.RewardWrapper):
  def reward(self, reward):
    reward = reward.astype(np.float32)
    reward = reward[:, np.newaxis]
    return pt.from_numpy(reward)

class TorchDone(gym.Wrapper):
  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    return observation, reward, self.done(done), info
  
  def done(self, done):
    done = done.astype(np.float32)
    done = done[:, np.newaxis]
    return pt.from_numpy(done)
  
def make_env(env_id, *wrappers, num_envs=1):
  env = gym.vector.make(
    env_id, 
    num_envs=num_envs, 
    wrappers=wrappers
  )
  env = TorchObservation(env)
  env = TorchAction(env)
  env = TorchReward(env)
  env = TorchDone(env)
  return env
