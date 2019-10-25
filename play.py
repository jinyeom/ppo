import gym
import numpy as np
import torch as pt
from agent import ContinuousPolicyAgent

def play(env, agent, device, repeat=1, render=False):
  perf = 0.0
  for i in range(repeat):
    done = False
    obs = env.reset()
    while not done:
      if render:
        env.render()
      with pt.no_grad():
        obs = pt.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
        action = agent.act(obs).sample().squeeze().cpu().numpy()
      obs, reward, done, _ = env.step(action)
      perf += reward
  return perf / repeat

def main(args):
  print("==== Creating a training environment...")
  env = gym.make('BipedalWalker-v2')
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  print("==== Creating an agent....")
  device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
  agent = ContinuousPolicyAgent(obs_dim, act_dim, args.hid_dim).to(device)
  agent.load_state_dict(pt.load(args.model_path))
  
  play(env, agent, device, render=True)
  env.close()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', type=str, default='model.pt')
  parser.add_argument('--hid-dim', type=int, default=64)
  args = parser.parse_args()
  main(args)
