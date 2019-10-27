import os
import gym
import torch as pt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from tensorbook import TensorBook
from utils import pretty_args, export_args, mkdir_exp
from env import make_env, NormalizeObservation
from agent import ContinuousPolicyAgent
from ppo import PPO
from play import play

def main(args):
  exp_path = mkdir_exp(f'{args.env_id}_PPO')
  export_args(args, os.path.join(exp_path, 'config.json'))

  np.random.seed(args.seed)
  pt.random.manual_seed(args.seed)

  print("==== Creating a training environment...")
  env = make_env(
    args.env_id,
    NormalizeObservation,
    num_envs=args.num_envs
  )

  print("==== Creating a evaluation environment...")
  eval_env = gym.make(args.env_id)
  obs_dim = eval_env.observation_space.shape[0]
  act_dim = eval_env.action_space.shape[0]
  
  print("==== Creating an agent....")
  device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
  agent = ContinuousPolicyAgent(obs_dim, act_dim, args.hid_dim).to(device)
  
  print("==== Creating a data storage...")
  data = TensorBook(args.env_id, args.rollout_steps)
  
  print("==== Creating a PPO optimizer...")
  optimizer = PPO(
    agent, 
    device,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    eps=args.eps,
    gamma=args.gamma,
    lam=args.lam,
    alpha=args.alpha,
    value_coef=args.value_coef,
    entropy_coef=args.entropy_coef,
    max_grad_norm=args.max_grad_norm,
    target_kldiv=args.target_kldiv
  )

  print("==== Creating a TensorBoard summary writer...")
  writer = SummaryWriter(log_dir=exp_path)
  
  print("IT'S DANGEROUS TO GO ALONE! TAKE THIS.")
  obs = env.reset().to(device)
  best_perf = -np.inf
  lr = args.lr

  num_updates = args.num_steps // args.rollout_steps // args.num_envs
  for i in tqdm(range(num_updates)):
    obs = agent.rollout(obs, env, data)
    info = optimizer.update(data)
    if args.lr_decay:
      lr = optimizer.update_lr(i, num_updates)
    
    # Evaluate the agent.
    perf = play(eval_env, agent, device, repeat=args.num_eval)
    if perf > best_perf:
      model_path = os.path.join(exp_path, f'{agent.__class__.__name__}.pt')
      pt.save(agent.state_dict(), model_path)
      best_perf = perf

    # Log training progress.
    step = i * args.rollout_steps * args.num_envs
    writer.add_scalar('Loss/train/policy', info['policy_loss'], step)
    writer.add_scalar('Loss/train/value', info['value_loss'], step)
    writer.add_scalar('Loss/train/entropy', info['entropy'], step)
    writer.add_scalar('Loss/train/total', info['total_loss'], step)
    writer.add_scalar('Reward/eval/mean', perf, step)
    writer.add_scalar('Reward/eval/best', best_perf, step)
    writer.add_scalar('LR/train/lr', lr, step)

  env.close()
  eval_env.close()
  writer.close()
  
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env-id', type=str, default='BipedalWalker-v2')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num-envs', type=int, default=16)
  parser.add_argument('--hid-dim', type=int, default=64)
  parser.add_argument('--num-steps', type=int, default=1000000)
  parser.add_argument('--rollout-steps', type=int, default=2048)
  parser.add_argument('--num-epochs', type=int, default=10)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--eps', type=float, default=0.2)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--lam', type=float, default=0.95)
  parser.add_argument('--alpha', type=float, default=0.0)
  parser.add_argument('--value-coef', type=float, default=0.5)
  parser.add_argument('--entropy-coef', type=float, default=0.0)
  parser.add_argument('--max-grad-norm', type=float, default=0.5)
  parser.add_argument('--target-kldiv', type=float, default=0.02)
  parser.add_argument('--num-eval', type=int, default=5)
  parser.add_argument('--lr-decay', action='store_true', default=False)
  args = parser.parse_args(); pretty_args(args)
  main(args)
