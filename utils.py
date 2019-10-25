import os
import json
from time import ctime

def pretty_args(args):
  """Print pretty arguments."""
  n = max(len(k) for k in args.__dict__.keys())
  for k, v in args.__dict__.items():
    print(str(k) + (n-len(k)+1)*' ' + str(v))
    
def export_args(args, path):
  """Export arguments as a JSON file."""
  args_json = json.dumps(args.__dict__, indent=4, sort_keys=True)
  with open(path, 'w') as f:
    f.write(args_json)

def mkdir_exp(name):
  """Create a root directory for a single run of experiment."""
  now_str = ctime().replace(' ', '_')
  exp_run_path = os.path.join(os.getcwd(), f'run_{name}_{now_str}')
  if not os.path.isdir(exp_run_path):
    os.makedirs(exp_run_path)
  return exp_run_path
