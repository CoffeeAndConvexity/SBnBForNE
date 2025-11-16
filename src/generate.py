import numpy as np
import h5py
import argparse

def generate_utility_matrices(n_players, m_actions, util_range=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    shape = (n_players,) + (m_actions,) * n_players
    
    if util_range is None:
        utilities = np.random.uniform(-1, 1, size=shape)
    else:
        utilities = np.random.randint(util_range[0], util_range[1] + 1, size=shape)
    
    return utilities

def save_utility_matrices(filename, utilities):
    with h5py.File(filename, 'w') as f:
        f.create_dataset("utilities", data=utilities)


parser = argparse.ArgumentParser(description="generate multiplayer game.")
parser.add_argument('-n', '--n_players', type=int, default=2, help="Number of players.")
parser.add_argument('-m', '--m_actions', type=int, default=2, help="Number of actions per player.")
parser.add_argument('-ur', '--util_range', type=int, nargs=2, default=None, help="Range of utility values as two integers (min, max).")
parser.add_argument('-s', '--seed', type=int, default=0, help="Random seed for reproducibility.")
parser.add_argument('-of', '--output_file', type=str, default=None, help="Output filename to save the utility matrices.")

args = parser.parse_args()

n_players = args.n_players
m_actions = args.m_actions
util_range = tuple(args.util_range) if args.util_range != None else None
seed = args.seed

output_file = "games/" + f"utility_matrices_{n_players}_{m_actions}_{seed}.h5" if args.output_file == None else args.output_file

utilities = generate_utility_matrices(n_players, m_actions, util_range, seed)
save_utility_matrices(output_file, utilities)

# print(utilities[0][0][0])
# print(utilities)
