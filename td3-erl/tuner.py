import numpy as np
import time
import random
from core import agent
import torch
import argparse
from parameters import Parameters
import wandb
import envs.config


'''                           Globals                                                        '''
num_episodes = 250
num_frames = num_episodes * 2000

# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()


parser.add_argument('-run_name', default='test', type=str)
parser.add_argument('-gamma', type = float)
parser.add_argument('-lr', type = float)
parser.add_argument('-num_layers', type = int)
parser.add_argument('-hidden_size', type = int)
parser.add_argument('-buffer_size', type = int)
parser.add_argument('-batch_size', type = int)

parser.add_argument('-noise_sd', type = float)


if __name__ == "__main__":
    cla = parser.parse_args()

    # Inject the cla arguments in the parameters object
    parameters = Parameters(cla)

    # Create Env
    env = envs.config.select_env(parameters.env_name)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    params_dict = parameters.write_params()

    # strat trackers
    print('\033[1;32m WandB logging started')
    run = wandb.init(project="sweeps-td3",
                    entity="vgavra",
                    dir='../logs',
                    config=params_dict)
    parameters.save_foldername = str(run.dir)
    # wandb.config.update({"run_name": run.name}, allow_val_change=True)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Print run paramters for sanity cheks
    parameters.write_params(stdout=True)

    # Create Agent
    agent = agent.Agent(parameters, env)
    print('Running', parameters.env_name, ' State_dim:',
          parameters.state_dim, ' Action_dim:', parameters.action_dim)

    # Main training loop:
    start_time = time.time()
    
    while agent.num_frames <= parameters.num_frames:

        # evaluate over all episodes
        stats = agent.train()

        # print('Epsiodes:', agent.num_episodes, 'Frames:', agent.num_frames,
        #       ' Train Max:', '%.2f' % stats['best_train_fitness'] if stats['best_train_fitness'] is not None else None,
        #       ' Test Max:', '%.2f' % stats['test_score'] if stats['test_score'] is not None else None,
        #       ' Test SD:', '%.2f' % stats['test_sd'] if stats['test_sd'] is not None else None,
        #       ' Population Avg:', '%.2f' % stats['pop_avg'],
        #       ' Weakest :', '%.2f' % stats['pop_min'],
        #       '\n',
        #       ' Avg. ep. len:', '%.2fs' % stats['avg_ep_len'],
        #       ' RL Reward:', '%.2f' % stats['rl_reward'],
        #       ' PG Objective:', '%.4f' % stats['PG_obj'],
        #       ' TD Loss:', '%.4f' % stats['TD_loss'],
        #       '\n')


        # Update loggers:
        stats['frames'] = agent.num_frames; stats['episodes'] = agent.num_episodes
        stats['time'] = time.time() - start_time


        wandb.log(stats)

    run.finish()
