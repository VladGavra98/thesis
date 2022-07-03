import numpy as np
import os
import time
import random
from core import agent
import torch
import argparse
from parameters import Parameters
import wandb
import envs.config


'''                           Globals                                                        '''
num_episodes = 200
num_frames = num_episodes * 2000

# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()

parser.add_argument('-should_log', help='Wether the WandB loggers are used', action='store_true')
parser.add_argument('-run_name', default='test', type=str)
parser.add_argument('-env', help='Environment Choices: (LunarLanderContinuous-v2) (PHLab)',type=str, default='PHlab_attitude')
parser.add_argument('-use_ddpg', help='Wether to use DDPG or TD3 for the RL part. Defaults to TD3',action='store_true', default=False)
parser.add_argument('-frames', help='Number of frames to learn from', default=num_frames, type=int)

parser.add_argument('-seed', help='Random seed to be used',type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-use_ounoise', help='Replace zero-mean Gaussian nosie with time-correletated OU noise', action='store_true')
parser.add_argument('-render', help='Render gym episodes', action='store_true')



parser.add_argument('-novelty', help='Use novelty exploration', action='store_true')
parser.add_argument('-proximal_mut', help='Use safe mutation',
                    action='store_true', default=True)
parser.add_argument('-distil', help='Use distilation crossover',
                    action='store_true', default=True)
parser.add_argument('-distil_type', help='Use distilation crossover. Choices: (fitness) (distance)',
                    type=str, default='distance')
parser.add_argument('-per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover',help='Make crossovers verbose', action='store_true')
parser.add_argument('-opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('-test_operators', help='Test the variational operators', action='store_true')

parser.add_argument('-sync_period', help="How often to sync to population", type=int)
parser.add_argument('-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('-next_save', help='Generation save frequency for save_periodic',
                    type=int, default=num_episodes//10)





if __name__ == "__main__":
    cla = parser.parse_args()

    # Inject the cla arguments in the parameters object
    parameters = Parameters(cla)

    # Create Env
    env = envs.config.select_env(cla.env)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    params_dict = parameters.write_params()

    # strat trackers
    if cla.should_log:
        print('\033[1;32m WandB logging started')
        run = wandb.init(project="pderl_phlab",
                        entity="vgavra",
                        dir='../logs',
                        name=cla.run_name,
                        config=params_dict)
        parameters.save_foldername = str(run.dir)
        wandb.config.update({"save_foldername": parameters.save_foldername,
                            "run_name": run.name}, allow_val_change=True)

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
    next_save = parameters.next_save
    start_time = time.time()
    
    while agent.num_frames <= parameters.num_frames:

        # evaluate over all episodes
        stats = agent.train()

        print('Epsiodes:', agent.num_episodes, 'Frames:', agent.num_frames,
              ' Train Max:', '%.2f' % stats['best_train_fitness'] if stats['best_train_fitness'] is not None else None,
              ' Test Max:', '%.2f' % stats['test_score'] if stats['test_score'] is not None else None,
              ' Test SD:', '%.2f' % stats['test_sd'] if stats['test_sd'] is not None else None,
              ' Population Avg:', '%.2f' % stats['pop_avg'],
              ' Weakest :', '%.2f' % stats['pop_min'],
              '\n',
              ' Avg. ep. len:', '%.2fs' % stats['avg_ep_len'],
              ' RL Reward:', '%.2f' % stats['rl_reward'],
              ' PG Objective:', '%.4f' % stats['PG_obj'],
              ' TD Loss:', '%.4f' % stats['TD_loss'],
              '\n')


        # Update loggers:
        stats['frames'] = agent.num_frames; stats['episodes'] = agent.num_episodes
        stats['time'] = time.time() - start_time
        stats['rl_elite_fraction'] = agent.evolver.selection_stats['elite'] / \
            agent.evolver.selection_stats['total']
        stats['rl_selected_fraction'] = agent.evolver.selection_stats['selected'] / \
            agent.evolver.selection_stats['total']
        stats['rl_discarded_fraction'] = agent.evolver.selection_stats['discarded'] / \
            agent.evolver.selection_stats['total']
        
        if cla.should_log:
            wandb.log(stats)                # main call to wandb logger

        # Get index of best actor
        elite_index = stats['elite_index']  # champion index

        # Save Policy
        if cla.should_log and agent.num_episodes > next_save:
            next_save += parameters.next_save
            agent.save_agent(parameters, elite_index)

    # Save final model:
    if cla.should_log:
        agent.save_agent(parameters, elite_index)

    if cla.should_log:
        run.finish()
