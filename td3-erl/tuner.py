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
# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()


parser.add_argument('--run_name', default='test', type=str)
parser.add_argument('--should_log', action='store_true')
parser.add_argument('--next_save', default=100, type=int)
parser.add_argument('--frames', default=800000, type=int)
parser.add_argument('--gamma', default = 0.99, type = float)
parser.add_argument('--lr', default =0.00045, type = float)
parser.add_argument('--num_layers', default = 3, type = int)
parser.add_argument('--hidden_size', default =64,  type = int)
parser.add_argument('--buffer_size', default = 100_000, type = int)
parser.add_argument('--batch_size',  default =64, type = int)
parser.add_argument('--activation_actor', default = 'relu', type = str)
parser.add_argument('--noise_sd', default =0.32, type = float)

parser.add_argument('--use_caps', action = 'store_true')

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
    run = wandb.init(project="phlab",
                    entity="vgavra",
                    dir='../logs',
                    name = cla.run_name,
                    config = params_dict)
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
    next_save = parameters.next_save
    
    while agent.num_frames <= parameters.num_frames:

        # evaluate over all episodes
        stats = agent.train()

        # Update loggers:
        stats['frames'] = agent.num_frames; stats['episodes'] = agent.num_episodes
        stats['time'] = time.time() - start_time


        wandb.log(stats)

        if parameters.should_log and agent.num_episodes > next_save:
            # elite_index = stats['elite_index']  # champion index
            next_save += parameters.next_save
            agent.save_agent(parameters)

    if parameters.should_log:
        agent.save_agent(parameters)
    
    run.finish()
