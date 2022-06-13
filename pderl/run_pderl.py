import numpy as np, os, time, random
from core import mod_utils as utils, agent
import gym, torch
import argparse
from parameters import Parameters
import wandb

'''                           Globals                                                        '''
num_games = 5000
num_frames = num_games * 200

# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()
parser.add_argument('-run_name', default = 'test', type = str)
parser.add_argument('-env', help='Environment Choices: (Swimmer-v2) (LunarLanderContinuous-v2)', type=str, default = 'LunarLanderContinuous-v2')
parser.add_argument('-use_ddpg', help='Wether to use DDPG or TD3 for the RL part. Defaults to TD3', action = 'store_true', default=False)
parser.add_argument('-frames', help = 'Number of frames to learn from', default = num_frames, type = int)
#  QD equivalent of num_games: 50 000 games = 400 iters x 5 emitters x 25 batch_size
parser.add_argument('-seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-use_ounoise', help='Replace zero-mean Gaussian nosie with time-correletated OU noise', action='store_true')
parser.add_argument('-render', help='Render gym episodes', action='store_true')
parser.add_argument('-sync_period', help="How often to sync to population", type=int)
parser.add_argument('-novelty', help='Use novelty exploration', action='store_true')
parser.add_argument('-proximal_mut', help='Use safe mutation', action='store_true', default=True)
parser.add_argument('-distil', help='Use distilation crossover', action='store_true', default = True)
parser.add_argument('-distil_type', help='Use distilation crossover. Choices: (fitness) (distance)',
                    type=str, default='distance')
parser.add_argument('-per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('-verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover', help='Make crossovers verbose', action='store_true')
parser.add_argument('-logdir', help='Folder where to save results', type=str, default = 'pderl/logs')
parser.add_argument('-opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('-opstat_freq', help='Frequency (in generations) to store operator statistics', type=int, default=1)
parser.add_argument('-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('-next_save', help='Generation save frequency for save_periodic', type=int, default=num_games//10)
parser.add_argument('-test_operators', help='Runs the operator runner to test the operators', action='store_true')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_agents(parameters : object, elite_index : int = None):
    """ Save the trained agents.

    Args:
        parameters (_type_): Container class of the trainign hyperparameters.
        elite_index (int: Index of the best performing agent i.e. the champion. Defaults to None.
    """    
    actors_dict = {}
    for i,ind in enumerate(agent.pop):
        actors_dict[f'actor_{i}'] = ind.actor.state_dict()
    torch.save(actors_dict, os.path.join(parameters.save_foldername,'evo_nets.pkl'))

    # Save best performing agent separately:
    if elite_index is not None:
        torch.save(agent.pop[elite_index].actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                   'elite_net.pkl'))

    print("Progress Saved")

if __name__ == "__main__":
    cla = parser.parse_args()
    parameters = Parameters(cla)  # Inject the cla arguments in the parameters object

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))

    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    params_dict = parameters.write_params(stdout=True)

    # strat trackers

    wandb.init(project="pderl_td3", entity="vgavra", name = cla.run_name,\
         config= params_dict)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)


    # Create Agent
    agent = agent.Agent(parameters, env)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    # Main training loop:
    next_save = parameters.next_save; time_start = time.time()
    while agent.num_frames <= parameters.num_frames:

        # evaluate over all games 
        stats = agent.train()

        print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
              ' Train Max:', '%.2f'%stats['best_train_fitness'] if stats['best_train_fitness'] is not None else None,
              ' Test Max:','%.2f'%stats['test_score'] if stats['test_score'] is not None else None,
              ' Test SD:','%.2f'%stats['test_sd'] if stats['test_sd'] is not None else None,
              ' Population Avg:', '%.2f'%stats['pop_avg'],
              ' Weakest :', '%.2f'%stats['pop_min'],
              '\n',
              ' RL Reward:', '%.2f'%stats['rl_reward'],
              ' PG Objective:', '%.4f' % stats['PG_obj'], 
              ' TD Loss:', '%.4f' % stats['TD_loss'],
              '\n')


        # Update loggers:
        stats['frames'] = agent.num_frames; stats['games']= agent.num_games
        stats['elite_fraction'] = agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']
        stats['selected_fraction'] = agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']
        stats['discarded_fraction'] = agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']
        wandb.log(stats)  # main call to wandb logger

        # Get index of best actor
        elite_index = stats['elite_index']  #champion index

        # Save Policy
        if agent.num_games > next_save:
            next_save += parameters.next_save
            save_agents(parameters, elite_index)

    # Save final model:
    save_agents(parameters, elite_index)

            











