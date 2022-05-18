import numpy as np, os, time, random
from core import mod_utils as utils, agent
import gym, torch
import argparse
from parameters import Parameters

'''                           Globals                                                        '''
num_games = 5000
num_frames = num_games * 200

# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (Swimmer-v2) (LunarLanderContinuous-v2)', type=str, default = 'LunarLanderContinuous-v2')
# parser.add_argument('-num_games', help = 'Number of complete games to play', default = num_games)
parser.add_argument('-frames', help = 'Number of frames to learn from', default = num_frames)
#  QD equivalent of num_games: 50 000 games = 400 iters x 5 emitters x 25 batch_size
parser.add_argument('-seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
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
    parameters = Parameters(parser)  # Inject the cla arguments in the parameters object
    
    tracker = utils.Tracker(parameters, ['erl', 'erl_std'], '_games.csv')  # Initiate tracker
    frame_tracker = utils.Tracker(parameters, ['erl'], '_frames.csv')  # Initiate tracker
    time_tracker = utils.Tracker(parameters, ['erl'], '_time.csv')
    ddpg_tracker = utils.Tracker(parameters, ['ddpg_reward', 'ddpg_std'], '_games.csv')
    ddpg_frames = utils.Tracker(parameters, ['ddpg_reward'], '_frames.csv')
    selection_tracker = utils.Tracker(parameters, ['elite', 'selected', 'discarded'], '_selection.csv')

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))

    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(stdout=True)

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

        #retrieve statistics
        best_train_fitness = stats['best_train_fitness']
        erl_score = stats['test_score']
        erl_std = stats['test_sd']
        pop_avg = stats['pop_avg']
        elite_index = stats['elite_index']  #champion index
        ddpg_reward = stats['ddpg_reward']
        ddpg_std = stats['ddpg_std']
        policy_gradient_loss = stats['pg_loss']
        behaviour_cloning_loss = stats['bc_loss']
        population_novelty = stats['pop_novelty']

        print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
              ' Train_Max:', '%.2f'%best_train_fitness if best_train_fitness is not None else None,
              ' Test_Max:','%.2f'%erl_score if erl_score is not None else None,
              ' Test_SD:','%.2f'%erl_std if erl_std is not None else None,
              ' Population_Avg:', '%.2f'%pop_avg if pop_avg is not None else None,
              '\n'
              ' Avg:','%.2f'%tracker.all_tracker[0][1],
              ' DDPG Reward:', '%.2f'%ddpg_reward,
              ' PG Loss:', '%.4f' % policy_gradient_loss, '\n')

        elite = agent.evolver.selection_stats['elite']/agent.evolver.selection_stats['total']
        selected = agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']
        discarded = agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']

        # Update loggers:
        tracker.update([erl_score, erl_std], agent.num_games)
        frame_tracker.update([erl_score, erl_std], agent.num_frames)
        ddpg_tracker.update([ddpg_reward, ddpg_std], agent.num_games)
        ddpg_frames.update([ddpg_reward, ddpg_std], agent.num_frames)
        time_tracker.update([erl_score, erl_std], time.time()-time_start)
        selection_tracker.update([elite, selected, discarded], agent.num_frames)

        # Save Policy
        if agent.num_games > next_save:
            next_save += parameters.next_save
            save_agents(parameters, elite_index)

    # Save final model:
    save_agents(parameters, elite_index)

            











