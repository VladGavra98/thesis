import numpy as np, os, time, random
from core import mod_utils as utils, agent
import gym, torch
import argparse
from parameters import Parameters

num_games = 10
num_frames = num_games * 200

parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (Swimmer-v2) (LunarLanderContinuous-v2)', type=str, default = 'LunarLanderContinuous-v2')
parser.add_argument('-num_games', help = 'Number of complete games to play', default = 10)
# parser.add_argument('-num_games', help = 'Number of complete games to play', default = num_games)
parser.add_argument('-num_frames', help = 'Number of frames to learn from', default = num_frames)
parser.add_argument('-seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-render', help='Render gym episodes', action='store_true')
parser.add_argument('-sync_period', help="How often to sync to population", type=int)
parser.add_argument('-novelty', help='Use novelty exploration', action='store_true')
parser.add_argument('-proximal_mut', help='Use safe mutation', action='store_true')
parser.add_argument('-distil', help='Use distilation crossover', action='store_true')
parser.add_argument('-distil_type', help='Use distilation crossover. Choices: (fitness) (distance)',
                    type=str, default='fitness')
parser.add_argument('-per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('-verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover', help='Make crossovers verbose', action='store_true')
parser.add_argument('-logdir', help='Folder where to save results', type=str, default = 'pderl/logs_ddpg')
parser.add_argument('-opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('-opstat_freq', help='Frequency (in generations) to store operator statistics', type=int, default=1)
parser.add_argument('-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('-next_save', help='Generation save frequency for save_periodic', type=int, default=num_games//10)
parser.add_argument('-test_operators', help='Runs the operator runner to test the operators', action='store_true')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))



if __name__ == "__main__":
    parameters = Parameters(parser)  # Inject the cla arguments in the parameters object

    time_tracker = utils.Tracker(parameters, ['ddpg_reward', 'ddpg_std'], '_time.csv')
    ddpg_tracker = utils.Tracker(parameters, ['ddpg_reward', 'ddpg_std'], '_games.csv')
    ddpg_frames = utils.Tracker(parameters, ['ddpg_reward', 'ddpg_std'], '_frames.csv')
   
    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))

    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(stdout=False)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Create Agent
    agent = agent.Agent_ddpg(parameters, env)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)

    # Main training loop:
    next_save = parameters.next_save; time_start = time.time()
    while agent.num_games <= parameters.num_games:

        # evaluate over all games 
        stats = agent.train()

        if stats:
            #retrieve statistics
            ddpg_reward = stats['ddpg_reward']
            ddpg_std = stats['ddpg_std']
            policy_gradient_loss = stats['pg_loss']

            print('#Games:', agent.num_games, '#Frames:', agent.num_frames,
                ' DDPG Reward:', '%.2f'%ddpg_reward,
                ' DDPG STD:', '%.2f'%ddpg_std,
                ' PG Loss:', '%.4f' % policy_gradient_loss, '\n')

            # Update loggers:
            ddpg_tracker.update([ddpg_reward, ddpg_std], agent.num_games)
            ddpg_frames.update([ddpg_reward, ddpg_std], agent.num_frames)
            time_tracker.update([ddpg_reward, ddpg_std], time.time()-time_start)


        # Save Policy
        if agent.num_games > next_save:
            next_save += parameters.next_save
            torch.save(agent.rl_agent.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                   'ddpg_net.pkl'))

            print("Progress Saved")

    torch.save(agent.rl_agent.actor.state_dict(), os.path.join(parameters.save_foldername,
                                                                                   'ddpg_net.pkl'))











