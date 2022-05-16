import numpy as np
import os
import random
from parameters import Parameters
from core import mod_utils as utils
from core.ddpg import GeneticAgent
import torch
import gym
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# my envs
from envs.lunarlander import simulate

parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choice',
                    type=str, default='LunarLanderContinuous-v2')
parser.add_argument('-seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument('-render', help='Render gym episodes',
                    action='store_true', default=False)
parser.add_argument('-load_champion', help='Loads the best performingactor',
                    action='store_true', default=False)
args = parser.parse_args()

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12})


def evaluate(agent, env, trials: int = 10, render: bool = False, kwargs : dict = None):
    """ Evaualte an individual for a couple of trails/games.

    Args:
        agent (_type_): Actor to evalaute.
        env (_type_): Environment for testing. Should be the same API as gym environments.
        trials (int, optional): Number of evaluation runs. Defaults to 10.
        render (bool, optional): Show policy in a video. Defaults to False.

    Returns:
        tuple: average reward, reward standard deviation, average behaviour characteristics
    """
    rewards, bcs = [], []

    for _ in range(trials):
        total_reward, impact_x_pos, impact_y_vel = simulate(agent, env, render, **kwargs)

        rewards.append(total_reward)
        bcs.append((impact_x_pos, impact_y_vel))

    bcs = np.asarray(bcs)
    return np.average(rewards), np.std(rewards), np.average(bcs, axis=0)




def load_genetic_agent(args, model_path: str, elite_path: str = None):
    actor_path = os.path.join(model_path)
    agents_pop = []
    checkpoint = torch.load(actor_path)

    for _, model in checkpoint.items():
        agent = GeneticAgent(args)
        agent.actor.load_state_dict(model)
        agents_pop.append(agent)

    if elite_path:
        agent.actor.load_state_dict(torch.load(elite_path))
    print("Model loaded from: " + model_path)

    return agents_pop


def load_rl_agent(args, model_path: str = 'ddpg/logs/evo_net.pkl'):
    actor_path = os.path.join(model_path)

    agent = GeneticAgent(args)
    agent.actor.load_state_dict(torch.load(actor_path))

    print("Model loaded from: " + model_path)

    return agent


def gen_heatmap(bcs_map: np.ndarray, rewards: np.ndarray, filename: str, save_figure: bool = False):
    """Saves a heatmap of the optimizer's archive to the filename.

    Args:
        bcs_map
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.scatter(bcs_map[:, 0], bcs_map[:, 1],
                     c=rewards, marker='s', cmap='magma')
    # fig.suptitle('Archive Illumiantion')
    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel(r"Impact $\dot{y}$")
    ax.set_xlabel(r"Impact $x$")

    cbar = fig.colorbar(img, orientation='vertical')
    cbar.ax.set_title('Mean Reward')

    plt.show()

    if save_figure:
        fig.savefig(filename)
        print('Figured saved.')


if __name__ == "__main__":
    env = utils.NormalizedActions(gym.make('LunarLanderContinuous-v2'))

    # Global paths:
    model_path = 'pderl/logs_s1_e3_buffer5e04/evo_nets.pkl'
    elite_path = 'pderl/logs_s1_e3_buffer5e04/elite_net.pkl'
    ddpg_path = 'pderl/logs_ddpg/ddpg_net.pkl'

    # Set parameters:
    parameters = Parameters(args, init=False)
    parameters.individual_bs = 0
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.use_ln = True
    parameters.device = torch.device('cuda')
    setattr(parameters, 'ls', 32)

    # Seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Evaluation params:
    num_trials = 2
    broken_engine= False
    state_noise = True
    noise_intensity = 0.05
    extra_args = {'broken_engine' : False, 'state_noise' : True, 'noise_intensity': noise_intensity}
    # ------------------------------------------------------------------------
    #                                Elite agent
    # -> evalaute the best perforing controller on the nominal system
    # ------------------------------------------------------------------------
    elite_agent = load_rl_agent(parameters, elite_path)
    reward_mean, reward_std, bcs = evaluate(elite_agent.actor, env,
                render=args.render, trials=num_trials, kwargs = extra_args)

    print(f'Elite:{reward_mean:.2f}, with SD = {reward_std:.2f}\n')

    # --------------------------------------------------------------------------
    #                            ERL Population
    # -> evaluate entire popualtion on the faulty system
    # --------------------------------------------------------------------------
    agents_pop = load_genetic_agent(parameters, model_path, elite_path)
    rewards, bcs_map, rewards_std = [], [], []
    for agent in tqdm(agents_pop):  # evaluate each member for # trials
        r_mean, r_std, bcs = evaluate(agent.actor, env,
                render=args.render, trials=num_trials,\
                     kwargs = extra_args)
        rewards.append(r_mean)
        bcs_map.append(bcs)
        rewards_std.append(r_std)

    rewards = np.asarray(rewards)
    bcs_map = np.asarray(bcs_map)
    rewards_std = np.asarray(rewards_std)
    new_elite = np.argmax(rewards)

    print(f'New elite: {rewards[new_elite]:.2f}, with SD = {rewards_std[new_elite]:.2f}\n')


    # ------------------------------------------------------------------------
    #                                RL agent
    # ------------------------------------------------------------------------
    rl_agent = load_rl_agent(parameters, ddpg_path)
    reward_mean, reward_std, bcs = evaluate(rl_agent.actor, env,
                render=args.render, trials=num_trials,\
                   kwargs = extra_args)

    print(f'RL (ddpg):{reward_mean:.2f}, with SD = {reward_std:.2f}\n')

    # -----------------------------------------------------------------------
    #                                   Plotting
    # ------------------------------------------------------------------------
    # gen_heatmap(bcs_map, rewards, 
    #           filename='Results_pderl/Plots/population_map.png')
    # gen_heatmap(bcs_map, rewards,
    #             filename='Results_pderl/Plots/population_map_broeknengine.png',\
    #             save_figure = False)
