import numpy as np
import os
import random
from parameters import Parameters
from core.genetic_agent import GeneticAgent
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path, PurePath

# my envs
from envs.lunarlander import LunarLanderWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choice',
                    type=str, default='LunarLanderContinuous-v2')
parser.add_argument('-seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument('-render', help='Render gym episodes',
                    action='store_true', default=False)
parser.add_argument('-load_champion', help='Loads the best-performing actor',
                    action='store_true', default=False)
args = parser.parse_args()

style = 'ggplot'
plt.style.use(style.lower()) 
plt.rcParams.update({'font.size': 12})
# plt.rcParams['figure.dpi'] = 200


# Global paths:
wrapper = LunarLanderWrapper()
model_path = PurePath('logs/logs_TD3test/evo_nets.pkl')
elite_path = PurePath('logs/logs_TD3test/elite_net.pkl')


def evaluate(agent, wrapper, trials: int = 10, render: bool = False, kwargs : dict = None):
    """ Evaluate performance statistics of one individual over nubmer of trails/games.

    Args:
        agent (_type_): Actor to evalaute.
        simualtee_func (executable): 
        trials (int, optional): Number of evaluation runs. Defaults to 10.
        render (bool, optional): Show policy in a video. Defaults to False.

    Returns:
        tuple: average reward, reward standard deviation, average behaviour characteristics
    """
    rewards, bcs = [], []

    for _ in range(trials):
        episode_dict = wrapper.simulate(agent, render, **kwargs)

        rewards.append(episode_dict['total_reward'])
        bcs.append(episode_dict['bcs'])

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


def gen_heatmap(bcs_map: np.ndarray, rewards: np.ndarray, filename: str, save_figure: bool = False, name : str = None):
    """Saves a heatmap of the optimizer's archive to the filename.

    Args:
        bcs_map
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.scatter(bcs_map[:, 0], bcs_map[:, 1],
                     c=rewards, marker='s', cmap='magma')

    if name is not None:
        fig.suptitle('Archive: ' + str(name))

    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel(r"Impact $\dot{y}$")
    ax.set_xlabel(r"Impact $x$")

    cbar = fig.colorbar(img, orientation='vertical')
    cbar.ax.set_title('Mean Reward')
    plt.tight_layout()

    if save_figure:
        fig.savefig(filename)
        print('Figured saved.')


def _extract_case(case : str, plotfolder : Path = PurePath('Results')) -> tuple:
    """Translate case into simulation paramaters:

    Args:
        case (str): Identifier to case to evaluate.

    Returns:
        tuple: argumetns to be passed to environment, name for plottign function
    """    

    case = str(case)
    filename = plotfolder / PurePath('/Plots/population_map.png')

    if 'nominal' in case.lower():
        print('Current case: nominal')
        extra_args = {'broken_engine' : False, 'state_noise' : False}
        plotname = 'nominal'
        filename = plotfolder / PurePath('/map.png')

    elif case.lower().find('nois') !=-1:
        print('Current case: faulty system - noisy state')
        extra_args = {'broken_engine' : False, 'state_noise' : True, 'noise_intensity': 0.05}
        plotname = 'noisy state (F1)'
        filename = plotfolder / PurePath('/map_noisystate.png')

    elif case.lower().find('broken') != -1:
        print('Current case: faulty system - broken engine')
        extra_args = {'broken_engine' : True, 'state_noise' : False}
        plotname = 'broken engine (F2)'
        filename = plotfolder / PurePath('/map_brokenengine.png')

    else:
        Warning('No case provided for evaluation!')

    return extra_args, plotname, filename


def gen_comparative_map(parameters, wrapper, num_trials : int, case : str, save_figure : bool = False):
    # NOTE needs refractoring
    # generate a comparative map based o nthe case
    extra_args, plotname,filename = _extract_case('nominal')  

    agents_pop = load_genetic_agent(parameters, str(model_path), str(elite_path))
    rewards, bcs_map, rewards_std = [], [], []
    for agent in tqdm(agents_pop):  # evaluate each member for # trials
        r_mean, r_std, bcs = evaluate(agent.actor, wrapper,
                render=args.render, trials=num_trials,\
                     kwargs = extra_args)
        rewards.append(r_mean)
        bcs_map.append(bcs)
        rewards_std.append(r_std)

    bcs_map_old = np.asarray(bcs_map)

    extra_args, plotname, filename = _extract_case(case)  
    agents_pop = load_genetic_agent(parameters, str(model_path), str(elite_path))
    rewards, bcs_map, rewards_std = [], [], []
    for agent in tqdm(agents_pop):  # evaluate each member for # trials
        r_mean, r_std, bcs = evaluate(agent.actor, wrapper,
                render=args.render, trials=num_trials,\
                     kwargs = extra_args)
        rewards.append(r_mean)
        bcs_map.append(bcs)
        rewards_std.append(r_std)

    rewards = np.asarray(rewards);bcs_map = np.asarray(bcs_map);rewards_std = np.asarray(rewards_std)
    new_elite = np.argmax(rewards)

    print(f'New elite: {rewards[new_elite]:.2f}, with SD = {rewards_std[new_elite]:.2f}\n')

    # Plotting: 
    fig, ax = plt.subplots(figsize=(8, 6))

    # draw arrows first
    u = bcs_map[:, 0] - bcs_map_old[:, 0]
    v = bcs_map[:, 1] - bcs_map_old[:, 1]
    width = 0.015
    for i in range(len(bcs_map[:,0])):
        ax.arrow(bcs_map_old[i, 0], bcs_map_old[i, 1],u[i],v[i], head_width = 0.7* width, head_length = width, color = (0,0,0,0.4), length_includes_head = True)

    img = ax.scatter(bcs_map_old[:, 0], bcs_map_old[:, 1],
                        c=rewards, marker='o', cmap='magma', s= 30, label = 'Initial BC')
    img = ax.scatter(bcs_map[:, 0], bcs_map[:, 1],
                        c=rewards, marker='s', cmap='magma', s= 40, label = 'Final BC')

    cbar = fig.colorbar(img, orientation='vertical')
    cbar.ax.set_title('Mean Reward')


    ax.set_title('Archive: ' + str(plotname))

    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel(r"Impact $\dot{y}$")
    ax.set_xlabel(r"Impact $x$")
    plt.legend(loc = 'upper center')
    plt.tight_layout()

    if save_figure:
        fig.savefig(filename)
        print('Figured saved as ' + filename)

    return bcs_map, rewards





if __name__ == "__main__":

    # Set parameters:
    parameters = Parameters(args, init=False)
    parameters.individual_bs = 0
    parameters.action_dim = wrapper.env.action_space.shape[0]
    parameters.state_dim = wrapper.env.observation_space.shape[0]
    parameters.use_ln = True
    parameters.device = torch.device('cuda')
    # setattr(parameters, 'ls', 300)

    # Seed
    wrapper.env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Evaluation params:
    num_trials = 2
    evaluate_elite = True
    evaluate_rl = False
    case = 'broken_compare'
    save_figure = False


    # ------------------------------------------------------------------------
    #                           Elite agent
    # -> evalaute the best perforing controller on the nominal system
    # ------------------------------------------------------------------------
    if evaluate_elite:
        extra_args, plotname,filename = _extract_case(case)    
        elite_agent = load_rl_agent(parameters, str(elite_path))
        reward_mean, reward_std, bcs = evaluate(elite_agent.actor, wrapper,
                    render=args.render, trials=num_trials, kwargs = extra_args)

        print(f'Elite:{reward_mean:.2f}, with SD = {reward_std:.2f}\n')

    # --------------------------------------------------------------------------
    #                         ERL Population
    # -> evaluate entire popualtion on the faulty system
    # --------------------------------------------------------------------------

    if 'comp' in case.lower():
        # comparative plot  to the nominal case (with arrows) 
        bcs_map, rewards = gen_comparative_map(parameters, wrapper, num_trials, case, save_figure=save_figure)
    else:
        # simpel plot
        extra_args, plotname,filename = _extract_case(case)  
        agents_pop = load_genetic_agent(parameters, model_path, elite_path)
        rewards, bcs_map, rewards_std = [], [], []
        for agent in tqdm(agents_pop):  # evaluate each member for # trials
            r_mean, r_std, bcs = evaluate(agent.actor, wrapper,
                    render=args.render, trials=num_trials,\
                        kwargs = extra_args)
            rewards.append(r_mean)
            bcs_map.append(bcs)
            rewards_std.append(r_std)

        rewards = np.asarray(rewards)
        bcs_map = np.asarray(bcs_map)
        rewards_std = np.asarray(rewards_std)
        new_elite = np.argmax(rewards)

        # Plotting the nominal case
        print(f'New elite: {rewards[new_elite]:.2f}, with SD = {rewards_std[new_elite]:.2f}\n')
        gen_heatmap(bcs_map, rewards, filename=filename, name = plotname, save_figure=save_figure)


    # ------------------------------------------------------------------------
    #                             RL agent
    # ------------------------------------------------------------------------
    if evaluate_rl:
        setattr(parameters, 'ls', 32)
        rl_agent = load_rl_agent(parameters, ddpg_path)
        reward_mean, reward_std, bcs = evaluate(rl_agent.actor, wrapper,
                    render=args.render, trials=num_trials,\
                    kwargs = extra_args)

        print(f'RL (ddpg):{reward_mean:.2f}, with SD = {reward_std:.2f}\n')

    # -----------------------------------------------------------------------
    #                                   Plotting
    # ------------------------------------------------------------------------
    
    # gen_heatmap(bcs_map, rewards,
    #             filename='Results_pderl/Plots/population_map_brokenengine.png',\
    #             save_figure = False)
    # gen_heatmap(bcs_map, rewards,
    #             filename='Results_pderl/Plots/population_map_noisystate.png', \
    #             name = 'noisy state (F1)',save_figure = True)

    plt.show()