import numpy as np, os,random
from core import mod_utils as utils
from core.ddpg import GeneticAgent
from parameters import Parameters
import torch
import gym
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) ' +
                                 '(Swimmer-v2) (Hopper-v2)', type=str,default = 'LunarLanderContinuous-v2')
parser.add_argument('-seed', help='Random seed to be used', type=int, default=7)
parser.add_argument('-render', help='Render gym episodes', action='store_true', default = False)
parser.add_argument('-load_champion', help='Loads the best performingactor', action='store_true', default = False)
args = parser.parse_args()

plt.style.use('ggplot') 
plt.rcParams.update({'font.size': 12})

def evaluate(agent, env, trials : int = 10, render : bool =False):
    """ Evaualte an individual for a couple of trails/games.

    Args:
        agent (_type_): Actor to evalaute.
        env (_type_): Environment for testing. Should be the same API as gym environments.
        trials (int, optional): Number of evaluation runs. Defaults to 10.
        render (bool, optional): Show policy in a video. Defaults to False.


    """
    rewards, bcs = [], []

    for _ in range(trials):
        total_reward = 0.0
        impact_x_pos = None
        impact_y_vel = None
        all_y_vels = []

        done = False

        state = env.reset()
        done = False
        while not done:
            if render: env.render()
            action = agent.actor.select_action(np.array(state))

            # Simulate one step in environment
            next_state, reward, done, info = env.step(action.flatten())
            total_reward += reward
            state = next_state

            x_pos = state[0]
            y_vel = state[3]
            leg0_touch = bool(state[6])
            leg1_touch = bool(state[7])
            all_y_vels.append(y_vel)

            # Check if the lunar lander is impacting for the first time.
            if impact_x_pos is None and (leg0_touch or leg1_touch):
                impact_x_pos = x_pos
                impact_y_vel = y_vel

        if impact_x_pos is None:
            impact_x_pos = x_pos
            impact_y_vel = min(all_y_vels)


        rewards.append(total_reward)
        bcs.append((impact_x_pos, impact_y_vel))


    bcs = np.asarray(bcs)
    return np.average(rewards), np.std(rewards), np.average(bcs, axis = 0)



def load_genetic_agent(args, load_champion : bool = False):
    actor_path = os.path.join(args.model_path)
    agents_pop = []

    checkpoint = torch.load(actor_path)

    for _,model in checkpoint.items():
        agent = GeneticAgent(args)
        agent.actor.load_state_dict(model)
        agents_pop.append(agent)

    if load_champion:
        agent.actor.load_state_dict(torch.load(elite_path))
    print("Model loaded from: " + parameters.model_path)

    return agents_pop

def gen_heatmap(bcs_map : np.ndarray, rewards : np.ndarray, filename : str, save_figure : bool = False):
    """Saves a heatmap of the optimizer's archive to the filename.

    Args:
        bcs_map
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.scatter(bcs_map[:,0],bcs_map[:,1], c=rewards, marker='s', cmap = 'magma')
    # fig.suptitle('Archive Illumiantion')
    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel(r"Impact $\dot{y}$")
    ax.set_xlabel(r"Impact $x$")

    cbar = fig.colorbar(img, orientation='vertical')
    cbar.ax.set_title('Mean Reward')

    plt.show()

    if save_figure:
        fig.savefig(filename)

if __name__ == "__main__":

    env = utils.NormalizedActions(gym.make(args.env))
    model_path = 'pderl/logs/evo_nets.pkl'
    elite_path = 'pderl/logs/elite_net.pkl'

    parameters = Parameters(args, init=False)
    parameters.individual_bs = 0
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.use_ln = True
    parameters.device = torch.device('cuda')
    setattr(parameters, 'model_path', model_path)
    setattr(parameters, 'elite_path', elite_path)
    setattr(parameters, 'ls', 32)
    
    #Seed
    env.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # Load popualtion for evaluation:
    agents_pop = load_genetic_agent(parameters)
    rewards,bcs_map = [],[]
    for agent in agents_pop:
        reward_mean,reward_std, bcs = evaluate(agent, env, render=args.render, trials = 2)
        rewards.append(reward_mean); bcs_map.append(bcs)
    rewards = np.asarray(rewards); bcs_map = np.asarray(bcs_map)

    # Plotting
    gen_heatmap(bcs_map, rewards, filename='Results_pderl/Plots/population_map.png')


