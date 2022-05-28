# basic OS interaction
from pathlib import Path

# python standard modules for AI
import gym
import numpy as np
from tqdm import tqdm
import pandas as pd


# my modules
from QD.saving_utils import *
from envs.lunarlander import simulate
from QD.agents import QD_agent, QD_agent_NN


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                              Test Traiend Policies
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



def run_evaluation(outdir, env_name ="LunarLanderContinuous-v2", env_seed=1339, random: bool = False, trials: bool = 5, kwargs : dict = None):
    """Simulates 10 random archive solutions and saves videos of them.

    Videos are saved to outdir / videos.

    Args:
        outdir (Path): Path object for the output directory from which to
            retrieve the archive and save videos.
        env_seed (int): Seed for the environment.
    """
    outdir = Path(outdir)
    df = pd.read_csv(outdir / "archive.csv")

    if random:
        indices = np.random.permutation(len(df))[:10]
    else:
        indices = range(len(df))

    # Since we are using multiple processes, it is simpler if each worker
    # just creates their own copy of the environment instead of trying to
    # share the environment. This also makes the function "pure."
    env = gym.make(env_name)

    video_env = gym.wrappers.Monitor(
        gym.make(env_name),
        str(outdir / "videos"),
        force=True,
        # Default is to write the video for "cubic" episodes -- 0,1,8,etc (see
        # https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py#L54).
        # This will ensure all the videos are written.
        video_callable=lambda idx: True,
    )

    if env_seed is not None:
        env.seed(env_seed)

    print(f'Evaluate archive:')
    MAX_REWARD = 200
    max_idx = 0
    actor = QD_agent(env)   # wrap the model in an agent class
    
    for idx in tqdm(indices):
        model = np.array(df.loc[idx, "solution_0":])
        actor.update_params(model)
        reward, impact_x_pos, impact_y_vel = simulate(
                                            actor, env, render=False,**kwargs)

        if reward > MAX_REWARD:
            print(f"Max reward {reward:0.3f} at index = {idx}")
            elite_actor = actor
            MAX_REWARD = reward
            max_idx = idx

    # simulate again the best, this time with video
    print(f'\nEvaluating identified elite (actor {max_idx})...')
    rewards, bcs = [], []
    for _ in tqdm(range(trials)):
        total_reward, impact_x_pos, impact_y_vel = simulate(
                                        elite_actor, env, render=False, **kwargs)
        rewards.append(total_reward)
        bcs.append((impact_x_pos, impact_y_vel))

    bcs = np.asarray(bcs)

    simulate(elite_actor, video_env, render=True, **kwargs)
    # close video env
    video_env.close()

    return np.average(rewards), np.std(rewards), np.average(bcs, axis=0)
