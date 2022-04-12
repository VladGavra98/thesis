import gym
import numpy as np


def simulate(model, env_name:str, seed=None, video_env=None):
    """Simulates the lunar lander model.

    Args:
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
        video_env (gym.Env): If passed in, this will be used instead of creating
            a new env. This is used primarily for recording video during
            evaluation.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
    """
    if video_env is None:
        # Since we are using multiple processes, it is simpler if each worker
        # just creates their own copy of the environment instead of trying to
        # share the environment. This also makes the function "pure."
        env = gym.make(env_name)
    else:
        env = video_env

    if seed is not None:
        env.seed(seed)

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []
    obs = env.reset()
    done = False

    while not done:
        action = model @ obs  # Linear policy.
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Refer to the definition of state here:
        # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L306
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

    # If the lunar lander did not land, set the x-pos to the one from the final
    # timestep, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)

    # Only close the env if it was not a video env.
    if video_env is None:
        env.close()

    return total_reward, impact_x_pos, impact_y_vel
