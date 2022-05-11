import numpy as np
import typing

def simulate(actor : object, env : object,render : bool = False, broken_engine : bool = False) -> tuple:
    """ Wrapper function for the gym Luanr LAnder enviornment. It can include the faulty cases:
            -> broken engine


    Args:
        actor (object): Actor class that has the select_action() method
        env (object): Environment with OpenAI Gym API (make(), reset(),step())
        render (bool, optional): Should render the video env. Defaults to False.
        broken_engine (bool, optional): Clip the main engine to 75% (fault cases for evaluation only). Defaults to False.
                                         
                                         
    Returns:
        tuple: Reward (float), Imapct x-position (float), impact y-velocity (float)
    """
    
    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []

    # reset env
    done = False
    state = env.reset()

    while not done:
        if render:
            env.render()

        # Actor:
        action = actor.select_action(np.array(state))

        # Simulate one step in environment
        if broken_engine:
            action[0] = np.clip(action[0], -1., 0.5)

        next_state, reward, done, info = env.step(action.flatten())

        total_reward += reward
        state = next_state

        # Boudnary characteristics (BCs):
        x_pos = state[0]
        y_vel = state[3]
        leg0_touch = bool(state[6])
        leg1_touch = bool(state[7])
        all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

    # if no impact edge case
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)

    return total_reward,impact_x_pos,impact_y_vel
