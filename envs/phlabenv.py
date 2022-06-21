from abc import ABC, abstractmethod
import gym
from gym.spaces import Box
import nacl
import numpy as np
import envs.citation as citation
from typing import Tuple, List, Dict




class BaseEnv(gym.Env, ABC):
    """ Base class to be able to write generic training & eval code
    that applies to all Citation environment variations. """

    @property
    @abstractmethod
    def action_space(self) -> Box:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        raise NotImplementedError

    @abstractmethod
    def get_reference(self) -> float:
        raise NotImplementedError

    # todo: The controller state and error should be vectors

    @abstractmethod
    def get_controlled_state(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_error(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self) -> float:
        pass

    def unscale_action(self, action : np.ndarray) -> np.ndarray:
        """
        Rescale the action from [action_space.low, action_space.high] to [-1, 1]

        Args:
            action (mp.ndarray): Action in the physical limits

        Returns:
            np.ndarray: Action vector in the [-1,1] interval for learning tasks
        """        
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0


    def scale_action(self, scaled_action : np.ndarray) -> np.ndarray:
        """ Scale the action from [-1, 1] to [action_space.low, action_space.high]
        (no need for symmetric action space)

        Args:
            scaled_action (mp.ndarray): _description_

        Returns:
            np.ndarray: action vector in the physical limits
        """        
        low, high = self.action_space.low, self.action_space.high
        return low + 0.5 * (scaled_action + 1.0) * (high - low)


class CitationEnv(BaseEnv):
    """ Example citation wrapper with p-control reward function. """

    n_actions_full = 10
    n_obs_full = 12

    def __init__(self, configuration : str = None):
        self.t = 0
        self.dt = 0.01
        self.t_max = 5.0   #5 sec episodes

        # Have an internal storage of state [12]
        # 0,  1, 2   -> p, q, r
        # 3,  4, 5   -> V, alpha, beta
        # 6,  7, 8   -> phi, theta, psi
        # 9, 10, 11  -> he, xe, ye
        self.obs: np.ndarray  = None    # observation vector
        self.x: np.ndarray  = None      # controllable state vector (useful for configurations)
        self.state_idx:list = None

        # Have an internal storage of last action [10]
        # Inputs:
        #   0 de      , 1 da      , 2 dr
        #   3 trim de , 4 trim da , 5 trim dr
        #   6 df      , 7 gear
        #   8 throttle1 9 throttle2
        self.last_u: np.ndarray = None

        if 'symmetric' or 'sym' in configuration.lower():
            print('Symmetric control only.')
            self.n_actions = 1   
            self.state_idx = [1,3,4,7]
            self.n_obs = len(self.state_idx)
        else:
            print('Full attitude control.')
            self.n_actions = 3
            self.state_idx = range(10)
            self.n_obs = len(self.state_idx)
            assert self.n_obs == self.n_obs_full

    @property
    def action_space(self) -> Box:
        return Box(
            low   = -30 * np.ones(self.n_actions),
            high  = 30 * np.ones(self.n_actions),
            dtype = np.float64,
        )

    @property
    def observation_space(self) -> Box:
        return Box(
            low   = -100 * np.ones(self.n_obs),
            high  = 100 * np.ones(self.n_obs),
            dtype = np.float64,
        )

    # todo: make this a vector
    def get_reference(self) -> float:
        # For example a step reference (  have my own class to generate signals)
        return 1.0 if self.t > 5.0 else 0.0

    # todo: make this a vector
    def get_controlled_state(self) -> float:
        # for example let's control p:
        return self.x[0]

    # todo: make this a vector
    def get_error(self) -> float:
        return self.get_reference() - self.get_controlled_state()

    def get_reward(self) -> float:
        # todo: parameterize reward function:
        return - 1000.0 * self.get_error()**2
    
    def get_bcs(self) -> Tuple[float, float]:
        """ Return behavioural characteristic """
        return (0.,0.)

    def reset(self, **kwargs) -> np.ndarray:
        # Reset time
        self.t = 0.0

        # todo: Reset actuators (in case of incmrenetal control

        # todo: If training -> randomize reference signal sequence

        # Initalize the simulink model
        # todo: check if initialize() resets the citation to initial conditions! (if not we have to reimport terminate first?)
        # citation.terminate()
        citation.initialize()

        # Make a full-zero input step to retreive the state
        u = np.zeros(self.n_actions_full)
        self.last_u = u

        self.obs = citation.step(u)
        self.x = self.obs[self.state_idx]  # isolate states from obs

        return self.x

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ Gym-like step function returns: (state, reward, done, info) 

        Args:
            action (np.ndarray): Un-sclaled action in the interval [-1,1] to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask, {refference signal value, behavioural characteristic}
        """        

        # pad action to correpond to the Simulink dimensions
        scaled_action = self.scale_action(action)
        scaled_action = np.pad(scaled_action,
                        (0, self.n_actions_full - self.n_actions), 
                        'constant', constant_values = (0.))

        self.last_u = scaled_action

        # todo: Implement incremental control: scaled_action is delta_u -> u += delta_u

        # Step the system
        self.obs = citation.step(scaled_action)

        # Step the time
        self.t += self.dt

        # Reward
        reward = self.get_reward()

        # Done:
        is_done = self.t >= self.t_max

        # info:
        info = {
            "ref": self.get_reference(),
        }

        return self.obs, reward, is_done, info

    def render(self, **kwargs):
        """ just to make the linter happy (we are deriving from gym.Env)"""
        pass

    @staticmethod
    def finish():
        """ Terminate the simulink thing."""
        citation.terminate()




class Actor():
    # NOTE for testing the environment implementation
    def __init__(self,state_dim, action_dim):
        # random linear policy
        self.policy = np.random.rand(action_dim, state_dim,)

    def select_action(self,state):
        return (self.policy @ state)/100

if __name__=='__main__':
    # NOTE for testing the environment implementation
    env = CitationEnv(configuration='symmetric')
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    

    # Simulate one episode
    total_reward = 0.0

    # reset env
    done = False
    state = env.reset()

    while not done:

        # Actor:
        scaled_action = actor.select_action(np.array(state))
        action = env.scale_action(scaled_action)
        

        # Simulate one step in environment
        obs, reward, done, info = env.step(action.flatten())
        next_state = obs[env.state_idx]

        print(f'de: {action[0]:.03}  q:{state[0]:.03f}  V:{state[1]:.03f}  alpha:{state[2]:.03f}  theta:{state[3]:.03f}')
        # Update
        total_reward += reward
        state = next_state


    env.finish()
