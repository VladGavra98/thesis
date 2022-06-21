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

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale the actions from [-1, 1] to the appropriate scale of the action space."""
        low, high = self.action_space.low, self.action_space.high
        action = low + 0.5 * (action + 1.0) * (high - low)
        return action


class CitationEnv(BaseEnv):
    """ Example citation wrapper with p-control reward function. """

    n_actions = 10
    n_obs = 12

    def __init__(self):
        self.t = 0
        self.dt = 0.01
        self.t_max = 5.0   #5 sec episodes

        # Have an internal storage of state [12]
        # 0, 1, 2   -> p, q, r
        # 3, 4, 5   -> V, alpha, beta
        # 6, 7, 8   -> phi, theta, psi
        # 9, 10, 11  -> he, xe, ye,
        self.x: np.ndarray = None

        # Have an internal storage of last action [10]
        self.last_u: np.ndarray = None

    @property
    def action_space(self) -> Box:
        return Box(
            low=-100 * np.ones(self.n_actions),
            high=100 * np.ones(self.n_actions),
            dtype=np.float64,
        )

    @property
    def observation_space(self) -> Box:
        return Box(
            low=-100 * np.ones(self.n_obs),
            high=100 * np.ones(self.n_obs),
            dtype=np.float64,
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

        # Make a zero input step to retreive the state
        u = np.zeros(self.n_actions)
        self.x = citation.step(u)
        self.last_u = u

        return self.x

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ Gym-like step function returns: (state, reward, done, info) 

        Args:
            action (np.ndarray): Un-sclaled action to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask, {refference signal value, behavioural characteristic}
        """        

        # todo: clip the action vector between [-1, 1]

        # Scale [1, 1] action to the action space
        scaled_action = self.scale_action(action)
        self.last_u = scaled_action

        # todo: Implement incremental control: scaled_action is delta_u -> u += delta_u

        # Step the system
        self.x = citation.step(scaled_action)

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

        return self.x, reward, is_done, info

    def render(self, **kwargs):
        """ just to make the linter happy (we are deriving from gym.Env)"""
        pass

    def simulate (self, actor : object,
                render : bool = False) -> Dict[float, tuple]:
        """ Wrapper function for the gym LunarLander enviornment. It can include the faulty cases:
                -> broken main engine
                -> faulty navigation snesors (i.e., nosiy position)

        Args:
            actor (object): Actor class that has the select_action() method
            env (object): Environment with OpenAI Gym API (make(), reset(),step())
            render (bool, optional): Should render the video env. Defaults to False.
    
        Returns:
            tuple: Reward (float),
        """
        
        total_reward = 0.0

        # reset env
        done = False
        state = self.reset()

        while not done:
            if render:
                self.render()

            # Actor:
            action = self.scale_action(actor.select_action(np.array(state)))

            # Simulate one step in environment
            next_state, reward, done, info = self.step(action.flatten())

            # Update
            total_reward += reward
            state = next_state



        return {'total_reward':total_reward}

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
        return self.policy @ state

if __name__=='__main__':
    # NOTE for testing the environment implementation
    env = CitationEnv()
    actor = Actor( env.observation_space.shape[0], env.action_space.shape[0])

    # Simulate one episode
    total_reward = 0.0

    # reset env
    done = False
    state = env.reset()

    while not done:

        # Actor:
        action = env.scale_action(actor.select_action(np.array(state)))

        # Simulate one step in environment
        next_state, reward, done, info = env.step(action.flatten())

        # Update
        total_reward += reward
        state = next_state

    env.finish()
