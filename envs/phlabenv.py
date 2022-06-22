import envs.citation as citation
from signals.stochastic_signals import RandomizedCosineStepSequence
import signals

from abc import ABC, abstractmethod
import gym
from gym.spaces import Box
import numpy as np
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
        """ Scale the action from [-1, 1] to [action_space.low, action_space.high]. 
        Might not be needed always since it depends on the activation of the output layer. 

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
        self.obs: np.ndarray     = None    # observation vector
        self.x: np.ndarray       = None      # controllable state vector (useful for configurations)
        self.state_idx:List[int] = None

        # Have an internal storage of last action [10]
        # Inputs:
        #   0 de      , 1 da      , 2 dr
        #   3 trim de , 4 trim da , 5 trim dr
        #   6 df      , 7 gear
        #   8 throttle1 9 throttle2
        self.last_u: np.ndarray = None

        # refference to track
        self.ref: signals.BaseSignal = None
        
        # actuator bounds
        self.rate_bound = np.deg2rad(20)    # 20 deg/s boudns

        if 'symmetric' or 'sym' in configuration.lower():
            print('Symmetric control only.')
            self.n_actions = 1   
            self.state_idx = [1,3,4,7]  # slicing for symmetric states --  q, V,alpha, theta
            self.n_obs = len(self.state_idx)
        elif 'att' in configuration.lower():
            print('Attitude control.')
            self.n_actions = 3   
            self.state_idx = range(8)  # all but no xe,ye
            self.n_obs = len(self.state_idx)
        else:
            print('Full attitude control.')
            self.n_actions = 3
            self.state_idx = range(10)    # all states 
            self.n_obs = len(self.state_idx)     
            assert self.n_obs == self.n_obs_full

        # error
        self.error : np.ndarray = np.zeros((1,self.n_actions))

        # reward stuff
        self.cost = 6/np.pi             # scaler
        self.max_bound = np.ones(self.error.shape)   # bounds

    @property
    def action_space(self) -> Box:
        return Box(
            low   = -self.rate_bound * np.ones(self.n_actions),
            high  = self.rate_bound * np.ones(self.n_actions),
            dtype = np.float64,
        )

    @property
    def observation_space(self) -> Box:
        return Box(
            low   = -100 * np.ones(self.n_obs),
            high  = 100 * np.ones(self.n_obs),
            dtype = np.float64,
        )

    @property
    def V(self):
        return self.obs[3] 
    @property
    def alpha(self) -> float:
        return self.obs[4] 
    @property
    def beta(self) -> float:
        return self.obs[5]        
    @property
    def theta(self):
        return self.obs[7]
    @property
    def phi(self) -> float:
        return self.obs[6]
    @property
    def psi(self) -> float:
        return self.obs[8]

    def get_reference(self) -> float:
        return self.ref(self.t)

    def get_controlled_state(self) -> float:
        return self.theta


    def get_error(self) -> float:
        self.error[0,:] = np.array(self.get_reference() - self.get_controlled_state())


    def get_reward(self) -> float:
        # compute error
        self.get_error()
        
        # reward shapes
        # reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30)**2, max_bound), -max_bound))  # square function
        # reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))  # rational function
        # reward_vec = - np.maximum(np.minimum(1 / (np.abs(self.error) * 10 + 1), max_bound),    - max_bound)  # abs. linear function

        reward_vec = 1/3 * np.linalg.norm( np.clip( self.cost * self.error , -self.max_bound, self.max_bound), ord=1)

        reward = -reward_vec.sum() / self.error.shape[0]
        return reward
    
    def get_bcs(self) -> Tuple[float, float]:
        """ Return behavioural characteristic """
        return (0.,0.)

    def check_nan():
        raise NotImplementedError
    
    def check_envelope_bounbds():
        raise NotImplementedError

    def reset(self, **kwargs) -> np.ndarray:
        # Reset time
        self.t = 0.0

        # todo: Reset actuators (in case of incremental control

        # Initalize the simulink model
        # citation.terminate()
        citation.initialize()

        # Randomize reference signal sequence
        self.ref = RandomizedCosineStepSequence(
                    t_max=self.t_max,
                    ampl_max=25,
                    block_width=9,
                    smooth_width=3.0,
                    n_levels=10,
                    vary_timings=0.1,
                    )

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
        scaled_action = self.scale_action(action)   # scaled to actuator limits i.e. -20,+20 deg/s
        scaled_action = np.pad(scaled_action,
                        (0, self.n_actions_full - self.n_actions), 
                        'constant', constant_values = (0.))


        # incremental control input: 
        u = self.last_u + scaled_action * self.dt
        self.last_u = u

        # Step the system
        self.obs = citation.step(u)

        # Update state based on perfect observations
        self.x   = self.obs[self.state_idx]

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

    # Generate refference signal:
    theta_ref = np.zeros(1000)

    # reset env
    done = False
    state = env.reset()

    while not done:
        # Actor:
        scaled_action = actor.select_action(np.array(state))
        action = env.scale_action(scaled_action)
        action[0] = 10 * state[0]
        
        # Simulate one step in environment
        obs, reward, done, info = env.step(action.flatten())
        next_state = obs[env.state_idx]

        print(f'de: {action[0]:.03}  q:{state[0]:.03f}  V:{state[1]:.03f}  alpha:{state[2]:.03f}  theta:{state[3]:.03f} ')
        print(f'Reward: {reward:0.03f}')

        # Update
        total_reward += reward
        state = next_state


    env.finish()
