from distutils.command.config import config
import envs.citation as citation
from signals.stochastic_signals import RandomizedCosineStepSequence, Step
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
    def get_reference_value(self) -> float:
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
        self.t_max = 30.0   #5 sec episodes -- TOO SHORT

        # Have an internal storage of state [12]
        # 0,  1, 2   -> p, q, r
        # 3,  4, 5   -> V, alpha, beta
        # 6,  7, 8   -> phi, theta, psi
        # 9, 10, 11  -> he, xe, ye
        self.obs: np.ndarray     = None    # observation vector
        self.x: np.ndarray       = None      # controllable state vector (useful for configurations)
        self.obs_idx:List[int] = None

        # Have an internal storage of last action [10]
        # Inputs:
        #   0 de      , 1 da      , 2 dr
        #   3 trim de , 4 trim da , 5 trim dr
        #   6 df      , 7 gear
        #   8 throttle1 9 throttle2
        self.n_actions : int    = None
        self.last_u: np.ndarray = None
        self.control_idx: List[int] = None

        # refference to track
        self.ref: signals.BaseSignal = None
        
        # actuator bounds
        self.rate_bound = np.deg2rad(20)    # 20 deg/s boudns
        if 'symmetric'  in configuration.lower():
            print('Symmetric control only.')
            self.n_actions = 1   
            self.obs_idx = [1,3,4]  # slicing for symmetric states --  q, V,alpha, theta
            self.control_idx = [7]    # theta 

        elif 'attitude' in configuration.lower():
            print('Attitude control.')
            self.n_actions = 3   
            self.obs_idx = [0,1,2,3,4,8]  # all but no xe,ye
            self.control_idx = [5,6,7]   # 6,  7, 8   -> beta, phi, theta

        else:
            print('Full state control.')
            self.n_actions = 3
            self.obs_idx = range(10)    # all states 
                 

        # observation space: aircraft state + actuator state + control states
        self.n_obs : int = len(self.obs_idx) + self.n_actions + len(self.control_idx)

        # error
        self.error : np.ndarray = np.zeros((1,self.n_actions))

        # reward stuff
        self.cost = 6/np.pi                          # scaler
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
    def q(self) -> float:
        return self.x[1]
    @property
    def V(self)-> float:
        return self.x[3] 
    @property
    def alpha(self) -> float:
        return self.x[4] 
    @property
    def beta(self) -> float:
        return self.x[5]        
    @property
    def phi(self) -> float:
        return self.x[6]
    @property
    def theta(self):
        return self.x[7]
    @property
    def psi(self) -> float:
        return self.x[8]
    @property
    def H(self)-> float:
        return self.x[9] 

    def init_ref(self):
        if self.n_actions ==1:
            ref =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=25,
                        block_width=9,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.1)
            self.ref = [ref]

        elif self.n_actions == 3:
            step_theta =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=25,
                        block_width=9,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.1)
            
            step_phi =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=25,
                        block_width=9,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.1)

            step_beta = step_theta * 0.

            self.ref = [step_beta,step_phi, step_theta]


    def get_reference_value(self) -> float:
        return np.array([ref_signal(self.t) for ref_signal in self.ref ])

    def get_controlled_state(self) -> float:
        if self.n_actions ==3:
            return np.array([self.beta,  self.phi, self.theta])
        else:
            return np.array([self.theta])

    def get_error(self) -> float:
        self.error[:self.n_actions] = self.get_reference_value() - self.get_controlled_state()


    def get_reward(self) -> float:
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


    def check_envelope_bounbds():
        raise NotImplementedError

    def reset(self, **kwargs) -> np.ndarray:
        # Reset time
        self.t = 0.0

        # Initalize the simulink model
        # citation.terminate()
        citation.initialize()

        # Randomize reference signal sequence
        self.init_ref()

        # Make a full-zero input step to retreive the state
        u = np.zeros(self.n_actions_full)
        self.last_u = u

        self.x = citation.step(u)
        self.obs = np.concatenate((self.error.flatten(), self.x[self.obs_idx], self.last_u[:self.n_actions]), axis = 0)

        return self.obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ Gym-like step function returns: (state, reward, done, info) 

        Args:
            action (np.ndarray): Un-sclaled action in the interval [-1,1] to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask, {refference signal value, behavioural characteristic}
        """        
        is_done = False
        
        # pad action to correpond to the Simulink dimensions
        action = np.clip(action, -1. , 1.)
        scaled_action = self.scale_action(action)   # scaled to actuator limits i.e. -20,+20 deg/s
        scaled_action = np.pad(scaled_action,
                        (0, self.n_actions_full - self.n_actions), 
                        'constant', constant_values = (0.))


        # incremental control input: 
        u = self.last_u + scaled_action * self.dt
        self.last_u = u

        # Step the system
        self.x = citation.step(u)
        
        # Reward
        self.get_error()
        reward   = self.get_reward()

        # Update observation based on perfect observations & actuator state
        self.obs = np.concatenate((self.error.flatten(), self.x[self.obs_idx], self.last_u[:self.n_actions]), axis = 0)

        # Step the time
        self.t  += self.dt

        # Done:
        if self.t >= self.t_max or np.abs(self.theta) > 45. or self.H < 100 or np.any(np.isnan(self.x)):
            is_done = True
   
        # info:
        info = {
            "ref": self.ref,
            "x":   self.x,
            "t":   self.t,
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
    import config


    env = config.select_env('phlab_symmetric')
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    
    # Simulate one episode
    total_reward = 0.0

    # Generate refference signal:
    theta_ref = np.zeros(1000)

    # reset env
    done = False
    obs = env.reset()

    ref_beta, ref_theta, ref_phi = [], [], []
    while not done:
        # Actor:
        scaled_action = actor.select_action(np.array(obs))
        action = env.scale_action(scaled_action)
        action[0] = 10 * env.q
        
        # Simulate one step in environment
        obs, reward, done, info = env.step(action.flatten())
        next_obs = obs

        print(f'de: {action[0]:.03}  q:{env.q:.03f}  V:{env.V:.03f}  alpha:{env.alpha:.03f}  theta:{env.theta:.03f}   H:{env.H:.03f}')
        print(f'Reward: {reward:0.03f}')
        print('Error:', env.error)



        # Update
        total_reward += reward
        obs = next_obs


        # save refs
        # ref_beta.append(env.ref[0](env.t)); ref_theta.append(env.ref[2](env.t)); ref_phi.append(env.ref[1](env.t))

    # import matplotlib.pyplot as plt

    # plt.plot(ref_theta, label = 'theta')
    # plt.plot(ref_phi, label = 'phi')
    # plt.plot(ref_beta, label = 'beta')
    # plt.legend(loc = 'best')
    # plt.show()
    env.finish()
