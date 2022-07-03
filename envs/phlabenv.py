from distutils.command.config import config

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
    def get_reference_value(self) -> List[float]:
        raise NotImplementedError

    # todo: The controller state and error should be vectors

    @abstractmethod
    def get_controlled_state(self) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def calc_error(self) -> np.array:
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


    def scale_action(self, clipped_action : np.ndarray) -> np.ndarray:
        """ Scale the action from [-1, 1] to [action_space.low, action_space.high]. 
        Might not be needed always since it depends on the activation of the output layer. 

        Args:
            clipped_action (mp.ndarray): Clipped action vector (deflections outputed by actor)

        Returns:
            np.ndarray: action vector in the physical limits
        """        
        low, high = self.action_space.low, self.action_space.high
        return low + 0.5 * (clipped_action + 1.0) * (high - low)


class CitationEnv(BaseEnv):
    """ Example citation wrapper with p-control reward function. """

    n_actions_full : int = 10
    n_obs_full : int = 12

    def __init__(self, configuration : str = None):
        self.t = 0
        self.dt = 0.01      # [s]
        self.t_max = 20.0   # [s]

        # Have an internal storage of state [12]
        # 0,  1, 2   -> p, q, r
        # 3,  4, 5   -> V, alpha, beta
        # 6,  7, 8   -> phi, theta, psi
        # 9, 10, 11  -> he, xe, ye
        self.x: np.ndarray       = None    # aircraft state vector 
        self.obs: np.ndarray     = None    # observation vector -- used for configurations & learning
        self.obs_idx:List[int]   = None   

        # Have an internal storage of last action [10]
        # Inputs:
        #   0 de      , 1 da      , 2 dr
        #   3 trim de , 4 trim da , 5 trim dr
        #   6 df      , 7 gear    , 8 throttle1 9 throttle2
        self.n_actions : int    = None
        self.last_u: np.ndarray = None

        # refference to track
        self.ref: List[signals.BaseSignal] = None
        
        # actuator bounds
        self.rate_bound = np.deg2rad(30)        #  [deg/s] 

        if 'symmetric'  in configuration.lower():
            print('Symmetric control only.')
            self.n_actions = 1                  # theta
            self.obs_idx = [1,3,4]              # q, V, alpha

        elif 'attitude' in configuration.lower():
            print('Attitude control.')
            self.n_actions = 3                  # de, da, dr
            self.obs_idx = [0,1,2]        # all but no xe,ye

        else:
            print('Full state control.')
            self.n_actions = 3
            self.obs_idx = range(10)            # all states         

        # observation space: aircraft state + actuator state + control states (equal to actuator states)
        self.n_obs : int = len(self.obs_idx) + 2 * self.n_actions 

        # error
        self.error : np.ndarray = np.zeros((self.n_actions))

        # reward stuff
        if self.n_actions == 1:
            self.cost = 6/np.pi*np.array([1.])  # individual reward scaler [theta]
        else:
            self.cost = 6/np.pi*np.array([1., 1.,4.])     # scaler [theta, phi, beta]
        self.reward_scale = -1/3                          # scaler
        self.cost         = self.cost[:self.n_actions]
        self.max_bound    = np.ones(self.error.shape)     # bounds

    @property
    def action_space(self) -> Box:
        return Box(
            low   = -self.rate_bound * np.ones(self.n_actions),
            high  =  self.rate_bound * np.ones(self.n_actions),
            dtype = np.float64,
        )

    @property
    def observation_space(self) -> Box:
        return Box(
            low   = -10 * np.ones(self.n_obs),
            high  =  10 * np.ones(self.n_obs),
            dtype = np.float64,
        )

    @property
    def p(self) -> float:  # roll rate
        return self.x[0]
    @property
    def q(self) -> float:
        return self.x[1]
    @property
    def r(self) -> float:
        return self.x[2]
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

    @property
    def nz(self) -> float:
        """ Load factor [g] """
        return 1 + self.V * self.q/ 9.80665 

    def init_ref(self):
        if self.n_actions ==1:
            ref =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=20,
                        block_width=4,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.1) \
                        + signals.Const(0.,self.t_max, self.theta)
            self.ref = [ref]

        elif self.n_actions == 3:
            step_theta =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=20,
                        block_width=4.0,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.04) \
                         + signals.Const(0.,self.t_max, value = 0.21)
            
            step_phi =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=30,
                        block_width=4.0,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.04)

            step_beta = signals.Const(0.0, self.t_max, value = 0.21) 

            self.ref = [step_theta, step_phi, step_beta]


    def get_reference_value(self) -> float:
        return np.asarray([np.deg2rad(_signal(self.t)) for _signal in self.ref])

    def get_controlled_state(self) -> float:
        _crtl= np.asarray([self.theta,  self.phi, self.beta])
  
        return _crtl[:self.n_actions]

    def calc_error(self) -> np.array:
        self.error[:self.n_actions] = self.get_reference_value() - self.get_controlled_state()

    def get_reward(self) -> float:
        self.calc_error()
        reward_vec =  np.linalg.norm(np.clip(self.cost * self.error,-self.max_bound, self.max_bound), ord=1)
        reward     = self.reward_scale * (reward_vec.sum() / self.error.shape[0])
        return reward
    
    def filter_action(self, action : np.ndarray, tau : float = 1) -> np.ndarray:
        """ Return low-pass filtered incremental control action. 
        """
        # w_0 = 2 * 2 * np.pi  # rad/s
        # u = self.last_u / (1 + w_0 * self.dt) + action * (w_0 * self.dt) / (1 + w_0 * self.dt)
        return (1 - tau) * self.last_u + tau * action * self.dt

    def check_envelope_bounds():
        raise NotImplementedError

    def reset(self, **kwargs) -> np.ndarray:
        # Reset time
        self.t = 0.0

        # Initalize the simulink model
        citation.initialize()

        # Make a full-zero input step to retreive the state
        self.last_u = np.zeros(self.action_space.shape[0])

        # Init state vector and observation vector
        _input = np.pad(self.last_u,(0, self.n_actions_full - self.n_actions), 
                        'constant', constant_values = (0.))
        self.x = citation.step(_input)

        # Randomize reference signal sequence
        self.init_ref()

        # Build observation
        self.obs = np.concatenate((self.error.flatten(), self.x[self.obs_idx], self.last_u), axis = 0)

        return self.obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ Gym-like step function returns: (state, reward, done, info) 

        Args:
            action (np.ndarray): Un-sclaled action in the interval [-1,1] to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask, {refference signal value, time, state}
        """        
        is_done = False

        # pad action to correpond to the Simulink dimensions
        action = self.scale_action(action)   # scaled to actuator limits 

        # incremental control input: 
        u = self.filter_action(action, tau = 1)
        self.last_u = u

        # Step the system
        citation_input = np.pad(action,(0, self.n_actions_full - self.n_actions), 
                                'constant', constant_values = (0.))
        self.x = citation.step(citation_input)
        
        # Reward using clipped error
        reward   = self.get_reward()

        # Update observation based on perfect observations & actuator state
        self.obs = np.concatenate((self.error.flatten(), self.x[self.obs_idx], self.last_u), axis = 0)

        # Step time
        self.t  += self.dt

        # Check if Done:
        if self.t >= self.t_max or np.abs(self.theta) > 60. or np.abs(self.phi) > 75.  or self.H < 200 or np.any(np.isnan(self.x)):
            is_done = True
            reward += (self.t_max - self.t) * self.reward_scale  # max. negative reward for dying soon
   
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


def evaluate(verbose : bool = False):
    """ Simulate one episode """

    # reset env
    done = False
    obs = env.reset()


    # PID gains
    p, i, d = 12, 5, 2
    
    ref_beta, ref_theta, ref_phi = [], [], []
    x_lst, rewards,u_lst, nz_lst = [],[], [], []
    error_int,error_dev = np.zeros((env.action_space.shape[0])), np.zeros((env.action_space.shape[0]))
    

    while not done:
        # PID actor:
        action = -(p * obs[:env.n_actions] + i * error_int + d * error_dev)
        action[-1]*=-1.5  # rudder needs some scaling
        error_dev  = obs[:env.n_actions]

        if verbose:
            print(f't:{env.t:0.2f} theta:{env.theta:.03f} q:{env.q:.03f} alpha:{env.alpha:.03f}   V:{env.V:.03f} H:{env.H:.03f}')
            

        # Simulate one step in environment
        action = np.clip(action,-1,1)
        obs, reward, done, info = env.step(action.flatten())
        next_obs = obs

        if verbose:
            print(f'Error: {obs[:env.n_actions]} Reward: {reward:0.03f} \n \n')

        assert obs[3] == env.p
        assert obs[4] == env.q
        assert obs[5] == env.r
        assert np.isclose(env.error, obs[:env.n_actions]).all()
 
        # Update
        obs = next_obs
        error_int += obs[:env.n_actions]*env.dt
        error_dev = ( obs[:env.n_actions]- error_dev)/env.dt

        # save 
        rewards.append(reward)
        x_lst.append(env.x)
        u_lst.append(env.last_u)
        nz_lst.append(env.nz)
        
    return ref_beta,ref_theta,ref_phi,x_lst,rewards,u_lst,nz_lst

if __name__=='__main__':
    # NOTE for testing the environment implementation
    import config
    from tqdm import tqdm

    # init env an actor
    env = config.select_env('phlab_attitude')
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    
    
    trials = 2
    verbose = False
    fitness_lst =[]
    for _ in tqdm(range(trials)):
        ref_beta, ref_theta, ref_phi, x_lst, rewards, u_lst, nz_lst = evaluate(verbose)
        fitness_lst.append(sum(rewards))

    print('Fitness: ', np.mean(fitness_lst), 'SD:', np.std(fitness_lst))

    # Plotting:
    import matplotlib.pyplot as plt
    x_lst = np.asarray(x_lst); u_lst = np.asarray(u_lst)
    time = np.linspace(0., env.t , len(x_lst))
    ref_values = np.array([[ref(t_i) for t_i in time] for ref in env.ref]).transpose()

    history = np.concatenate((ref_values, u_lst, x_lst ), axis = 1)

 
    fig, axs = plt.subplots(4,2)
    axs[0,0].plot(time,history[:,0], linestyle = '--',label = 'ref_theta')
    axs[1,0].plot(time,history[:,1],linestyle = '--' ,label = 'ref_phi')
    axs[2,0].plot(time,history[:,2], linestyle = '--',label = 'ref_beta')

    axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = 'alpha')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = 'q')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = 'theta')

    axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = 'beta')
    axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = 'phi')
    axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), label = 'p')
    axs[3,0].plot(time,x_lst[:,9], label = 'H')


    # plot actions
    axs[0,1].plot(time,u_lst[:,0], linestyle = '--',label = 'de')
    axs[1,1].plot(time,u_lst[:,1], linestyle = '--',label = 'da')
    axs[2,1].plot(time,u_lst[:,2], linestyle = '--',label = 'dr')
    axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = 'nz')
  
    fig2, ax_reward = plt.subplots()
    ax_reward.plot(time,rewards)
    ax_reward.set_ylabel('Reward [-]')
    for i in range(4):
        for j in range(2):
            axs[i,j].set_xlabel('Time [s]')
            axs[i,j].legend(loc = 'best')
    
    plt.tight_layout()
    plt.show()
    env.finish()


# reward designs
# reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30)**2, max_bound), -max_bound))  # square function
# reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))  # rational function
# reward_vec = - np.maximum(np.minimum(1 / (np.abs(self.error) * 10 + 1), max_bound),    - max_bound)  # abs. linear function