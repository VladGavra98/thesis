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
            clipped_action (np.ndarray): Clipped action vector (deflections outputed by actor)

        Returns:
            np.ndarray: action vector in the physical limits
        """     
        low, high = self.action_space.low, self.action_space.high
        return low + 0.5 * (clipped_action + 1.0) * (high - low)


class CitationEnv(BaseEnv):
    """ Example citation wrapper with p-control reward function. """

    n_actions_full : int = 10
    n_obs_full : int = 12

    def __init__(self, configuration : str = None, mode : str = ""):
        if 'symmetric'  in configuration.lower():
            print('Symmetric control only.')
            self.n_actions = 1                  # theta
            self.obs_idx   = [1]                # q
        elif 'attitude' in configuration.lower():
            print('Attitude control.')
            self.n_actions = 3                  # de, da, dr
            self.obs_idx = [0,1,2]              # all but no xe,ye
        else:
            print('Full state control.')
            self.n_actions = 3
            self.obs_idx = range(10)            # all states         

        # use incremental control
        self.use_incremental = 'incremental' in mode.lower()
        if self.use_incremental: print('Incremental control.')

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

        # Have an internal storage of last action [10]
        # Inputs:
        #   0 de      , 1 da      , 2 dr
        #   3 trim de , 4 trim da , 5 trim dr
        #   6 df      , 7 gear    , 8 throttle1 9 throttle2
        self.last_u: np.ndarray = None

        # refference to track
        self.ref: List[signals.BaseSignal] = None
        
        # actuator bounds
        if self.use_incremental:
            self.bound = np.deg2rad(25)        #  [deg/s] 
        else:
            self.bound = np.deg2rad(10)        #  [deg]

        # state bounds
        self.max_theta = np.deg2rad(60.)
        self.max_phi = np.deg2rad(75.)
        
        # observation space: 
        if self.use_incremental:
            # aircraft state + actuator state + control states error (equal to actuator states)
            self.n_obs : int = len(self.obs_idx) +  2 * self.n_actions 
        else:
            # aircraft state + control states error
            self.n_obs : int = len(self.obs_idx) +  self.n_actions 

        # error
        self.error : np.ndarray = np.zeros((self.n_actions))

        # reward stuff
        if self.n_actions == 1:
            self.cost = 6/np.pi*np.array([1.])  # individual reward scaler [theta]
        else:
            self.cost = 6/np.pi*np.array([1., 1.,1.])     # scaler [theta, phi, beta]
        self.reward_scale = -1/3                          # scaler
        self.cost         = self.cost[:self.n_actions]
        self.max_bound    = np.ones(self.error.shape)     # bounds

    @property
    def action_space(self) -> Box:
        return Box(
            low   = -self.bound * np.ones(self.n_actions),
            high  =  self.bound * np.ones(self.n_actions),
            dtype =  np.float64,
        )
    @property
    def observation_space(self) -> Box:
        return Box(
            low   = -30 * np.ones(self.n_obs),
            high  =  30 * np.ones(self.n_obs),
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
        return 1 + self.V * self.q/9.80665 

    def init_ref(self):
        if self.n_actions ==1:
            ref =  RandomizedCosineStepSequence(
                        t_max=self.t_max,
                        ampl_max=20,
                        block_width=4,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.1) \
                        + signals.Const(0.,self.t_max, value = 0.21)
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
                        ampl_max=20,
                        block_width=4.0,
                        smooth_width=3.0,
                        n_levels=10,
                        vary_timings=0.04)

            step_beta = signals.Const(0.0, self.t_max, value = 0.0) 

            self.ref = [step_theta, step_phi, step_beta]


    def get_reference_value(self) -> float:
        return np.asarray([np.deg2rad(_signal(self.t)) for _signal in self.ref])

    def get_controlled_state(self) -> float:
        _crtl = np.asarray([self.theta,  self.phi, self.beta])
        return _crtl[:self.n_actions]

    def calc_error(self) -> np.array:
        self.error[:self.n_actions] = self.get_reference_value() - self.get_controlled_state()

    def get_reward(self) -> float:
        self.calc_error()
        reward_vec = np.abs(np.clip(self.cost * self.error,-self.max_bound, self.max_bound))
        reward     = self.reward_scale * (reward_vec.sum() / self.error.shape[0])
        return reward
    
    def incremental_control(self, action : np.ndarray) -> np.ndarray:
        """ Return low-pass filtered incremental control action. 
        """
        return self.last_u + action * self.dt

    def check_envelope_bounds():
        raise NotImplementedError

    def pad_action(self, action : np.ndarray) -> np.ndarray:
        """ Pad action with 0 to correpond to the Simulink dimensions. 
        """
        citation_input = np.pad(action,
                                (0, self.n_actions_full - self.n_actions), 
                                'constant', 
                                constant_values = (0.)) 
        return citation_input

    def reset (self, **kwargs) -> np.ndarray:
        # Reset time
        self.t = 0.0

        # Initalize the simulink model
        citation.initialize()

        # Make a full-zero input step to retreive the state
        self.last_u = np.zeros(self.action_space.shape[0])

        # Init state vector and observation vector
        _input = self.pad_action(self.last_u)
        self.x = citation.step(_input)

        # Randomize reference signal sequence
        self.init_ref()

        # Build observation
        self.obs = np.hstack((self.error.flatten(), self.x[self.obs_idx]))
        if self.use_incremental: self.obs =  np.hstack((self.obs,self.last_u))

        return self.obs

    def step (self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """ Gym-like step function returns: (state, reward, done, info) 

        Args:
            action (np.ndarray): Un-sclaled action in the interval [-1,1] to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask, {refference signal value, time, state}
        """        
        is_done = False

        # scale action to actuator rate bounds 
        action = self.scale_action(action)   # scaled to actuator limits 

        # incremental control input: 
        if self.use_incremental:
            u = self.incremental_control(action)
        else:
            u = action

        # citation input 
        _input = self.pad_action(u)
        self.x = citation.step(_input)
        
        # Reward using clipped error
        reward = self.get_reward()

        # Update observation based on perfect observations & actuator state
        self.obs = np.hstack((self.error.flatten(), self.x[self.obs_idx]))
        self.last_u = u
        if self.use_incremental: self.obs =  np.hstack((self.obs,self.last_u))
        

        # Step time
        self.t  += self.dt

        if self.t >= self.t_max \
            or np.abs(self.theta) > self.max_theta \
            or np.abs(self.phi)   > self.max_phi:

            is_done = True
            reward += 1/self.dt * (self.t_max - self.t) * self.reward_scale * 10 # negative reward for dying soon

        if np.any(np.isnan(self.x)):
            print('NaN encountered: ', self.x)
            is_done = True
            reward += 1/self.dt * (self.t_max - self.t) * self.reward_scale * 10 # negative reward for dying soon
        
        # info:
        info = {
            "ref": self.ref,
            "x":   self.x,
            "t":   self.t,
        }
        return self.obs, reward, is_done, info

    # def check_envelope_bounds(self, reward):
    #     # Check if Done:
 
    #     return is_done

    def render(self, **kwargs):
        """ just to make the linter happy (we are deriving from gym.Env)"""
        pass

    @staticmethod
    def finish():
        """ Terminate the simulink thing."""
        citation.terminate()



def evaluate(verbose : bool = False):
    """ Simulate one episode """
    # reset env
    done = False
    obs = env.reset()

    # PID gains
    p, i, d = 6, 6,5
    
    ref_beta, ref_theta, ref_phi = [], [], []
    x_lst, rewards,u_lst, nz_lst = [],[], [], []
    error_int,error_dev = np.zeros((env.action_space.shape[0])), np.zeros((env.action_space.shape[0]))
    
    while not done:
        u_lst.append(env.last_u)
        x_lst.append(env.x)
        nz_lst.append(env.nz)

        # PID actor:
        action = -(p * obs[:env.n_actions] + i * error_int + d * error_dev)

        if action.shape[0] > 1:
            action[-1]*=-1.5  # rudder needs some scaling
        error_dev  = obs[:env.n_actions]

        # Simulate one step in environment
        action = np.clip(action,-1,1)

        if verbose:
            print(f'Action: {np.rad2deg(action)} -> deflection: {np.rad2deg(env.last_u)}')
            print(f't:{env.t:0.2f} theta:{env.theta:.03f} q:{env.q:.03f} alpha:{env.alpha:.03f}   V:{env.V:.03f} H:{env.H:.03f}')
            
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


    # BCs:
    actions = np.asarray(u_lst)
    bcs = np.std(actions, axis = 0)

    env.finish()
    print('bcs:', bcs)

        
    return ref_beta,ref_theta,ref_phi,x_lst,rewards,u_lst,nz_lst

if __name__=='__main__':
    # NOTE for testing the environment implementation
    import config
    from tqdm import tqdm

    # init env an actor
    env = config.select_env('phlab_attitude')

    
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
    axs[0,0].plot(time,history[:,0], linestyle = '--',label = r'$\theta_{ref}$')
    axs[1,0].plot(time,history[:,1],linestyle = '--' ,label = r'$\phi_{ref}$')
    axs[2,0].plot(time,history[:,2], linestyle = '--',label = r'$\beta_{ref}$')

    axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), label = r'$q$')
    axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$')

    axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = r'$\beta$')
    axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = r'$\phi$')
    axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), label = r'$p$')
    axs[3,0].plot(time,x_lst[:,9], label = 'H')


    # plot actions
    axs[0,1].plot(time,np.rad2deg(u_lst[:,0]), linestyle = '--',label = r'$\delta_e$ [deg]')
    axs[1,1].plot(time,np.rad2deg(u_lst[:,1]), linestyle = '--',label = r'$\delta_a$ [deg]')
    axs[2,1].plot(time,np.rad2deg(u_lst[:,2]), linestyle = '--',label = r'$\delta_r$ [deg]')
    axs[3,1].plot(time,nz_lst[:], linestyle = '--',label = r'$n_z$ [g]')
  
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