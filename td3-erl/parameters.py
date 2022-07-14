import pprint
import os
import torch


class Parameters:
    def __init__(self, cla, init=True):
        if not init:
            return

        # Set the device to run on CUDA or CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('Current device:', self.device)

        # Render episodes
        self.env_name = 'PHlab_attitude'
        self.save_periodic =  False
        
        # Number of Frames to Run
        
        self.num_frames = 800_000
  
        # Novelty Search
        self.ns = False
        self.ns_epochs = 10


        # ==================================  RL (DDPG) Params =============================================
        self.use_ddpg = False     # default isFalse
        self.test_ea = False
        self.frac_frames_train = 1.  # default training 

        self.gamma = cla.gamma
        self.lr    = cla.lr
        self.tau   = 0.005   
        self.seed  = 7
        self.batch_size = cla.batch_size
        self.use_done_mask = True
        self.buffer_size = cla.buffer_size        
        self.noise_sd = cla.noise_sd
        self.use_ounoise = False

        # hidden layer
        self.num_layers = cla.num_layers
        self.hidden_size = cla.hidden_size    # 64 for TD3-only 
        self.activation_actor   = 'relu'
        self.activation_critic  = 'elu'  
        self.learn_start = 10_000       # frames accumulated before grad updates            
        # self.total_steps = self.num_frames

        # Prioritised Experience Replay
        self.per = False
        if self.per:
            self.replace_old = True
            self.alpha = 0.7
            self.beta_zero = 0.5

        # ==================================    TD3 Params  =============================================
 
        self.policy_update_freq = 3      # minimum for TD3
        self.noise_clip = 0.5                # default for TD3

        # =================================   NeuroEvolution Params =====================================
        # Number of actors in the population
        self.pop_size = 0
        self.rl_to_ea_synch_period = 3
        self.use_champion_target = False
        
        # Genetic memory size
        self.individual_bs = 10_000

        # Save Results
        self.state_dim = None   # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.save_foldername = './logs/tmp/'

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=False) -> dict:
        """ Transfer parmaters obejct to a state dictionary. 
        Args:
            stdout (bool, optional): Print. Defaults to True.

        Returns:
            dict: Parameters dict
        """
        params = pprint.pformat(vars(self), indent=4)
        if stdout:
            print(params)

        return self.__dict__

  