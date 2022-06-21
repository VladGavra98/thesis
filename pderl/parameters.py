import pprint
import os
import torch


class Parameters:
    def __init__(self, cla, init=True):
        if not init:
            return

        # Set the device to run on CUDA or CPU
        if not cla.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('Current device:', self.device)

        # Render episodes
        self.env_name = cla.env
        self.save_periodic = cla.save_periodic if cla.save_periodic else False
        
        # Number of Frames to Run
        if cla.frames:
            self.num_frames = cla.frames
        else:
            self.num_frames = 1_000_000

        # Synchronization
        # Overwrite sync from command line if value is passed
        if cla.sync_period is not None:
            self.rl_to_ea_synch_period = cla.sync_period
        else:
            self.rl_to_ea_synch_period = 1

        # Novelty Search
        self.ns = cla.novelty
        self.ns_epochs = 10

        # Model save frequency if save is active
        self.next_save = cla.next_save

        # ==================================  DDPG legacy Params =============================================
        self.use_ddpg = cla.use_ddpg   # default should be False

        self.gamma = 0.98
        self.tau = 0.005   
        self.seed = cla.seed
        self.batch_size = 256
        self.frac_frames_train = 1.0
        self.use_done_mask = True
        self.buffer_size = 200_000  #50000
        self.noise_sd = 0.1
        self.use_ounoise = cla.use_ounoise

        # hidden layer
        self.hidden_size = 128

        # Prioritised Experience Replay
        self.per = cla.per
        self.replace_old = True
        self.alpha = 0.7
        self.beta_zero = 0.5
        self.learn_start = (1 + self.buffer_size / self.batch_size) * 2
        # self.total_steps = self.num_frames

        # ==================================    TD3 Params  =============================================
        self.policy_update_freq = 2    # minimum for TD3
        self.noise_clip         = 0.5  # default

        # =================================   NeuroEvolution Params =====================================

        # Num of trials
        if cla.env == 'Walker2d-v2':
            self.num_evals = 5
        else:
            self.num_evals = 3

        # Elitism Rate
        self.elite_fraction = 0.2
 
        # Number of actors in the population
        self.pop_size = 30

        # Mutation and crossover
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9
        self.mutation_mag = cla.mut_mag
        self.mutation_batch_size = 256
        self.proximal_mut = cla.proximal_mut
        self.distil = cla.distil
        self.distil_type = cla.distil_type
        self._verbose_mut = cla.verbose_mut
        self._verbose_crossover = cla.verbose_crossover

        # Genetic memory size
        self.individual_bs = 10_000

        # Variation operator statistics
        self.opstat = cla.opstat
        self.opstat_freq = 1
        self.test_operators = cla.test_operators

        # Save Results
        self.state_dim = None  # To be initialised externally
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
  