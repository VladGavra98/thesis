import numpy as np
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
from core import replay_memory
from core import ddpg as ddpg
from core import td3 as td3
from core import replay_memory
from parameters import Parameters
from pderl.core import genetic_agent, mod_utils

from tqdm import tqdm

class Agent:
    def __init__(self, args: Parameters, wrapper):
        self.args = args; 
        self.env = wrapper.env

        # Init population
        self.pop = []
        self.buffers = []
        for _ in range(args.pop_size):
            self.pop.append(genetic_agent.GeneticAgent(args))

        # Define RL Agent
        print('Anget type: ' + ('DDPG' if args.use_ddpg else 'TD3'))
        if args.use_ddpg:
            self.rl_agent = ddpg.DDPG(args)
        else:
            self.rl_agent = td3.TD3(args)

        # Define Memory Buffer:
        if args.per:
            self.replay_buffer = replay_memory.PrioritizedReplayMemory(args.buffer_size, args.device,
                                                                       beta_frames=self.args.num_frames)
        else:
            self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size, args.device)

        # Define noise process:
        if args.use_ounoise:
            print('Using OU noise')
            self.noise_process = mod_utils.OUNoise(args.action_dim)
        else:
            print('Using Gaussian noise')
            self.noise_process = mod_utils.GaussianNoise(args.action_dim, sd = args.noise_sd)

        # Initialise evolutionary loop
        self.evolver = utils_ne.SSNE(self.args, self.rl_agent.critic, self.evaluate)

        # Testing
        self.validation_tests = 5

        # Population novelty
        self.ns_r = 1.0
        self.ns_delta = 0.1
        self.best_train_reward = 0.0
        self.time_since_improv = 0
        self.step = 1

        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None
        self.rl_iteration = 0 # for TD3 delyed policy updates

    def evaluate(self,agent: genetic_agent.GeneticAgent or ddpg.DDPG or td3.TD3, 
                is_action_noise=False,
                 store_transition = True) -> tuple:
        """ Play one game to evaualute the agent.

        Args:
            agent (GeneticAgentor): Agent class. 
            is_action_noise (bool, optional): Add OU noise to action. Defaults to False.
            store_transition (bool, optional): Add frames to memory buffer. Defaults to True.

        Returns:
            tuple: Reward, temporal difference error
        """
        total_reward = 0.0

        state = self.env.reset()
        done = False

        while not done: 
            # play one 'game'

            action = agent.actor.select_action(np.array(state))
            if is_action_noise:
                clipped_noise = np.clip(self.noise_process.noise(),-self.args.noise_clip, self.args.noise_clip)
                action += clipped_noise
                action = np.clip(action, -1.0, 1.0)

            # Simulate one step in environment
            next_state, reward, done, _ = self.env.step(action.flatten())
            total_reward += reward

            # Add experiences to buffer:
            if store_transition:
                transition = (state, action, next_state, reward, float(done))
                self.num_frames += 1; self.gen_frames += 1
                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)

            state = next_state

        # updated games if is done
        if store_transition: 
            self.num_games += 1

        return {'reward': total_reward}

    def rl_to_evo(self, rl_agent: ddpg.DDPG or td3.TD3, evo_net: genetic_agent.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self):
        epochs = self.args.ns_epochs
        novelties = np.zeros(len(self.pop))
        for _ in range(epochs):
            transitions = self.replay_buffer.sample(self.args.batch_size)
            batch = replay_memory.Transition(*zip(*transitions))
            # each agent novelty
            for i, net in enumerate(self.pop):
                novelties[i] += (net.get_novelty(batch))
        return novelties / epochs

    def train_rl(self):
        """ Train the RL agent on the same number of frames seens by the entire actor populaiton during the last generation.
            The frames are sampled from the common buffer.
        """
        print('Train RL agent ...')
        pgs_obj, TD_loss = [], []
        if len(self.replay_buffer) > self.args.batch_size * 5:  # agent has seen some experiences already
            for _ in tqdm(range(int(self.gen_frames))):
                self.rl_iteration+=1
                batch = self.replay_buffer.sample(self.args.batch_size)

                pgl, TD = self.rl_agent.update_parameters(batch, self.rl_iteration)

                if pgl is not None:
                    pgs_obj.append(-pgl)
                TD_loss.append(TD)

        return {'PG_obj': pgs_obj, 'TD_loss': TD_loss}

    def train(self):
        self.gen_frames = 0
        self.iterations += 1

        '''+++++++++++++++++++++++++++++++++   Evolution   +++++++++++++++++++++++++++++++++++++++++++'''
        rewards     = np.zeros(len(self.pop))
        test_scores = np.zeros(self.validation_tests)
        tests_rl    = np.zeros(self.validation_tests)

        # Evaluate genomes/individuals
        # -loop over population AND store experiences
        for i, net in enumerate(self.pop):   
            for _ in range(self.args.num_evals):
                episode = self.evaluate(net, is_action_noise=False)
                rewards[i] += episode['reward']
        rewards /= self.args.num_evals
  

        # Validation test for NeuroEvolution 
        best_train_fitness  = np.max(rewards)  #  champion -- highest reward
        worst_train_fitness = np.min(rewards)
        population_avg      = np.average(rewards)    #  population_avg -- average over the entire agent population
        champion            = self.pop[np.argmax(rewards)]


        # Evaluate the champion
        for i in range(self.validation_tests):
            # do NOT  store these trials
            episode = self.evaluate(champion, is_action_noise=False, store_transition=False)
            test_scores[i] = episode['reward']
        test_score = np.average(test_scores); test_sd = np.std(test_scores)

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, rewards)

        ''' +++++++++++++++++++++++++++++++   RL (DDPG | TD3)    +++++++++++++++++++++++++++++++++++++++++++'''
        # Collect experience for training
        self.evaluate(self.rl_agent, is_action_noise=True)

        # Gradient update
        losses = self.train_rl()

        # Validation test for RL agent
        for _ in range(self.validation_tests):
            rl_stats = self.evaluate(self.rl_agent, store_transition=False, is_action_noise=False)   # do NOT  store these trials
            tests_rl[i]= rl_stats['reward']
        rl_reward = np.average(tests_rl); rl_std = np.std(tests_rl)


        ''' +++++++++++++++++++++++++++++++   Actor Injection   +++++++++++++++++++++++++++++++++++++++++++'''
        if self.iterations % self.args.rl_to_ea_synch_period == 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(rewards)

            if replace_index == elite_index:
                replace_index = (replace_index + 1) % len(self.pop)

            self.rl_to_evo(self.rl_agent, self.pop[replace_index])
            self.evolver.rl_policy = replace_index
            print('Sync from RL --> Evolution')

        # # Get popualtion nvelty:
        # TODO: chek this later
        # pop_novelty = self.get_pop_novelty()
        # -------------------------- Collect statistics --------------------------
        return {
            'best_train_fitness': best_train_fitness,
            'test_score':  test_score,
            'test_sd':     test_sd,
            'pop_avg':     population_avg,
            'pop_min':     worst_train_fitness,
            'elite_index': elite_index,
            'rl_reward':  rl_reward,
            'rl_std':     rl_std,
            'PG_obj':     np.mean(losses['PG_obj']),
            'TD_loss':     np.mean(losses['TD_loss']),
            'pop_novelty': 0.,
        }


class Agent_ddpg:
    """  Pure RL agent using DDPG """
    # TODO: change this later to include TD3 for comparison
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Init population
        self.buffers = []

        # Init RL Agent
        self.rl_agent = ddpg.DDPG(args)
        if args.per:
            self.replay_buffer = replay_memory.PrioritizedReplayMemory(args.buffer_size, args.device,
                                                                       beta_frames=self.args.num_frames)
        else:
            self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size, args.device)

        # Define noise process:
        if args.use_ounoise:
            print('Usign OU noise')
            self.noise_process = ddpg.OUNoise(args.action_dim)
        else:
            print('Using Gaussian noise')
            self.noise_process = ddpg.GaussianNoise(args.action_dim, sd = self.rl_agent.noise_sd)

        # Testing:
        self.validation_freq  = 30  # let it equal to the population size for better comaprison
        self.validation_tests = 5

        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None
    
    def evaluate(self,agent: genetic_agent.GeneticAgent or ddpg.DDPG, is_action_noise=False,
                 store_transition=True) -> tuple:
        """ Play one game to evaualute the agent.

        Args:
            agent (ddpg.GeneticAgentorddpg.DDPG): Agent class. 
            is_action_noise (bool, optional): Add OU noise to action. Defaults to False.
            store_transition (bool, optional): Add frames to memory buffer. Defaults to True.

        Returns:
            tuple: Reward, temporal difference error
        """
        total_reward = 0.0
        total_error = 0.0

        state = self.env.reset()
        done = False

        while not done:
            # play one 'game'

            action = agent.actor.select_action(np.array(state))
            if is_action_noise:
                action += self.noise_process.noise()
                action = np.clip(action, -1.0, 1.0)

            # Simulate one step in environment
            next_state, reward, done, info = self.env.step(action.flatten())
            total_reward += reward

            # Add experiences to buffer:
            if store_transition:
                transition = (state, action, next_state, reward, float(done))
                self.num_frames += 1; self.gen_frames += 1

                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)
            state = next_state

        # updated games if is done
        if store_transition: 
            self.num_games += 1

        return {'reward': total_reward, 'td_error': total_error}

    def train_ddpg(self):
        bcs_loss, pgs_loss = [], []
        if len(self.replay_buffer) > self.args.batch_size * 5:  # agent has seen some experiences already
            for _ in range(int(self.gen_frames)):
                
                batch = self.replay_buffer.sample(self.args.batch_size)
                pgl, delta = self.rl_agent.update_parameters(batch)
                pgs_loss.append(pgl)

        return {'bcs_loss': 0, 'pgs_loss': pgs_loss}

    def train(self):
        self.gen_frames = 0
        self.iterations += 1

        # Collect experience for training
        ddpg_stats = self.evaluate(self.rl_agent, is_action_noise=True)
        
        # Pass over the experiences:
        losses = self.train_ddpg()

        # Validation test for RL agent
        # do NOT  store these trials
        if self.iterations % self.validation_freq ==0:
            tests_ddpg = []
            for _ in range(self.validation_tests):
                ddpg_stats = self.evaluate(self.rl_agent, store_transition=False, is_action_noise=False)
                tests_ddpg.append(ddpg_stats['reward'])
            ddpg_reward = np.average(tests_ddpg)
            ddpg_std = np.std(tests_ddpg)
            

            # -------------------------- Collect statistics --------------------------
            return {
                'ddpg_reward': ddpg_reward,
                'ddpg_std':    ddpg_std,
                'pg_loss':     np.mean(losses['pgs_loss']),
            }

# class Archive:
#     """A record of past behaviour characterisations (BC) in the population"""

#     def __init__(self, args):
#         self.args = args
#         # Past behaviours
#         self.bcs = []

#     def add_bc(self, bc):
#         if len(self.bcs) + 1 > self.args.archive_size:
#             self.bcs = self.bcs[1:]
#         self.bcs.append(bc)

#     def get_novelty(self, this_bc):
#         if self.size() == 0:
#             return np.array(this_bc).T @ np.array(this_bc)
#         distances = np.ravel(distance.cdist(np.expand_dims(this_bc, axis=0), np.array(self.bcs), metric='sqeuclidean'))
#         distances = np.sort(distances)
#         return distances[:self.args.ns_k].mean()

#     def size(self):
#         return len(self.bcs)