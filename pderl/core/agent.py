import numpy as np
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
from core import replay_memory
from core import ddpg as ddpg
from scipy.spatial import distance
from core import replay_memory
from parameters import Parameters



class Agent:
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Init population
        self.pop = []
        self.buffers = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.GeneticAgent(args))

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
            self.noise_process = ddpg.GaussianNoise(args.action_dim, sd = args.noise_sd)

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
    

    def evaluate(self,agent: ddpg.GeneticAgent or ddpg.DDPG, is_render=False, is_action_noise=False,
                 store_transition=True) -> tuple:
        """ Play one game to evaualute the agent.

        Args:
            agent (ddpg.GeneticAgentorddpg.DDPG): Agent class. 
            is_render (bool, optional): Show render. Defaults to False.
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
            if self.args.render and is_render: 
                self.env.render()

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

    def rl_to_evo(self, rl_agent: ddpg.DDPG, evo_net: ddpg.GeneticAgent):
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

    def train_ddpg(self):
        bcs_loss, pgs_loss = [], []
        if len(self.replay_buffer) > self.args.batch_size * 5:  # agent has seen some experiences already
            for _ in range(int(self.gen_frames * self.args.frac_frames_train)):
                batch = self.replay_buffer.sample(self.args.batch_size)

                pgl, delta = self.rl_agent.update_parameters(batch)
                pgs_loss.append(pgl)

        return {'bcs_loss': 0, 'pgs_loss': pgs_loss}

    def train(self):
        self.gen_frames = 0
        self.iterations += 1

        '''+++++++++++++++++++++++++++++++   EVOLUTION    +++++++++++++++++++++++++++++++++++++++++++'''
        # Evaluate genomes/individuals
        rewards = np.zeros(len(self.pop))
        errors  = np.zeros(len(self.pop))
        # -loop over population AND store experiences
        for i, net in enumerate(self.pop):   
            for _ in range(self.args.num_evals):
                episode = self.evaluate(net, is_render=False, is_action_noise=False)
                rewards[i] += episode['reward']
                errors[i] += episode['td_error']

        rewards /= self.args.num_evals
        errors /= self.args.num_evals

        # all_fitness = 0.8 * rankdata(rewards) + 0.2 * rankdata(errors)
        all_fitness = rewards

        # Validation test for NeuroEvolution 
        # champion -- highest reward
        #  population_avg -- avergae over the entire agent population
        best_train_fitness = np.max(rewards)
        population_avg = np.average(rewards)
        champion = self.pop[np.argmax(rewards)]

        # print("Best TD Error:", np.max(errors))

        # Evaluate the champion
        test_scores = []
        for _ in range(self.validation_tests):
            # do NOT  store these trials
            episode = self.evaluate(champion, is_render=True, is_action_noise=False, store_transition=False)
            test_scores.append(episode['reward'])
        test_score = np.average(test_scores)
        test_sd = np.std(test_scores)

        # NeuroEvolution's probabilistic selection and recombination step
        elite_index = self.evolver.epoch(self.pop, all_fitness)

        ''' +++++++++++++++++++++++++++++++   DDPG    +++++++++++++++++++++++++++++++++++++++++++'''
        # Collect experience for training
        self.evaluate(self.rl_agent, is_action_noise=True)

        losses = self.train_ddpg()

        # Validation test for RL agent
        tests_ddpg = []
        for _ in range(self.validation_tests):
            # do NOT  store these trials
            ddpg_stats = self.evaluate(self.rl_agent, store_transition=False, is_action_noise=False)
            tests_ddpg.append(ddpg_stats['reward'])
        ddpg_reward = np.average(tests_ddpg)
        ddpg_std = np.std(tests_ddpg)

        # Sync RL Agent to NE every few steps
        if self.iterations % self.args.rl_to_ea_synch_period == 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(all_fitness)
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
            'elite_index': elite_index,
            'ddpg_reward': ddpg_reward,
            'ddpg_std':    ddpg_std,
            'pg_loss':     np.mean(losses['pgs_loss']),
            'bc_loss':     np.mean(losses['bcs_loss']),
            'pop_novelty': 0.,
        }


class Agent_ddpg:
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

    def evaluate(self,agent: ddpg.GeneticAgent or ddpg.DDPG, is_render=False, is_action_noise=False,
                 store_transition=True) -> tuple:
        """ Play one game to evaualute the agent.

        Args:
            agent (ddpg.GeneticAgentorddpg.DDPG): Agent class. 
            is_render (bool, optional): Show render. Defaults to False.
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
            if self.args.render and is_render: 
                self.env.render()

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
            for _ in range(int(self.gen_frames * self.args.frac_frames_train)):
                
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