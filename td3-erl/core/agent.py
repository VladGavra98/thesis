from core import genetic_agent, mod_utils, replay_memory
from core import mod_neuro_evo as utils_ne
from core import ddpg as ddpg
from core import td3 as td3
from parameters import Parameters
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm
import os


@dataclass                                                                                                                                      
class Episode: 
    """ Output of one episode. 
    """                                                                                                                     
    bcs : Tuple[np.float64]                                                                                                                           
    reward : float                                                                                                                         
    length        : np.float64 
    state_history : List
    ref_signals   : List
    actions       : List
    reward_lst    : List


class Agent:
    def __init__(self, args: Parameters, environment):
        self.args = args; 
        self.env = environment

        # Init population
        self.pop : List = []
        self.buffers : List = []
        self.pop = [genetic_agent.GeneticAgent(args) for _ in range(args.pop_size)]

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
            self.noise_process = mod_utils.GaussianNoise(args.action_dim, sd = args.noise_sd)

        # Initialise evolutionary loop
        if len(self.pop):
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
        self.num_episodes = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None
        self.rl_iteration = 0            # for TD3 delyed policy updates
        self.champion : genetic_agent.GeneticAgent = None
        self.champion_actor: genetic_agent.Actor   = None
        self.champion_history : np.ndarray = None


    def evaluate (self,
                  agent: genetic_agent.GeneticAgent or ddpg.DDPG or td3.TD3, 
                  is_action_noise : bool,
                  store_transition : bool) -> Episode:
        """ Play one game to evaluate the agent.

        Args:
            agent (GeneticAgentor): Agent class. 
            is_action_noise (bool): Add Gaussian/OU noise to action.
            store_transition (bool, optional): Add frames to memory buffer for training. Defaults to True.

        Returns:
            Episode: data class with the episode stats
        """
        # init states, env and 
        rewards, state_lst, action_lst = [],[], []
        obs = self.env.reset()
        done = False

        # actor for evaluation 
        agent.actor.eval()

        while not done: 
            # start with 0 for stability
            # select  actor ation
            action = agent.actor.select_action(np.array(obs))

            # add exploratory noise
            if is_action_noise:
                clipped_noise = np.clip(self.args.noise_sd * np.random.randn(action.shape[0]),\
                                        - self.args.noise_clip, self.args.noise_clip)
                action = np.clip(action + clipped_noise, -1.0, 1.0)

            # Simulate one step in environment
            next_obs, reward, done, info = self.env.step(action.flatten())
            rewards.append(reward)
            action_lst.append(self.env.last_u)  # actuator deflection 

            # Add experiences to buffer:
            if store_transition:
                # store for training
                transition = (obs, action, next_obs, reward, float(done))
                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)
                self.num_frames += 1; self.gen_frames += 1
            else:
                # save for future validation
                state_lst.append(self.env.x)
                
            # update agent obs
            obs = next_obs

        # End env NOTE might be very relevant 
        self.env.finish()

        # updated episodes if is done
        if store_transition: 
            self.num_episodes += 1

        # Compute BCs:
        actions = np.asarray(action_lst)
        bcs = np.sum(np.abs(actions), axis = 0)

        return Episode(reward = np.sum(rewards), bcs = bcs, length = info['t'],\
                       state_history=state_lst, ref_signals = info['ref'], \
                       actions = actions, reward_lst = rewards)

    def rl_to_evo(self, rl_agent: ddpg.DDPG or td3.TD3, evo_net: genetic_agent.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)


    def get_pop_novelty(self, bcs : np.array):
        return np.sum(np.std(bcs, axis = 0))/bcs.shape[1]

    def train_rl(self) -> Dict[float, float]:
        """ Train the RL agent on the same number of frames seens by the entire actor populaiton during the last generation.
            The frames are sampled from the common buffer.
        """
        
        pgs_obj, TD_loss = [],[]

        if len(self.replay_buffer) > self.args.learn_start: 
            print('Train RL agent ...')
            # prepare for training
            self.rl_agent.actor.train()

            # select target policy
            if self.args.use_champion_target:
                if self.champion_actor is not None:
                    self.evo_to_rl(self.rl_agent.actor_target, self.champion_actor)
            else:
                self.champion_actor = None 

            # train over generation experiences
            for _ in tqdm(range(int(self.gen_frames * self.args.frac_frames_train))):
                self.rl_iteration+=1

                batch = self.replay_buffer.sample(self.args.batch_size)
                pgl, TD = self.rl_agent.update_parameters(batch, self.rl_iteration, self.champion_actor)

                if pgl is not None: pgs_obj.append(-pgl)
                if TD is not None: TD_loss.append(TD)
                
        return {'PG_obj': np.mean(pgs_obj), 'TD_loss': np.median(TD_loss)}

    def validate_agent (self, agent : genetic_agent.Actor) -> Tuple[float, float, Episode]:
        """ Evaluate the  given actor and do NOT store these trials. 
        """
        test_scores, bcs = [], []

        for _ in range(self.validation_tests):
            last_episode = self.evaluate(agent, is_action_noise = False,\
                                        store_transition = False) 
            test_scores.append(last_episode.reward)
            bcs.append(last_episode.bcs)

        test_score = np.mean(test_scores)
        test_sd = np.std(test_scores)

        return test_score,test_sd, last_episode

    def get_history (self, episode : Episode) -> np.ndarray:
        time = np.linspace(0, episode.length, len(episode.state_history))
        ref_values = np.array([[ref(t_i) for t_i in time] for ref in episode.ref_signals]).transpose()
        reward_lst = np.asarray(episode.reward_lst).reshape((len(episode.state_history),1))

        return np.concatenate((ref_values, episode.actions,episode.state_history,reward_lst), axis = 1)

    def train(self):
        self.gen_frames = 0
        self.iterations += 1
        
        lengths = []

        ''' +++++++++++++++++++++++++++++++   RL  ++++++++++++++++++++++++++++++++++++++'''
        # Collect extra experience for RL training 
        rl_extra_evals = 5
        rewards = np.zeros((rl_extra_evals))
        bcs     = np.zeros((rl_extra_evals,3))

        print('Info: playing extra episodes with action nosie for RL training')
        for i in range(rl_extra_evals):
            episode = self.evaluate(self.rl_agent, is_action_noise=True, store_transition=True)
            lengths.append(episode.length)
            rewards[i] = episode.reward
            bcs[i,:] = episode.bcs

        print(f'RL training reward: {np.average(rewards):0.1f}')
        ep_len_avg = np.average(lengths); ep_len_sd = np.std(lengths)

        self.evaluate(self.rl_agent, is_action_noise = True, store_transition=True)

        # Gradient updates of RL actor and critic:
        rl_train_scores = self.train_rl()

        # Validate RL actor separately:
        rl_reward,rl_std, rl_episode = self.validate_agent(self.rl_agent)
        self.rl_history = self.get_history(rl_episode)

        # -------------------------- Collect statistics --------------------------
        return {
            'rl_reward':   rl_reward,
            'rl_std':      rl_std,
            'avg_ep_len':  ep_len_avg, 
            'ep_len_sd':   ep_len_sd, 
            'PG_obj':      rl_train_scores['PG_obj'],
            'TD_loss':     rl_train_scores['TD_loss'],
        }


    def save_agent (self, parameters: object, elite_index: int = None) -> None:
        """ Save the trained agent(s).

        Args:
            parameters (object): Container class of the trainign hyperparameters.
            elite_index (int: Index of the best performing agent i.e. the champion. Defaults to None.
        """
        # Save gentic popualtion
        if len(self.pop):
            pop_dict = {}
            for i, ind in enumerate(self.pop):
                pop_dict[f'actor_{i}'] = ind.actor.state_dict()
            torch.save(pop_dict, os.path.join(
                parameters.save_foldername, 'evo_nets.pkl'))

            # Save best performing agent separately:
            torch.save(self.pop[elite_index].actor.state_dict(), 
                        os.path.join(parameters.save_foldername,'elite_net.pkl'))

            # Save state history of the champion
            filename = 'statehistory_episode' + str(self.num_episodes) + '.txt'
            np.savetxt(os.path.join(parameters.save_foldername,filename),
                self.champion_history, header = str(self.num_episodes))

        
        # Save RL actor seprately:
        torch.save(self.rl_agent.actor.state_dict(), 
                        os.path.join(parameters.save_foldername,'rl_net.pkl'))

        filename = 'rl_statehistory_episode' + str(self.num_episodes) + '.txt'
        np.savetxt(os.path.join(parameters.save_foldername,filename),
                self.rl_history, header = str(self.num_episodes))

        # NOTE might want to save RL state-history for future cheks
        print('> Saved state history in ' + str(filename) + '\n')



    # def get_pop_novelty(self): OLD
    #     epochs = self.args.ns_epochs
    #     novelties = np.zeros(len(self.pop))
    #     for _ in range(epochs):
    #         transitions = self.replay_buffer.sample(self.args.batch_size)
    #         batch = replay_memory.Transition(*zip(*transitions))
    #         # each agent novelty
    #         for i, net in enumerate(self.pop):
    #             novelties[i] += (net.get_novelty(batch))
    #     return novelties / epochs