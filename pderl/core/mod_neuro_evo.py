import random
import numpy as np
from typing import List, Tuple, Dict
import fastrand
import torch
import torch.distributions as dist
from core.mod_utils import hard_update, soft_update
from parameters import Parameters
import os

from core.genetic_agent import GeneticAgent

class SSNE:
    def __init__(self, args: Parameters, critic : torch.nn, evaluate : callable):
        self.current_gen = 0
        self.args = args
        self.critic = critic
        self.population_size = self.args.pop_size
        self.num_elitists = max(int(self.args.elite_fraction * args.pop_size),1)
        self.evaluate = evaluate
        self.stats = PopulationStats(self.args)
        
        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}

    def selection_tournament(self, index_rank : List[int], num_offsprings : int, tournament_size : int ) -> List[GeneticAgent]:
        """ Returns a list of non-elite offsprings.
        """
        total_choices = len(index_rank)
        offsprings = []
        for _ in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        weight = torch.clamp(weight, -mag, mag)

        return weight

    def distilation_crossover(self, gene1: GeneticAgent, gene2: GeneticAgent) -> GeneticAgent:
        new_agent = GeneticAgent(self.args)
        new_agent.buffer.add_latest_from(gene1.buffer, self.args.individual_bs // 2)
        new_agent.buffer.add_latest_from(gene2.buffer, self.args.individual_bs // 2)
        new_agent.buffer.shuffle()

        hard_update(new_agent.actor, gene2.actor)
        batch_size = min(128, len(new_agent.buffer))
        iters = len(new_agent.buffer) // batch_size
        losses = []
        for epoch in range(12):
            for i in range(iters):
                batch = new_agent.buffer.sample(batch_size)
                losses.append(new_agent.update_parameters(batch, gene1.actor, gene2.actor, self.critic))

        if self.args.opstat and self.stats.should_log():
            test_score_p1 = 0
            trials = 5
            for eval in range(trials):
                episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p1 += episode['reward']
            test_score_p1 /= trials

            test_score_p2 = 0
            for eval in range(trials):
                episode = self.evaluate(gene2, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p2 += episode['reward']
            test_score_p2 /= trials

            test_score_c = 0
            for eval in range(trials):
                episode = self.evaluate(new_agent, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c += episode['reward']
            test_score_c /= trials

            if self.args.verbose_crossover:
                print("==================== Distillation Crossover ======================")
                print("MSE Loss:", np.mean(losses[-40:]))
                print("Parent 1", test_score_p1)
                print("Parent 2", test_score_p2)
                print("Crossover performance: ", test_score_c)

            self.stats.add({
                'cros_parent1_fit': test_score_p1,
                'cros_parent2_fit': test_score_p2,
                'cros_child_fit': test_score_c,
            })

        return new_agent

    
    def proximal_mutate(self, gene: GeneticAgent, mag):
        # Based on code from https://github.com/uber-research/safemutations 
        trials = 5
        if self.stats.should_log():
            test_score_p = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_p += episode['reward']
            test_score_p /= trials

        model = gene.actor

        batch = gene.buffer.sample(min(self.args.mutation_batch_size, len(gene.buffer)))
        state, _, _, _, _ = batch
        output = model(state)

        params = model.extract_parameters()
        tot_size = model.count_parameters()
        num_outputs = output.size()[1]


        # initial perturbation
        normal = dist.Normal(torch.zeros_like(params), torch.ones_like(params) * mag)
        delta = normal.sample()


        # we want to calculate a jacobian of derivatives of each output's sensitivity to each parameter
        jacobian = torch.zeros(num_outputs, tot_size).to(self.args.device)
        grad_output = torch.zeros(output.size()).to(self.args.device)

        # do a backward pass for each output
        for i in range(num_outputs):
            model.zero_grad()
            grad_output.zero_()
            grad_output[:, i] = 1.0

            output.backward(grad_output, retain_graph=True)
            jacobian[i] = model.extract_grad()

        # summed gradients sensitivity
        scaling = torch.sqrt((jacobian**2).sum(0))
        scaling[scaling == 0] = 1.0
        scaling[scaling < 0.01] = 0.01
        delta /= scaling
        new_params = params + delta

        model.inject_parameters(new_params)

        if self.stats.should_log():
            test_score_c = 0
            for eval in range(trials):
                episode = self.evaluate(gene, is_render=False, is_action_noise=False, store_transition=False)
                test_score_c += episode['reward']
            test_score_c /= trials

            self.stats.add({
                'mut_parent_fit': test_score_p,
                'mut_child_fit': test_score_c,
            })

            if self.args.verbose_mutation:
                print("==================== Mutation ======================")
                print("Fitness before: ", test_score_p)
                print("Fitness after: ", test_score_c)
                print("Mean mutation change:", torch.mean(torch.abs(new_params - params)).item())

    def clone(self, master: GeneticAgent, replacee: GeneticAgent):  # Replace the replacee individual with master
        """ Copy weights from master to replacee.
        """
        for target_param, source_param in zip(replacee.actor.parameters(), master.actor.parameters()):
            target_param.data.copy_(source_param.data)
        replacee.buffer.reset()
        replacee.buffer.add_content_of(master.buffer)

    def reset_genome(self, gene: GeneticAgent):
        for param in (gene.actor.parameters()):
            param.data.copy_(param.data)

    @staticmethod
    def sort_groups_by_fitness(genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)
    
    @staticmethod
    def get_distance(gene1: GeneticAgent, gene2: GeneticAgent):
        batch_size = min(256, min(len(gene1.buffer), len(gene2.buffer)))
        batch_gene1 = gene1.buffer.sample_from_latest(batch_size, 1000)
        batch_gene2 = gene2.buffer.sample_from_latest(batch_size, 1000)

        return gene1.actor.get_novelty(batch_gene2) + gene2.actor.get_novelty(batch_gene1)
    
    @staticmethod
    def sort_groups_by_distance(genomes, pop):
        """ Adds all posssible parent-pairs to a group,
        then sorts them based on distance from largest to smallest.

        Args:
            genomes (_type_): Parent wieghts.
            pop (_type_): List of genetic actors.

        Returns:
            list : sorted groups from most different to msot similar
        """        
        groups = []

        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                groups.append((second, first, SSNE.get_distance(pop[first], pop[second])))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    def epoch(self, pop: List[GeneticAgent], fitness_evals : np.array or List[float]):
        """ Entire epoch is handled with indices; 
            Index ranks  nets by fitness evaluation - 0 is the best after reversing.
        """ 
        index_rank = np.argsort(fitness_evals)[::-1]
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard -- first indeces
        # print('Elites:', elitist_index)

        # Selection
        # offsprings are kep for crossover and mutation together with elites
        offsprings = self.selection_tournament(index_rank, 
                                               num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # COMPUTE RL_SELECTION RATE
        if self.rl_policy is not None: # RL Transfer happened
            self.selection_stats['total'] += 1.0

            if self.rl_policy in elitist_index: self.selection_stats['elite'] += 1.0
            elif self.rl_policy in offsprings: self.selection_stats['selected'] += 1.0
            elif self.rl_policy in unselects: self.selection_stats['discarded'] += 1.0
            self.rl_policy = None

        # Elitism 
        # >> assigning elite candidates to some unselects
        for i in elitist_index:
            try: replacee = unselects.pop(0)
            except: replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee])

        # Crossover 
        # >> between elite and offsprings for the unselected genes with 100 percent probability
        if 'fitness' in self.args.distil_type.lower():
            sorted_groups = SSNE.sort_groups_by_fitness(new_elitists + offsprings, fitness_evals)
        elif 'dist' in self.args.distil_type.lower():
            sorted_groups = SSNE.sort_groups_by_distance(new_elitists + offsprings, pop)
        else:
            raise NotImplementedError('Unknown distilation type')

        for i, unselected in enumerate(unselects):
            first, second, _ = sorted_groups[i % len(sorted_groups)]
            if fitness_evals[first] < fitness_evals[second]:
                first, second = second, first
            self.clone(self.distilation_crossover(pop[first], pop[second]), pop[unselected])

        # Crossover for selected offsprings
        if self.args.crossover_prob > 0.01:  # so far this is not called
            for i in offsprings:
                if random.random() < self.args.mutation_prob:
                    others = offsprings.copy()
                    others.remove(i)
                    off_j = random.choice(others)
                    self.clone(self.distilation_crossover(pop[i], pop[off_j]), pop[i])

        # Mutate all genes in the population 
        #  EXCEPT the new elitists
        for i in index_rank[self.num_elitists:]:
            if random.random() < self.args.mutation_prob:
                # print(f'actor {i} mutated - fitness: {fitness_evals[i]}')
                self.proximal_mutate(pop[i], mag=self.args.mutation_mag)


        if self.stats.should_log():
            self.stats.log()
        self.stats.reset()

        return new_elitists[0]


def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))


class PopulationStats:
    def __init__(self, args: Parameters, file='population.csv'):
        self.data = {}
        self.args = args
        self.save_path = os.path.join(args.save_foldername, file)
        self.generation = 0

        if not os.path.exists(args.save_foldername):
            os.makedirs(args.save_foldername)

    def add(self, res):
        for k, v in res.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log(self):
        with open(self.save_path, 'a+') as f:
            if self.generation == 0:
                f.write('generation,')
                for i, k in enumerate(self.data):
                    if i > 0:
                        f.write(',')
                    f.write(k)
                f.write('\n')

            f.write(str(self.generation))
            f.write(',')
            for i, k in enumerate(self.data):
                if i > 0:
                    f.write(',')
                f.write(str(np.mean(self.data[k])))
            f.write('\n')

    def should_log(self):
        return self.generation % self.args.opstat_freq == 0 and self.args.opstat

    def reset(self):
        for k in self.data:
            self.data[k] = []
        self.generation += 1


