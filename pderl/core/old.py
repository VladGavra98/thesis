def crossover_inplace(self, gene1: GeneticAgent, gene2: GeneticAgent):
    # Evaluate the parents
    trials = 5
    if self.args.opstat and self.stats.should_log():
        test_score_p1 = 0
        for _ in range(trials):
            episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
            test_score_p1 += episode['reward']
        test_score_p1 /= trials

        test_score_p2 = 0
        for _ in range(trials):
            episode = self.evaluate(gene2, is_render=False, is_action_noise=False, store_transition=False)
            test_score_p2 += episode['reward']
        test_score_p2 /= trials

    for param1, param2 in zip(gene1.actor.parameters(), gene2.actor.parameters()):
        # References to the variable tensors
        W1 = param1.data
        W2 = param2.data

        if len(W1.shape) == 2: #Weights no bias
            num_variables = W1.shape[0]
            # Crossover opertation [Indexed by row]
            num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
            for i in range(num_cross_overs):
                receiver_choice = random.random()  # Choose which gene to receive the perturbation
                if receiver_choice < 0.5:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W1[ind_cr, :] = W2[ind_cr, :]
                else:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W2[ind_cr, :] = W1[ind_cr, :]

        elif len(W1.shape) == 1: #Bias
            num_variables = W1.shape[0]
            # Crossover opertation [Indexed by row]
            num_cross_overs = fastrand.pcg32bounded(num_variables)  # Lower bounded on full swaps
            for i in range(num_cross_overs):
                receiver_choice = random.random()  # Choose which gene to receive the perturbation
                if receiver_choice < 0.5:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W1[ind_cr] = W2[ind_cr]
                else:
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                    W2[ind_cr] = W1[ind_cr]

    # Evaluate the children
    if self.args.opstat and self.stats.should_log():
        test_score_c1 = 0
        for _ in range(trials):
            episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
            test_score_c1 += episode['reward']
        test_score_c1 /= trials

        test_score_c2 = 0
        for _ in range(trials):
            episode = self.evaluate(gene1, is_render=False, is_action_noise=False, store_transition=False)
            test_score_c2 += episode['reward']
        test_score_c2 /= trials

        if self.args.verbose_crossover:
            print("==================== Classic Crossover ======================")
            print("Parent 1", test_score_p1)
            print("Parent 2", test_score_p2)
            print("Child 1", test_score_c1)
            print("Child 2", test_score_c2)

        self.stats.add({
            'cros_parent1_fit': test_score_p1,
            'cros_parent2_fit': test_score_p2,
            'cros_child_fit': np.mean([test_score_c1, test_score_c2]),
            'cros_child1_fit': test_score_c1,
            'cros_child2_fit': test_score_c2,
        })
