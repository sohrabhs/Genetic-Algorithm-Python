class GeneticAlgorithm_FeatureSelection_WrapperMethod(GeneticAlgorithm):
    def __init__(self, X_train, y_train, X_test, y_test, wrapperModel, iteration, populationSize,
                 crossover_rate, mutation_rate, silent=True):
        self.iteration = iteration
        self.popSize = populationSize
        self.rc = crossover_rate
        self.rm = mutation_rate
        self.silent = silent
        self.wrapperModel = wrapperModel
        features = [i + 1 for i in range(0, X_train.shape[1])]
        self.features = features
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.featureSize = len(features)
        self.selected_features = None
        self.max_accuracy = None
        self.best_equal_combinations_set = set()
        super().__init__(iteration, populationSize, crossover_rate, mutation_rate, minimization=False, silent=silent)

    def optimize(self):
        super().optimize()
        self.selected_features = sorted(self.bestSolution[:self.bestSolution[-1]])
        self.max_accuracy = self.bestObj

    def initial_solution(self):
        pop = [0] * self.popSize
        for i in range(len(pop)):
            pop[i] = np.random.permutation(self.features).tolist() + [
                np.random.randint(1, self.featureSize + 1)]
        return pop

    def objective_function(self, pop):
        obj = [0] * len(pop)
        for i in range(len(pop)):
            chrome = pop[i]
            permutation = chrome[:-1]
            subsetSize = chrome[-1]
            selected_features = permutation[:subsetSize]
            X_train, y_train, X_test, y_test = self.feature_selected(self.X_train, self.y_train, self.X_test,
                                                                        self.y_test, selected_features)
            obj[i] = self.accuracy_calc(X_train, y_train, X_test, y_test, self.wrapperModel)
        return obj

    def crossover(self, chrome1, chrome2):
        parent1 = chrome1[:-1]
        parent2 = chrome2[:-1]
        r = np.random.randint(1, len(parent1) - 1, 2)
        while r[0] == r[1]:
            r = np.random.randint(1, len(parent1) - 1, 2)
        r = sorted(r)

        child1 = parent1[:r[0]]
        for p in range(len(parent1[r[0]:r[1]])):
            for k in range(len(parent1[r[0]:r[1]])):
                if parent2[r[0]:r[1]][k] not in child1:
                    child1 = child1 + [parent2[r[0]:r[1]][k]]
                elif parent1[r[0]:r[1]][k] not in child1:
                    child1 = child1 + [parent1[r[0]:r[1]][k]]
        for p in range(len(parent1[r[1]:])):
            for k in range(len(parent1[r[1]:])):
                if parent2[r[1]:][k] not in child1:
                    child1 = child1 + [parent2[r[1]:][k]]
                elif parent1[r[1]:][k] not in child1:
                    child1 = child1 + [parent1[r[1]:][k]]
        if len(child1) < len(parent2):
            for g in parent2:
                if g not in child1:
                    child1 = child1 + [g]
        child1 = child1 + [chrome1[-1]]

        child2 = parent2[:r[0]]
        for p in range(len(parent2[r[0]:r[1]])):
            for k in range(len(parent2[r[0]:r[1]])):
                if parent1[r[0]:r[1]][k] not in child2:
                    child2 = child2 + [parent1[r[0]:r[1]][k]]
                elif parent2[r[0]:r[1]][k] not in child2:
                    child2 = child2 + [parent2[r[0]:r[1]][k]]
        for p in range(len(parent2[r[1]:])):
            for k in range(len(parent2[r[1]:])):
                if parent1[r[1]:][k] not in child2:
                    child2 = child2 + [parent1[r[1]:][k]]
                elif parent2[r[1]:][k] not in child2:
                    child2 = child2 + [parent2[r[1]:][k]]
        if len(child2) < len(parent1):
            for g in parent1:
                if g not in child2:
                    child2 = child2 + [g]
        child2 = child2 + [chrome2[-1]]

        return child1, child2

    def mutation(self, chrome):
        parent = chrome[:-1]
        r = np.random.randint(0, len(parent), 2)
        while r[0] == r[1]:
            r = np.random.randint(0, len(parent), 2)
        g0 = parent[r[0]]
        g1 = parent[r[1]]
        parent[r[0]] = g1
        parent[r[1]] = g0
        parent = parent + [np.random.randint(1, len(parent) + 1, 1)[0]]
        return parent

    def feature_selected(self, X_train, y_train, X_test, y_test, selected):
	features = [i + 1 for i in range(0, X_train.shape[1])]
	mask = [i - 1 for i in features if i in selected]
	X_train = X_train[:, mask]
	y_train = y_train
	X_test = X_test[:, mask]
	y_test = y_test
	return X_train, y_train, X_test, y_test

