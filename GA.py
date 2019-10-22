import numpy as np

class GeneticAlgorithm:
    def __init__(self, it=100, popSize=100, rc=0.8, rm=0.05, minimization=True, silent=True):
        self.minimization = minimization
        self.silent = silent
        self.it = it
        self.popSize = popSize
        self.nc = round(rc * popSize * 2) / 2
        self.nm = round(rm * popSize)
        self.bestSolution = None
        self.bestObj = None

    def sortedFirstBySecond(self, first, second, reverse=False):
        index = np.array(sorted(range(len(second)), key=lambda k: second[k], reverse=reverse))
        second = np.array(sorted(second, reverse=reverse))
        first = np.array(first)
        first = first[index]
        first = first.tolist()
        second = second.tolist()
        return first, second

    def optimize(self):
        self.pop = self.initial_solution()
        self.obj = self.objective_function(self.pop)
        for p in range(0, self.it):
            # cross-over -------
            for cr in range(1, round(self.nc / 2)):
                selection = self.selection_roulette_wheel(2)
                parent1 = self.pop[selection[0]]
                parent2 = self.pop[selection[1]]
                child1, child2 = self.crossover(parent1, parent2)

                try:
                    pop_c = pop_c + list([child1, child2])
                except Exception:
                    pop_c = [child1, child2]

            # # mutation-----
            for cr in range(1, round(self.nm)):
                selection = self.selection_roulette_wheel(1)
                parent = self.pop[selection[0]]
                child = self.mutation(parent)
                try:
                    pop_m = pop_m + [child]
                except Exception:
                    pop_m = [child]

            try:
                pop = self.pop + pop_c + pop_m
            except Exception:
                pop = self.pop + pop_c

            obj = self.objective_function(pop)
            if self.minimization:
                pop, obj = self.sortedFirstBySecond(pop, obj, reverse=False)
            else:
                pop, obj = self.sortedFirstBySecond(pop, obj, reverse=True)

            # remove dominated population
            self.pop = pop[:self.popSize]
            self.obj = obj[:self.popSize]

            if not self.silent:
                print("it", p + 1, "obj", self.obj[0])
        self.bestSolution = self.pop[0]
        self.bestObj = self.obj[0]

    def selection_roulette_wheel(self, n):
        obj = np.array(self.obj)
        if self.minimization:
            obj = 1 / obj
            pdf = obj / sum(obj)
        else:
            pdf = obj / sum(obj)
        cdf = np.cumsum(pdf)
        selection = []
        for j in range(0, n):
            np.random.seed(None)
            r = np.random.random(1)
            for i in range(0, len(cdf)):
                if r <= cdf[i]:
                    selection.append(i)
                    break
        return selection

    def initial_solution(self):
        pass

    def objective_function(self, pop):
        pass

    def crossover(self, parent1, parent2):
        pass

    def mutation(self, parent):
        pass

