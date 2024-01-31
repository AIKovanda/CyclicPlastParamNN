import abc
import hashlib
import json
import pickle
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import fmin
from tqdm import tqdm

from rcpl.config import BASE_DIR


class SearchSpace:
    def __init__(self, yaml_setup: str, sort_values: list[tuple] = None):
        self.setup = yaml.unsafe_load(yaml_setup)  # list for interval, set for discrete
        self.n = len(self.setup)
        self.column_names = list(self.setup.keys())
        self.column_types = [eval(i['type']) for i in self.setup.values()]
        self.column_ranges = [eval(i['range']) for i in self.setup.values()]
        self.sort_values = sort_values if sort_values is not None else []

    def sample(self) -> np.ndarray:
        res = np.zeros(self.n)
        for i in range(self.n):
            if isinstance(self.column_ranges[i], list):
                res[i] = np.random.uniform(self.column_ranges[i][0], self.column_ranges[i][1])
            elif isinstance(self.column_ranges[i], set):
                res[i] = np.random.choice(list(self.column_ranges[i]))
            else:
                raise ValueError('Invalid setup')

        return self.sort_individual(res)

    def sort_individual(self, res: np.ndarray):
        res = res.copy()
        for sort_column_names in self.sort_values:
            sort_column_ids = [self.column_names.index(i) for i in sort_column_names]
            res[sort_column_ids] = sorted(res[sort_column_ids])
        return res

    def is_valid(self, x):
        for i in range(self.n):
            if isinstance(self.column_ranges[i], list):
                if x[i] < self.column_ranges[i][0] or x[i] > self.column_ranges[i][1]:
                    return False
            elif isinstance(self.column_ranges[i], set):
                if x[i] not in self.column_ranges[i]:
                    return False
            else:
                raise ValueError('Invalid setup')
        return True

    def closest_valid(self, x):
        res = np.zeros(self.n)
        for i in range(self.n):
            if isinstance(self.column_ranges[i], list):
                res[i] = np.clip(x[i], self.column_ranges[i][0], self.column_ranges[i][1])
            elif isinstance(self.column_ranges[i], set):
                res[i] = min(self.column_ranges[i], key=lambda y: abs(x[i] - y))
            else:
                raise ValueError('Invalid setup')
        return res


class OptimizationMethod:
    def __init__(self, func, search_space: SearchSpace, run_name, save_path, dataset, csv_extract_columns=None, **kwargs):
        self.__func = func
        self.kwargs = kwargs
        self.search_space = search_space
        self.cache = {}  # Dictionary to store experiment setups and results
        self.meta = {}
        self.cache_definitions = {}  # Dictionary to store experiment setup definitions
        self.called_n = 0
        self.dataset = dataset
        self.run_name = run_name
        self.run_meta = {}
        self.save_path = save_path
        self.csv_extract_columns = csv_extract_columns if csv_extract_columns is not None else {}

    def eval_func(self, individual, save=False):
        assert self.search_space.is_valid(individual), f'Invalid individual: {individual} | {self.search_space.column_names}'
        kwargs = {
                self.search_space.column_names[i]: self.search_space.column_types[i](individual[i])
                for i in range(len(individual))
            }
        hash_ = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode('utf-8')).hexdigest()[:8]
        if hash_ in self.cache:
            return self.cache[hash_]
        else:
            self.called_n += 1
            self.cache[hash_], self.meta[hash_] = self.__func(run_name=self.run_name, hash=hash_, dataset=self.dataset, save_path=self.save_path, **kwargs)
            self.cache_definitions[hash_] = kwargs
            if save is not None:
                self.save(self.save_path)
            return self.cache[hash_]

    @abc.abstractmethod
    def optimize(self, n) -> dict:
        pass

    def optimize_multiple(self, m, n):
        res = []
        for i in range(m):
            res.append(self.optimize(n))
        return res

    def hist_results(self, m, n, x_lim=4):
        best_fitnesses = []
        for res in self.optimize_multiple(m, n):
            best_fitnesses.append(min(x_lim, res['best_fitness']))
        plt.figure(figsize=(14, 2))
        plt.hist(best_fitnesses, bins=100)
        plt.xlabel('Best Fitness')
        plt.ylabel('Frequency')
        plt.title(f'Optimization Method: {self.__class__.__name__}')
        plt.xlim([0, x_lim])
        plt.show()
        print(f'Median: {np.median(best_fitnesses)}')
        print(f'Mean: {np.mean(best_fitnesses)}')
        print(f'Std: {np.std(best_fitnesses)}')

    def save(self, save_dir):
        with open(save_dir / f'save_{self.called_n}.pkl', 'wb') as f:
            pickle.dump([self.run_meta, self.cache, self.cache_definitions, self.meta, self.called_n], f)
        df = pd.DataFrame([{
            'run_name': self.run_name,
            'hash': k,
            **v,
            'score': self.cache[k],
            **{col_name: col_lambda(self.meta[k]) for col_name, col_lambda in self.csv_extract_columns.items()},
            'meta': self.meta[k],
        } for k, v in self.cache_definitions.items()])
        df.to_csv(save_dir / f'_df.csv', index=False)

    def load(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.run_meta, self.cache, self.cache_definitions, self.meta, self.called_n = pickle.load(f)

    def load_definitions(self, definitions: list[dict]):
        for definition in definitions:
            self.eval_func([definition[i] for i in self.search_space.column_names], save=True)


class RandomSearch(OptimizationMethod):
    def optimize(self, n) -> dict:
        best_individual = None
        best_fitness = np.inf
        for i in range(n):
            individual = self.search_space.sample()
            fitness = self.eval_func(individual)
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = individual
        return {'best_individual': best_individual, 'best_fitness': best_fitness}


class NelderMead(OptimizationMethod):
    def optimize(self, n) -> dict:
        best_individual = fmin(self.eval_func, self.search_space.sample(), disp=False, maxfun=n)
        best_fitness = self.eval_func(best_individual)
        return {'best_individual': best_individual, 'best_fitness': best_fitness}


class GeneticAlgorithm(OptimizationMethod):

    def __init__(self, func, search_space, run_name, save_path, dataset, csv_extract_columns=None, population_size=10, mutation_rate=0.02, last_rel_chance=.5, **kwargs):
        super().__init__(func,  search_space, run_name, save_path, dataset, csv_extract_columns, **kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.chances = np.linspace(1, last_rel_chance, population_size//2)
        self.chances = self.chances / np.sum(self.chances)
        self.run_meta['population'] = []

    def crossover(self, parent1, parent2):
        ids = np.random.randint(0, 2, size=len(parent1))
        child1 = parent1.copy()
        for i in range(len(parent1)):
            if ids[i] == 1:
                child1[i] = parent2[i]
        return self.search_space.sort_individual(child1)

    def mutate(self, individual):
        ids = np.random.rand(len(individual)) < self.mutation_rate
        random_sample = self.search_space.sample()
        for i in range(len(individual)):
            if ids[i]:
                individual[i] = random_sample[i]
        return self.search_space.sort_individual(individual)

    def optimize(self, n, load_n=None) -> dict:

        self.called_n = 0
        assert (n - self.population_size) % (self.population_size // 2) == 0
        if load_n is not None:
            raise NotImplementedError
            # self.load(BASE_DIR / 'GA' / self.run_name / self.dataset / f'save_{load_n}.pkl')
            # self.run_meta['population'] = [i for i in self.run_meta['population'] if self.search_space.is_valid(i)]
            # self.run_meta['population'] = self.run_meta['population'] + [self.search_space.sample() for _ in range(self.population_size - len(self.run_meta['population']))]
        else:
            self.run_meta['population'] = [self.search_space.sample() for _ in range(self.population_size)]

        fitnesses = np.array([self.eval_func(individual, save=True) for individual in tqdm(self.run_meta['population'])])
        ids = np.argsort(fitnesses)
        while self.called_n < n:
            self.run_meta['population'] = [self.run_meta['population'][i] for i in ids[:self.population_size//2]]
            for j in range(min(self.population_size//2, n - self.called_n)):
                parent1 = self.run_meta['population'][j]
                parent2 = self.run_meta['population'][np.random.choice(list(range(self.population_size//2)), p=self.chances)]
                child1 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                self.run_meta['population'].append(child1)

            fitnesses = np.array([self.eval_func(individual, save=True) for individual in tqdm(self.run_meta['population'])])
            ids = np.argsort(fitnesses)
        return {'best_individual': self.run_meta['population'][ids[0]], 'best_fitness': fitnesses[ids[0]]}

# Example Usage:
# Define your objective function and initial guess
# def objective_function(x):
#     return x ** 2  # Replace with your actual objective function
#
# initial_guess = 3  # Replace with your actual initial guess
#
# # Create an instance of the OptimizationMethod
# opt_method = OptimizationMethod(name="SimpleOptMethod")
#
# # Run multiple experiments
# opt_method.run_multiple_experiments(m=5, n=10, func=objective_function, initial_guess=initial_guess)
#
# # Get the results
# results = opt_method.get_results()
# print("Results:", results)


def quadratic_function(x):
    a, b, c, d = 1, 2, 3, 4  # Adjust coefficients as needed
    return a * x[0]**2 + b * x[1]**2 + c * x[2]**2 + d * x[3]**2


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley_function(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.25 * np.sum(xi**2 for xi in x)))
    term2 = -np.exp(0.25 * np.sum(np.cos(2 * np.pi * xi) for xi in x))
    return term1 + term2 + 20 + np.exp(1)


def rastrigin(x):
    a = 10
    return a * len(x) + np.sum(x**2 - a * np.cos(2 * np.pi * x))
