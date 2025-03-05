from .base_deployment import SensorDeployment
from typing import List, Tuple
import numpy as np
import random
import math

class GeneticDeployment(SensorDeployment):
    """
    Genetic algorithm for sensor deployment.
    
    This algorithm uses a genetic algorithm to optimize sensor positions on a grid.
    The fitness function is computed as a weighted sum of the burn probability and fire
    intensity values at the sensor positions, minus a penalty for any pair of sensors that
    are closer than the sensor_radius.
    """
    def __init__(self, 
                 num_sensors: int, 
                 sensor_radius: float, 
                 population_size: int = 50, 
                 num_generations: int = 100,
                 mutation_rate: float = 0.1, 
                 alpha: float = 0.5, 
                 beta: float = 0.5):
        """
        :param num_sensors: Number of sensors to deploy.
        :param sensor_radius: Minimum separation radius between sensors.
        :param population_size: Number of candidate solutions per generation.
        :param num_generations: Number of generations to evolve.
        :param mutation_rate: Probability of mutation per sensor in a candidate.
        :param alpha: Weight for the burn probability in fitness evaluation.
        :param beta: Weight for the fire intensity in fitness evaluation.
        """
        self.num_sensors = num_sensors
        self.sensor_radius = sensor_radius
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta

    def place_sensors(self, fire_map: np.ndarray, bp_map: np.ndarray, time_step: int, prev_positions=None) -> List[Tuple[int, int]]:
        rows, cols = bp_map.shape

        def random_candidate():
            # Generate a candidate solution: a list of sensor positions (row, col)
            candidate = []
            for _ in range(self.num_sensors):
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                candidate.append((r, c))
            return candidate

        def fitness(candidate):
            # Compute fitness as the weighted sum of sensor values minus penalties for closeness.
            value = 0.0
            penalty = 0.0
            for (r, c) in candidate:
                # Higher value in areas with higher burn probability and fire intensity.
                value += self.alpha * bp_map[r, c] + self.beta * fire_map[r, c]
            # Penalize pairs of sensors that are too close.
            for i in range(len(candidate)):
                for j in range(i+1, len(candidate)):
                    r1, c1 = candidate[i]
                    r2, c2 = candidate[j]
                    dist = math.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                    if dist < self.sensor_radius:
                        penalty += (self.sensor_radius - dist)
            return value - penalty

        def crossover(parent1, parent2):
            # Single-point crossover between two parent candidates.
            point = random.randint(1, self.num_sensors - 1)
            child = parent1[:point] + parent2[point:]
            # Remove duplicates: if a position appears twice, replace it with a random position.
            unique_child = []
            for pos in child:
                if pos not in unique_child:
                    unique_child.append(pos)
                else:
                    r = random.randint(0, rows - 1)
                    c = random.randint(0, cols - 1)
                    unique_child.append((r, c))
            # Ensure the candidate has exactly num_sensors positions.
            while len(unique_child) < self.num_sensors:
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                if (r, c) not in unique_child:
                    unique_child.append((r, c))
            return unique_child[:self.num_sensors]

        def mutate(candidate):
            # Mutate candidate by randomly changing sensor positions with probability mutation_rate.
            new_candidate = candidate.copy()
            for i in range(self.num_sensors):
                if random.random() < self.mutation_rate:
                    new_candidate[i] = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            return new_candidate

        # Initialize population with random candidates.
        population = [random_candidate() for _ in range(self.population_size)]
        best_candidate = None
        best_fitness = -float('inf')

        # Evolve population over a number of generations.
        for _ in range(self.num_generations):
            # Evaluate fitness for each candidate.
            population_fitness = [fitness(candidate) for candidate in population]

            # Update the best candidate found so far.
            for candidate, fit in zip(population, population_fitness):
                if fit > best_fitness:
                    best_fitness = fit
                    best_candidate = candidate

            # Selection: tournament selection.
            new_population = []
            while len(new_population) < self.population_size:
                tournament = random.sample(population, 3)
                tournament_fitness = [fitness(candidate) for candidate in tournament]
                winner = tournament[tournament_fitness.index(max(tournament_fitness))]
                new_population.append(winner)
            # Crossover and mutation to form the next generation.
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = new_population[i]
                parent2 = new_population[(i+1) % self.population_size]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                child1 = mutate(child1)
                child2 = mutate(child2)
                next_population.extend([child1, child2])
            population = next_population[:self.population_size]

        return best_candidate
