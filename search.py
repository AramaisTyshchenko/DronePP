import heapq
import itertools
import random

from helpers import calculate_distance


def a_star(start_node, end_node, graph):
    """
    A* pathfinding algorithm with final improvements.
    """

    # Initialize g(n) and h(n)
    g = {node: float('inf') for node in graph.nodes}
    h = {node: calculate_distance(node, end_node) for node in graph.nodes}

    # The actual cost from start to the current node
    g[start_node] = 0

    # Nodes that have already been analyzed and have a path from the start to them
    closed_set = set()

    # Nodes that have been identified as a neighbor of an analyzed node, but have
    # yet to be fully analyzed
    open_set = {start_node}

    # Nodes' predecessors in the optimal path from start to goal
    predecessors = {node: None for node in graph.nodes}

    # Priority queue for nodes to explore next based on f(n) = g(n) + h(n)
    f_start = h[start_node]

    # Counter for the tiebreaker
    count = itertools.count()

    heap = [(f_start, h[start_node], next(count), start_node)]

    while open_set:
        # Get the node in open set with the lowest f(n)
        current = heapq.heappop(heap)[3]  # Extract node from the tuple
        open_set.remove(current)

        # Goal reached, reconstruct the path
        if current == end_node:
            path = []
            while current:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            return path

        closed_set.add(current)

        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue
            tentative_g = g[current] + graph[current][neighbor]['weight']

            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g >= g[neighbor]:
                continue

            predecessors[neighbor] = current
            g[neighbor] = tentative_g
            f_neighbor = g[neighbor] + h[neighbor]
            heapq.heappush(heap, (f_neighbor, h[neighbor], next(count), neighbor))

    # If the goal wasn't reached
    return None


def greedy_best_first_search(start_node, end_node, graph):
    """
    Greedy Best-First Search algorithm with final improvements.
    """

    # Initialize h(n)
    h = {node: calculate_distance(node, end_node) for node in graph.nodes}

    # Nodes that have already been analyzed and have a path from the start to them
    closed_set = set()

    # Nodes that have been identified as a neighbor of an analyzed node, but have
    # yet to be fully analyzed
    open_set = {start_node}

    # Nodes' predecessors in the optimal path from start to goal
    predecessors = {node: None for node in graph.nodes}

    # Counter for the tiebreaker
    count = itertools.count()

    # Priority queue for nodes to explore next based on h(n)
    heap = [(h[start_node], next(count), start_node)]

    while open_set:
        # Get the node in open set with the lowest h(n)
        current = heapq.heappop(heap)[2]  # Extract node from the tuple
        open_set.remove(current)

        # Goal reached, reconstruct the path
        if current == end_node:
            path = []
            while current:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            return path

        closed_set.add(current)

        for neighbor in graph[current]:
            if neighbor in closed_set:
                continue

            if neighbor not in open_set:
                open_set.add(neighbor)
                predecessors[neighbor] = current
                heapq.heappush(heap, (h[neighbor], next(count), neighbor))

    # If the goal wasn't reached
    return None


class GeneticAlgorithm:

    def __init__(self, graph, start_node, end_node, population_size=100, mutation_rate=0.01, generations=100):
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self):
        population = []

        for _ in range(self.population_size):
            path = [self.start_node]
            while path[-1] != self.end_node and len(path) < 10:  # Ensure a minimum path length
                next_node = random.choice(list(self.graph[path[-1]].keys()))
                if next_node not in path:  # Avoid loops in the path
                    path.append(next_node)
            if path[-1] != self.end_node:
                path.append(self.end_node)
            population.append(path)

        return population

    def fitness(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i + 1]]['weight']
        return -cost  # We want to maximize the fitness function, so we return negative cost

    def crossover(self, parent1, parent2):
        if len(parent1) < 3 or len(parent2) < 3:
            return parent1, parent2  # If parents are too short, skip crossover

        # One-point crossover
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 2)
        child1 = parent1[:crossover_point] + [node for node in parent2 if node not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [node for node in parent1 if node not in parent2[:crossover_point]]

        return child1, child2

    def mutate(self, path):
        # If the path is too short, skip mutation
        if len(path) < 3:
            return path

        # Randomly change a node in the path (except start and end nodes)
        if random.uniform(0, 1) < self.mutation_rate:
            idx = random.randint(1, len(path) - 2)
            previous_node = path[idx - 1]
            # Only choose neighbors of the previous node
            next_node = random.choice(list(self.graph[previous_node].keys()))
            path[idx] = next_node
        return path

    def run(self):
        population = self.initialize_population()

        for generation in range(self.generations):
            # Selection
            population = sorted(population, key=lambda x: self.fitness(x), reverse=True)
            next_generation = population[:2]  # Keep the 2 best paths

            while len(next_generation) < self.population_size:
                parent1, parent2 = random.choices(population[:10], k=2)  # Select 2 parents from top 10 paths
                child1, child2 = self.crossover(parent1, parent2)
                next_generation += [self.mutate(child1), self.mutate(child2)]

            population = next_generation

        # Return the best path after all generations
        best_path = max(population, key=lambda x: self.fitness(x))
        return best_path
