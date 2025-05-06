import random
import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class Item:
    def __init__(self, width: int, height: int, name: str = None):
        self.width = width
        self.height = height
        self.name = name
        self.area = width * height

    def rotate(self) -> None:
        self.width, self.height = self.height, self.width

    def __repr__(self) -> str:
        return f"{self.name or 'Item'}({self.width}x{self.height})"


class PackingSolution:
    def __init__(self, container_width: int, container_height: int, items: List[Item]):
        self.container_width = container_width
        self.container_height = container_height
        self.original_items = deepcopy(items)
        self.items = deepcopy(items)
        self.placement = []  # List of (x, y, rotated) tuples
        self.fitness = 0
        self.occupancy_grid = np.zeros((container_height, container_width), dtype=bool)

    def decode_chromosome(self, chromosome: List[float]) -> None:
        """Convert genetic representation to actual packing solution"""
        self.placement = []
        self.items = deepcopy(self.original_items)
        self.occupancy_grid.fill(False)

        # Chromosome structure:
        # [item1_order, item1_rot, item1_x, item1_y, item2_order, ...]
        gene_index = 0

        # Determine item order
        order_genes = chromosome[::4]
        ranked_indices = np.argsort(order_genes)

        # Process items in determined order
        for item_idx in ranked_indices:
            if gene_index + 3 >= len(chromosome):
                break

            item = self.items[item_idx]

            # Get rotation (0-1 value converted to boolean)
            rotation = chromosome[gene_index + 1] > 0.5
            if rotation:
                item.rotate()

            # Get position (scaled to container dimensions)
            x = int(chromosome[gene_index + 2] * (self.container_width - item.width))
            y = int(chromosome[gene_index + 3] * (self.container_height - item.height))

            # Ensure positions are within bounds
            x = max(0, min(x, self.container_width - item.width))
            y = max(0, min(y, self.container_height - item.height))

            # Try to place the item
            if self.can_place(item, x, y):
                self.place_item(item, x, y)
                self.placement.append((x, y, rotation))

            # Reset rotation for next evaluation
            if rotation:
                item.rotate()

            gene_index += 4

    def can_place(self, item: Item, x: int, y: int) -> bool:
        """Check if item fits at (x,y) without overlapping"""
        if x + item.width > self.container_width or y + item.height > self.container_height:
            return False

        return not np.any(self.occupancy_grid[y:y + item.height, x:x + item.width])

    def place_item(self, item: Item, x: int, y: int) -> None:
        """Place item at specified position"""
        self.occupancy_grid[y:y + item.height, x:x + item.width] = True

    def calculate_fitness(self) -> float:
        """Calculate fitness based on packed area and compactness"""
        packed_area = np.sum(self.occupancy_grid)
        total_area = self.container_width * self.container_height

        # Calculate center of mass
        y_indices, x_indices = np.where(self.occupancy_grid)
        if len(x_indices) == 0:
            return 0

        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)

        # Distance from center (we want items to be centered)
        center_distance = np.sqrt(
            (center_x - self.container_width / 2) ** 2 +
            (center_y - self.container_height / 2) ** 2
        )

        # Normalize center distance to [0, 1]
        max_distance = np.sqrt((self.container_width / 2) ** 2 + (self.container_height / 2) ** 2)
        normalized_distance = center_distance / max_distance if max_distance > 0 else 0

        # Fitness components
        area_ratio = packed_area / total_area
        compactness = 1 - normalized_distance

        # Weighted sum (adjust weights as needed)
        self.fitness = 0.8 * area_ratio + 0.2 * compactness
        return self.fitness

    def visualize(self) -> None:
        """Visualize the packed items"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw container
        container = patches.Rectangle(
            (0, 0), self.container_width, self.container_height,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(container)

        # Draw items
        color_map = plt.cm.get_cmap('tab10', len(self.items))

        for i, (item, (x, y, rotated)) in enumerate(zip(self.items, self.placement)):
            if rotated:
                item.rotate()

            rect = patches.Rectangle(
                (x, y), item.width, item.height,
                linewidth=1, edgecolor='black',
                facecolor=color_map(i), alpha=0.7,
                label=f"{item.name} ({'R' if rotated else 'N'})"
            )
            ax.add_patch(rect)

            plt.text(
                x + item.width / 2, y + item.height / 2,
                f"{item.name}\n{item.width}x{item.height}",
                ha='center', va='center', fontsize=8
            )

            if rotated:
                item.rotate()  # Rotate back

        plt.xlim(0, self.container_width)
        plt.ylim(0, self.container_height)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Packing Solution\nFitness: {self.fitness:.4f}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


class GeneticPackingOptimizer:
    def __init__(self, container_width: int, container_height: int, items: List[Item]):
        self.container_width = container_width
        self.container_height = container_height
        self.items = items
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.elitism_ratio = 0.1
        self.tournament_size = 3

    def initialize_population(self) -> List[List[float]]:
        """Create initial random population"""
        population = []
        num_genes = len(self.items) * 4  # order, rotation, x, y for each item

        for _ in range(self.population_size):
            chromosome = [random.random() for _ in range(num_genes)]
            population.append(chromosome)

        return population

    def evaluate_population(self, population: List[List[float]]) -> List[Tuple[float, List[float]]]:
        """Evaluate fitness of each chromosome"""
        evaluated = []

        for chromosome in population:
            solution = PackingSolution(self.container_width, self.container_height, self.items)
            solution.decode_chromosome(chromosome)
            fitness = solution.calculate_fitness()
            evaluated.append((fitness, chromosome))

        return sorted(evaluated, key=lambda x: x[0], reverse=True)

    def select_parents(self, evaluated_population: List[Tuple[float, List[float]]]) -> List[List[float]]:
        """Select parents using tournament selection"""
        parents = []

        # Keep top performers (elitism)
        elite_count = int(self.elitism_ratio * self.population_size)
        parents.extend([chromosome for (fitness, chromosome) in evaluated_population[:elite_count]])

        # Tournament selection for the rest
        while len(parents) < self.population_size:
            tournament = random.sample(evaluated_population, self.tournament_size)
            winner = max(tournament, key=lambda x: x[0])
            parents.append(winner[1])

        return parents

    def crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Perform crossover between two parents"""
        child = []
        crossover_point = random.randint(1, len(parent1) - 1)

        # Single-point crossover
        child.extend(parent1[:crossover_point])
        child.extend(parent2[crossover_point:])

        return child

    def mutate(self, chromosome: List[float]) -> List[float]:
        """Apply mutation to a chromosome"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                # Different mutation strategies for different gene types
                if i % 4 == 0:  # Order gene
                    chromosome[i] = random.random()
                elif i % 4 == 1:  # Rotation gene
                    chromosome[i] = 1 - chromosome[i]  # Flip rotation
                else:  # Position genes (x or y)
                    chromosome[i] = max(0, min(1, chromosome[i] + random.gauss(0, 0.1)))

        return chromosome

    def evolve(self) -> PackingSolution:
        """Run the genetic algorithm"""
        population = self.initialize_population()
        best_solution = None
        best_fitness = 0

        for generation in range(self.generations):
            evaluated = self.evaluate_population(population)
            current_best_fitness, current_best_chromosome = evaluated[0]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = PackingSolution(
                    self.container_width, self.container_height, self.items
                )
                best_solution.decode_chromosome(current_best_chromosome)
                print(f"Generation {generation}: New best fitness = {best_fitness:.4f}")

            parents = self.select_parents(evaluated)
            next_generation = []

            # Elitism - keep top performers
            next_generation.extend(parents[:int(self.elitism_ratio * self.population_size)])

            # Breed new offspring
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation

        return best_solution


# Example usage
if __name__ == "__main__":
    # Define container and items
    container_width = 20
    container_height = 15

    items = [
        Item(4, 6, "A"),
        Item(5, 3, "B"),
        Item(3, 4, "C"),
        Item(7, 2, "D"),
        Item(2, 5, "E"),
        Item(4, 4, "F"),
        Item(3, 3, "G"),
        Item(6, 2, "H"),
        Item(2, 7, "I"),
        Item(5, 2, "J"),
    ]

    # Run genetic optimization
    optimizer = GeneticPackingOptimizer(container_width, container_height, items)
    best_solution = optimizer.evolve()

    # Visualize the best solution
    if best_solution:
        print("\nBest solution found:")
        print(f"Fitness: {best_solution.fitness:.4f}")
        print(f"Packed area: {np.sum(best_solution.occupancy_grid)}/{container_width * container_height}")
        best_solution.visualize()
    else:
        print("No valid solution found")