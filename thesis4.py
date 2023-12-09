import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import random
import math


class WarehouseSpanningTree:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout
        self.graph = self.create_graph_with_dummy_nodes()
        # self.validate_graph_connectivity(self.graph)

    def create_graph_with_dummy_nodes(self):
        G = nx.Graph()

        # Define dummy nodes at the top and bottom rows and at specific columns throughout the grid
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 16) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        # Define actual nodes, which are all other nodes not defined as dummy nodes
        actual_nodes = [(x, y) for x in range(19) for y in range(1, 15) if (x, y) not in dummy_nodes]

        # Add dummy nodes to the graph with a 'is_dummy' attribute set to True
        G.add_nodes_from(dummy_nodes, is_dummy=True)

        # Add actual nodes to the graph with a 'is_dummy' attribute set to False
        G.add_nodes_from(actual_nodes, is_dummy=False)

        # Connect dummy nodes horizontally and vertically with a weight of 1
        self._add_edges(G, dummy_nodes, weight=1)

        # Connect actual nodes to adjacent dummy nodes with a weight of 0
        self._add_edges(G, actual_nodes, weight=0, connect_to_dummy=True)

        # Define special edges with a weight of 2/5
        self._add_special_edges(G, range(0, 18), weight=2 / 5)

        # Define impassable edges with a weight of infinity to block certain paths
        self._add_impassable_edges(G)

        critical_coords = [
            ((5, 14), (5, 15)),
            ((8, 14), (8, 15)),
            ((11, 14), (11, 15)),
            ((14, 14), (14, 15)),
            ((17, 14), (17, 15))
        ]

        for edge in critical_coords:
            G.add_edge(*edge, weight=1)

        return G

    def _add_edges(self, G, nodes, weight=1, connect_to_dummy=False):
        for node in nodes:
            x, y = node
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for nx, ny in neighbors:
                if connect_to_dummy:
                    if (nx, ny) in G and G.nodes[(nx, ny)]['is_dummy']:
                        G.add_edge(node, (nx, ny), weight=weight)
                else:
                    if (nx, ny) in G:
                        G.add_edge(node, (nx, ny), weight=weight)

    def _add_special_edges(self, G, range_values, weight):
        for i in range_values:
            if (i, 0) in G:
                G.add_edge((i, 0), (i + 1, 0), weight=weight)
            if (i, 16) in G:
                G.add_edge((i, 16), (i + 1, 16), weight=weight)

    def _add_impassable_edges(self, G):
        for i in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18]:
            G.add_edge((i, 15), (i, 16), weight=float('inf'))
            G.add_edge((i, 0), (i, 1), weight=float('inf'))

    # def validate_graph_connectivity(self, G):
    #     if not nx.is_connected(G):
    #         raise ValueError("The graph must be connected for MST and TSP calculations.")
    def find_shortest_path(self, orders):
        """Find the shortest path for TSP, considering each order as a possible starting point."""
        if not orders:
            print("No orders provided.")
            return [], 0

        best_path = []
        best_length = float('inf')

        critical_coords = [
            ((5, 14), (5, 15)),
            ((8, 14), (8, 15)),
            ((11, 14), (11, 15)),
            ((14, 14), (14, 15)),
            ((17, 14), (17, 15))
        ]

        for start_node in orders:
            # Create a cycle with the start node as the first and last element
            tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]

            # Find the TSP path
            tsp_path = nx.approximation.traveling_salesman_problem(self.graph, nodes=tsp_orders, cycle=True)

            # Calculate the total length of the path
            total_length = sum(
                self.graph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))

            for i in range(len(tsp_path) - 1):
                # Check both directions since the path can traverse in either
                if (tsp_path[i], tsp_path[i + 1]) in critical_coords or (
                        tsp_path[i + 1], tsp_path[i]) in critical_coords:
                    total_length += 1
                    # print(f"Added weight for critical coordinate: {tsp_path[i]} -> {tsp_path[i + 1]}")

                    # Check if this is the shortest path so far
            if total_length < best_length:
                best_length = total_length
                best_path = tsp_path

            # Filter out dummy nodes from the best path
        collected_orders_path = [node for node in best_path if self.graph.nodes[node]['is_dummy'] == False]
        print("Collected Orders Path:", collected_orders_path)
        print("Best Starting Point TSP Path:", best_path, "\nLength of the Path:", best_length -1)
        return best_path

    def plot_spanning_tree(self):
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 16) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 16):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

        # Draw nodes
        nx.draw_networkx_nodes(mst, pos, nodelist=actual_nodes, node_color='green', node_size=100)
        nx.draw_networkx_nodes(mst, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
        nx.draw_networkx_labels(mst, pos, {node: node for node in actual_nodes}, font_size=8)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(mst, 'weight')
        nx.draw_networkx_edge_labels(mst, pos, edge_labels=edge_labels)

        plt.legend(['Actual Nodes', 'Dummy Nodes', 'Edge Weights'])
        plt.title("Warehouse Minimum Spanning Tree with Edge Weights")
        plt.axis('off')
        plt.show()

    def plot_path_and_tree_for_tsp(self, orders):
        """Plot the spanning tree and the shortest path for given orders."""
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}  # Positions based on grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)


        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 16) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 16):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

        # Draw nodes
        nx.draw_networkx_nodes(mst, pos, nodelist=actual_nodes, node_color='green', node_size=100)
        nx.draw_networkx_nodes(mst, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
        nx.draw_networkx_labels(mst, pos, {node: node for node in actual_nodes}, font_size=8)

        # Find and draw the shortest path
        shortest_path = self.find_shortest_path(orders)
        if shortest_path:  # Check if the shortest path is not empty
            path_edges = list(zip(shortest_path, shortest_path[1:]))
            print("Path Edges for Drawing:", path_edges)  # Added for debugging
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='red', width=2)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=shortest_path, node_color='red', node_size=50)
        else:
            print("No shortest path found or path is empty.")

        plt.legend(['Actual Nodes', 'Dummy Nodes', 'Shortest Path'])
        plt.title("Warehouse Path and Spanning Tree")
        plt.axis('off')
        plt.show()

    def plot_path_and_tree_for_others(self, orders, best_path):
        """Plot the spanning tree and the provided shortest path for given orders."""
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}  # Positions based on grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 16) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 16):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

        # Draw nodes
        nx.draw_networkx_nodes(mst, pos, nodelist=actual_nodes, node_color='green', node_size=100)
        nx.draw_networkx_nodes(mst, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
        nx.draw_networkx_labels(mst, pos, {node: node for node in actual_nodes}, font_size=8)

        # Draw the provided shortest path
        if best_path:
            path_edges = list(zip(best_path, best_path[1:]))
            print("Path Edges for Drawing:", path_edges)
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='red', width=2)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=best_path, node_color='red', node_size=50)
        else:
            print("No shortest path found or path is empty.")

        plt.legend(['Actual Nodes', 'Dummy Nodes', 'Shortest Path'])
        plt.title("Warehouse Path and Spanning Tree")
        plt.axis('off')
        plt.show()

    def generate_distance_matrix(self, orders):
        """Generate a distance matrix for the given orders."""
        distance_matrix = {}

        for i, from_node in enumerate(orders):
            distances = []
            for to_node in orders:
                if from_node == to_node:
                    distance = 0
                else:
                    distance = nx.shortest_path_length(self.graph, from_node, to_node, weight='weight')
                distances.append(distance)
            distance_matrix[f"Order {i + 1} ({from_node})"] = distances

        distance_matrix_df = pd.DataFrame(distance_matrix,
                                          index=[f"Order {i + 1} ({node})" for i, node in enumerate(orders)])
        return distance_matrix_df

    def plot_distance_matrix(self, distance_matrix):
        """Plot the distance matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, cmap="YlGnBu", fmt=".2f")  # Using .2f for floating-point numbers
        plt.title("Distance Matrix between Orders")
        plt.show()

    # Genetic Algorithm
    def genetic_algorithm(self, orders, population_size=100, generations=100):
        # Initialize population
        population = self._initialize_population(orders, population_size)
        print(f"Initial Population Size: {len(population)}")  # Debugging output

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual) for individual in population]

            # Selection
            selected = self._select(population, fitness_scores)

            # Crossover
            crossovered = self._crossover(selected)

            # Mutation
            mutated = self._mutate(crossovered)

            # Update population
            population = mutated

            # Debugging output
            print(f"Generation {generation}: Population Size: {len(population)}")

        # Find best solution
        if population:
            best_individual = min(population, key=self._evaluate_fitness)
            return best_individual
        else:
            raise ValueError("Population is empty. Check the genetic algorithm implementation.")

    def _initialize_population(self, orders, population_size):
        # Initialize a population with random paths
        return [random.sample(orders, len(orders)) for _ in range(population_size)]

    def _select(self, population, fitness_scores, tournament_size=5):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def _crossover(self, selected, crossover_rate=0.9):
        crossovered = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            if random.random() < crossover_rate:
                cut = random.randint(1, len(parent1) - 1)
                child1 = parent1[:cut] + parent2[cut:]
                child2 = parent2[:cut] + parent1[cut:]
                crossovered.extend([child1, child2])
            else:
                crossovered.extend([parent1, parent2])
        return crossovered

    def _mutate(self, crossovered, mutation_rate=0.02):
        mutated = []
        for individual in crossovered:
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
            mutated.append(individual)
        return mutated

    def _evaluate_fitness(self, individual):
        total_length = 0
        critical_coords = [
            ((5, 14), (5, 15)),
            ((8, 14), (8, 15)),
            ((11, 14), (11, 15)),
            ((14, 14), (14, 15)),
            ((17, 14), (17, 15))
        ]
        for i in range(len(individual) - 1):
            try:
                edge_weight = self.graph.edges[individual[i], individual[i + 1]]['weight']
            except KeyError:
                edge_weight = float('inf')  # Assign a very high cost to invalid paths
            total_length += edge_weight
            # Add extra weight for critical coordinates
            if (individual[i], individual[i + 1]) in critical_coords:
                total_length += 1
        return total_length

    # Simulated Annealing
    def simulated_annealing(self, initial_state, temperature=10000, cooling_rate=0.003):
        current_state = initial_state
        current_cost = self._calculate_cost(current_state)

        while temperature > 1:
            new_state = self._get_neighbor(current_state)
            new_cost = self._calculate_cost(new_state)

            if self._acceptance_probability(current_cost, new_cost, temperature) > random.random():
                current_state = new_state
                current_cost = new_cost

            temperature *= 1 - cooling_rate

        return current_state

    def _calculate_cost(self, state):
        # Calculate the total cost of the path, including extra weight for critical coordinates
        critical_coords = [
            ((5, 14), (5, 15)),
            ((8, 14), (8, 15)),
            ((11, 14), (11, 15)),
            ((14, 14), (14, 15)),
            ((17, 14), (17, 15))
        ]
        total_cost = 0
        for i in range(len(state) - 1):
            total_cost += self.graph.edges[state[i], state[i + 1]]['weight']
            if (state[i], state[i + 1]) in critical_coords:
                total_cost += 1
        return total_cost

    def _get_neighbor(self, state):
        neighbor = state[:]
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    def _acceptance_probability(self, old_cost, new_cost, temperature):
        # Calculate the acceptance probability
        if new_cost < old_cost:
            return 1
        return math.exp((old_cost - new_cost) / temperature)

    # Ant Colony Optimization
    def ant_colony_optimization(self, orders, ant_count=100, iterations=100):
        pheromones = self._initialize_pheromones()
        best_path = None
        best_length = float('inf')

        for _ in range(iterations):
            paths = [self._move_ant(pheromones, orders) for _ in range(ant_count)]
            self._update_pheromones(pheromones, paths)
            for path in paths:
                length = self._path_cost(path)
                if length < best_length:
                    best_path, best_length = path, length

        return best_path
        # # Initialize pheromones
        # pheromones = self._initialize_pheromones()
        #
        # for iteration in range(iterations):
        #     # Move ants
        #     paths = [self._move_ant(pheromones, orders) for _ in range(ant_count)]
        #
        #     # Update pheromones
        #     self._update_pheromones(pheromones, paths)
        #
        # # Find the best path
        # best_path = min(paths, key=lambda path: self._path_cost(path))
        # return best_path

    def _initialize_pheromones(self):
        pheromones = {}
        for i in range(19):
            for j in range(1, 16):
                if i != j:
                    pheromones[(i, j)] = 1.0  # Initial pheromone level
        return pheromones

    def _move_ant(self, pheromones, orders):
        path = [random.choice(orders)]
        while len(path) < len(orders):
            next_order = self._choose_next_order(path[-1], orders, pheromones)
            path.append(next_order)
        return path

    def _choose_next_order(self, current_order, orders, pheromones):
        # Implement logic to choose the next order based on pheromones
        pass

    def _update_pheromones(self, pheromones, paths):
        # Reduce all pheromone levels (simulating evaporation)
        for key in pheromones.keys():
            pheromones[key] *= 0.95

        # Increase pheromones on paths taken by ants
        for path in paths:
            contribution = 1.0 / self._path_cost(path)
            for i in range(len(path) - 1):
                pheromones[(path[i], path[i + 1])] += contribution

    def _path_cost(self, path):
        # Calculate the total cost of the path, including extra weight for critical coordinates
        critical_coords = [
            ((5, 14), (5, 15)),
            ((8, 14), (8, 15)),
            ((11, 14), (11, 15)),
            ((14, 14), (14, 15)),
            ((17, 14), (17, 15))
        ]
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += self.graph.edges[path[i], path[i + 1]]['weight']
            if (path[i], path[i + 1]) in critical_coords:
                total_cost += 1
        return total_cost


layout = [['0' for _ in range(19)] for _ in range(17)]  # Initialize layout with '0's
for i in range(19):  # Adjust top and bottom rows for dummy nodes
    layout[0][i] = 'w'
    layout[16][i] = 'w'
for i in range(1, 16):  # Adjust columns for dummy nodes
    for j in [2, 5, 8, 11, 14, 17]:
        layout[i][j] = 'w'

orders = [(16, 14), (1, 5), (1, 8), (7, 13), (13, 9), (1, 13), (4, 11), (4, 5), (7, 4), (7, 7), (16, 4),
          (16, 7)]  # Correct format for orders

spanning_tree_solver = WarehouseSpanningTree(layout)

print("Running TSP Algorithm...")
spanning_tree_solver.plot_spanning_tree()
spanning_tree_solver.plot_path_and_tree_for_tsp(orders)
# # Call Genetic Algorithm
# print("Running Genetic Algorithm...")
# best_path_ga = spanning_tree_solver.genetic_algorithm(orders)
# spanning_tree_solver.plot_path_and_tree_for_others(orders, best_path_ga)
# print("Best path found by GA:", best_path_ga)
# #
# # Call Simulated Annealing
# print("Running Simulated Annealing...")
# initial_state_sa = random.sample(orders, len(orders))  # Random initial state
# best_path_sa = spanning_tree_solver.simulated_annealing(initial_state_sa)
# spanning_tree_solver.plot_path_and_tree_for_others(orders, best_path_sa)
# print("Best path found by SA:", best_path_sa)
#
# # Call Ant Colony Optimization
# print("Running Ant Colony Optimization...")
# best_path_aco = spanning_tree_solver.ant_colony_optimization(orders)
# spanning_tree_solver.plot_path_and_tree_for_others(orders, best_path_aco)
# print("Best path found by ACO:", best_path_aco)
#
# spanning_tree_solver = WarehouseSpanningTree(layout)
# spanning_tree_solver.plot_spanning_tree()
# spanning_tree_solver.plot_path_and_tree(orders)
# distance_matrix_df = spanning_tree_solver.generate_distance_matrix(orders)
# print(distance_matrix_df)
# spanning_tree_solver.plot_distance_matrix(distance_matrix_df)
