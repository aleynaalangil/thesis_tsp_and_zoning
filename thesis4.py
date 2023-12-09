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
            # G.add_edge((i, 15), (i, 16), weight=float('inf'))
            G.add_edge((i, 0), (i, 1), weight=float('inf'))

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

    # def plot_path_and_tree_for_others(self, orders, best_path):
    #     """Plot the spanning tree and the provided shortest path for given orders."""
    #     mst = nx.minimum_spanning_tree(self.graph)
    #     pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}  # Positions based on grid coordinates
    #
    #     plt.figure(figsize=(13, 13))
    #     nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)
    #
    #     # Define actual and dummy nodes based on the layout provided
    #     dummy_nodes = [(x, 0) for x in range(19)] + [(x, 16) for x in range(19)]
    #     for y in range(1, 16):
    #         dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])
    #
    #     actual_nodes = []
    #     for x in range(19):
    #         for y in range(1, 16):
    #             if (x, y) not in dummy_nodes:
    #                 actual_nodes.append((x, y))
    #
    #     # Draw nodes
    #     nx.draw_networkx_nodes(mst, pos, nodelist=actual_nodes, node_color='green', node_size=100)
    #     nx.draw_networkx_nodes(mst, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
    #     nx.draw_networkx_labels(mst, pos, {node: node for node in actual_nodes}, font_size=8)
    #
    #     # Draw the provided shortest path
    #     if best_path:
    #         path_edges = list(zip(best_path, best_path[1:]))
    #         print("Path Edges for Drawing:", path_edges)
    #         nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='red', width=2)
    #         nx.draw_networkx_nodes(self.graph, pos, nodelist=best_path, node_color='red', node_size=50)
    #     else:
    #         print("No shortest path found or path is empty.")
    #
    #     plt.legend(['Actual Nodes', 'Dummy Nodes', 'Shortest Path'])
    #     plt.title("Warehouse Path and Spanning Tree")
    #     plt.axis('off')
    #     plt.show()

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



layout = [['0' for _ in range(19)] for _ in range(17)]  # Initialize layout with '0's
for i in range(19):  # Adjust top and bottom rows for dummy nodes
    layout[0][i] = 'w'
    layout[16][i] = 'w'
for i in range(1, 16):  # Adjust columns for dummy nodes
    for j in [2, 5, 8, 11, 14, 17]:
        layout[i][j] = 'w'

orders = [(16, 15), (1, 5), (1, 8), (7, 13), (13, 9), (1, 13), (4, 11), (4, 5), (7, 4), (7, 7), (16, 4),
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
distance_matrix_df = spanning_tree_solver.generate_distance_matrix(orders)
print(distance_matrix_df)
spanning_tree_solver.plot_distance_matrix(distance_matrix_df)
