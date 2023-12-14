import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from gurobipy import *


class WarehouseSpanningTree:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout
        self.graph = self._create_graph_with_dummy_nodes()
        self.path_results = {}  # To store path results
        pass

    def _create_graph_with_dummy_nodes(self):
        G = nx.Graph()

        # Add nodes to the graphG.add_edge((5,14),(5,15),weight=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 17) for x in range(19)]
        for y in range(1, 17):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 17):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))
        G.add_edge((8, 14), (8, 15), weight=1)
        G.add_edge((11, 14), (11, 15), weight=1)
        G.add_edge((14, 14), (14, 15), weight=1)
        G.add_edge((17, 14), (17, 15), weight=1)
        # Add nodes to the graph
        for node in dummy_nodes:
            G.add_node(node, is_dummy=True)
        for node in actual_nodes:
            G.add_node(node, is_dummy=False)

        # Add edges between dummy nodes
        for node in dummy_nodes:
            x, y = node
            if (x + 1, y) in dummy_nodes:
                G.add_edge(node, (x + 1, y), weight=1)
            if (x - 1, y) in dummy_nodes:
                G.add_edge(node, (x - 1, y), weight=1)
            if (x, y + 1) in dummy_nodes:
                G.add_edge(node, (x, y + 1), weight=1)
            if (x, y - 1) in dummy_nodes:
                G.add_edge(node, (x, y - 1), weight=1)

        # Add edges between actual nodes and adjacent dummy nodes
        for node in actual_nodes:
            x, y = node
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (x + dx, y + dy) in dummy_nodes:
                    G.add_edge(node, (x + dx, y + dy), weight=0)

        # Adding specific edges with different weights
        for i in range(0, 18):
            if (i, 0) in dummy_nodes:  # Check if the edge endpoint is in dummy_nodes
                G.add_edge((i, 0), (i + 1, 0), weight=2 / 3)
            if (i, 17) in dummy_nodes:  # Check if the edge endpoint is in dummy_nodes
                G.add_edge((i, 17), (i + 1, 17), weight=2 / 3)
        for i in range(0, 16, 3):
            if (0, i) in dummy_nodes:
                G.add_edge((0, i), (1, i), weight=float('inf'))

        G.add_edge((2, 0), (2, 1), weight=0)
        G.add_edge((5, 0), (5, 1), weight=0)
        G.add_edge((8, 0), (8, 1), weight=0)
        G.add_edge((11, 0), (11, 1), weight=0)
        G.add_edge((14, 0), (14, 1), weight=0)
        G.add_edge((17, 0), (17, 1), weight=0)

        G.add_edge((2, 16), (2, 17), weight=0)
        G.add_edge((5, 16), (5, 17), weight=0)
        G.add_edge((8, 16), (8, 17), weight=0)
        G.add_edge((11, 16), (11, 17), weight=0)
        G.add_edge((14, 16), (14, 17), weight=0)
        G.add_edge((17, 16), (17, 17), weight=0)

        G.add_edge((0, 16), (0, 17), weight=float('inf'))
        G.add_edge((1, 16), (1, 17), weight=float('inf'))
        G.add_edge((3, 16), (3, 17), weight=float('inf'))
        G.add_edge((4, 16), (4, 17), weight=float('inf'))
        G.add_edge((6, 16), (6, 17), weight=float('inf'))
        G.add_edge((7, 16), (7, 17), weight=float('inf'))
        G.add_edge((9, 16), (9, 17), weight=float('inf'))
        G.add_edge((10, 16), (10, 17), weight=float('inf'))
        G.add_edge((12, 16), (12, 17), weight=float('inf'))
        G.add_edge((13, 16), (13, 17), weight=float('inf'))
        G.add_edge((15, 16), (15, 17), weight=float('inf'))
        G.add_edge((16, 16), (16, 17), weight=float('inf'))
        G.add_edge((18, 16), (18, 17), weight=float('inf'))
        G.add_edge((0, 0), (0, 1), weight=float('inf'))
        G.add_edge((1, 0), (1, 1), weight=float('inf'))
        G.add_edge((3, 0), (3, 1), weight=float('inf'))
        G.add_edge((4, 0), (4, 1), weight=float('inf'))
        G.add_edge((6, 0), (6, 1), weight=float('inf'))
        G.add_edge((7, 0), (7, 1), weight=float('inf'))
        G.add_edge((9, 0), (9, 1), weight=float('inf'))
        G.add_edge((10, 0), (10, 1), weight=float('inf'))
        G.add_edge((12, 0), (12, 1), weight=float('inf'))
        G.add_edge((13, 0), (13, 1), weight=float('inf'))
        G.add_edge((15, 0), (15, 1), weight=float('inf'))
        G.add_edge((16, 0), (16, 1), weight=float('inf'))
        G.add_edge((18, 0), (18, 1), weight=float('inf'))

        non_aisles = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18]
        for i in range(0, 17, 3):
            for j in range(0, 17):
                G.add_edge((i, j), (i+1, j), weight=float('inf'))

        for i in range(len(non_aisles)):
            for j in range(0, 17):
                G.add_edge((non_aisles[i], j), (non_aisles[i], j+1), weight=float('inf'))

        return G


    # def find_shortest_path(self, orders):
    # #     """Find the shortest path for TSP, considering each order as a possible starting point."""
    # #     if not orders:
    # #         print("No orders provided.")
    # #         return [], 0
    # #
    # #     best_path = []
    # #     best_length = float('inf')
    # #
    # #     for start_node in orders:
    # #         # Create a cycle with the start node as the first and last element
    # #         tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]
    # #
    # #         # Find the TSP path
    # #         tsp_path = nx.approximation.traveling_salesman_problem(self.graph, nodes=tsp_orders, cycle=True)
    # #
    # #         # Calculate the total length of the path
    # #         total_length = sum(
    # #             self.graph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    # #
    # #         # Check if this is the shortest path so far
    # #         if total_length < best_length:
    # #             best_length = total_length
    # #             best_path = tsp_path
    # #
    # #         # Filter out dummy nodes from the best path
    # #     collected_orders_path = [node for node in best_path if self.graph.nodes[node]['is_dummy'] == False]
    # #     print("Collected Orders Path:", collected_orders_path)
    # #     print("Best Starting Point TSP Path:", best_path, "\nLength of the Path:", best_length)
    # #     orders_tuple = tuple(orders)  # Convert list to tuple for dictionary key
    # #     self.calculated_paths[orders_tuple] = (best_path, best_length)
    # #
    # #     return best_path
    #     orders_tuple = tuple(orders)
    #     if orders_tuple in self.path_results:
    #         return self.path_results[orders_tuple]
    #
    #     if not orders:
    #         print("No orders provided.")
    #         return [], 0  # Return an empty path and length of 0 if no orders are provided
    #
    #     if len(orders) == 0:
    #         return [], 0  # Handle the case of an empty subgraph
    #
    #         # Create a subgraph of the main graph containing only the selected orders
    #     subgraph = self.graph.subgraph(orders)
    #
    #     # Check if the subgraph is empty
    #     if len(subgraph.nodes) == 0:
    #         return [], 0
    #
    #     # Check if the subgraph is connected
    #     if not nx.is_connected(subgraph):
    #         return [], float('inf')  # Return an empty path and infinity length for disconnected subgraphs
    #
    #     best_length = float('inf')
    #     best_path = []
    #
    #     for start_node in orders:
    #         tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]
    #
    #         # Find the TSP path
    #         tsp_path = nx.approximation.traveling_salesman_problem(
    #             subgraph, nodes=tsp_orders, cycle=True, method=nx.approximation.christofides
    #         )
    #
    #         # Calculate the total length of the path
    #         total_length = sum(subgraph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    #
    #         # Check if this is the shortest path so far
    #         if total_length < best_length:
    #             best_length = total_length
    #             best_path = tsp_path
    #
    #     # Filter out dummy nodes from the best path
    #     collected_orders_path = [node for node in best_path if not self.graph.nodes[node]['is_dummy']]
    #     print("Collected Orders Path:", collected_orders_path)
    #     print("Best Starting Point TSP Path:", best_path, "\nLength of the Path:", best_length)
    #     self.path_results[orders_tuple] = (best_path, best_length)
    #     return best_path, best_length

    # def find_shortest_path(self, orders):
    #     orders_tuple = tuple(orders)
    #     if orders_tuple in self.path_results:
    #         return self.path_results[orders_tuple]
    #
    #     if not orders:
    #         return [], 0  # No orders provided
    #
    #     # Create a subgraph with the given orders
    #     subgraph = self.graph.subgraph(orders)
    #
    #     # Check if the subgraph is empty or not connected
    #     if len(subgraph.nodes) == 0 or not nx.is_connected(subgraph):
    #         return [], float('inf')  # Return empty path and infinite length
    #
    #     best_path = []
    #     best_length = float('inf')
    #
    #     for start_node in orders:
    #         # Create a cycle with the start node as the first and last element
    #         tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]
    #
    #         # Find the TSP path
    #         tsp_path = nx.approximation.traveling_salesman_problem(subgraph, nodes=tsp_orders, cycle=True)
    #
    #         # Calculate the total length of the path
    #         total_length = sum(subgraph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    #
    #         # Update best path if a shorter one is found
    #         if total_length < best_length:
    #             best_length = total_length
    #             best_path = tsp_path
    #
    #     # Store and return the best path found
    #     self.path_results[orders_tuple] = (best_path, best_length)
    #     return best_path, best_length

   # def find_shortest_path(self, orders):
   #     """Find the shortest path for TSP, considering each order as a possible starting point."""

    #
    #     # Find the TSP path
    #     tsp_path = nx.approximation.traveling_salesman_problem(self.graph, nodes=tsp_orders, cycle=True)
    #
    #     # Calculate the total length of the path
    #     total_length = sum(
    #         self.graph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))
    #
    #     # Check if this is the shortest path so far
    #     if total_length < best_length:
    #         best_length = total_length
    #         best_path = tsp_path
    #
    # # Filter out dummy nodes from the best path
    # collected_orders_path = [node for node in best_path if self.graph.nodes[node]['is_dummy'] == False]
    # print("Collected Orders Path:", collected_orders_path)
    # print("Best Starting Point TSP Path:", best_path, "\nLength of the Path:", best_length)
    # self.path_results[orders_tuple] = (best_path, best_length)
    # return best_path, best_length
    def find_shortest_path(self, orders):
        """Find the shortest path for TSP, considering each order as a possible starting point."""

        orders_tuple = tuple(orders)
        if orders_tuple in self.path_results:
            return self.path_results[orders_tuple]

        if not orders:
            print("No orders provided.")
            return [], 0  # Return an empty path and length of 0 if no orders are provided

        best_path = []
        best_length = float('inf')

        for start_node in orders:
            # Create a cycle with the start node as the first and last element
            tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]

            # Find the TSP path
            tsp_path = nx.approximation.traveling_salesman_problem(self.graph, nodes=tsp_orders, cycle=True,
                                                                   method=nx.approximation.christofides)

            # Calculate the total length of the path
            total_length = sum(
                self.graph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))

            # Check if this is the shortest path so far
            if total_length < best_length:
                best_length = total_length
                best_path = tsp_path

            # Filter out dummy nodes from the best path
            collected_orders_path = [node for node in best_path if not self.graph.nodes[node]['is_dummy']]
            print("Collected Orders Path:", collected_orders_path)
            print("Best Starting Point TSP Path:", best_path, "\nLength of the Path:", best_length)

        self.path_results[orders_tuple] = (best_path, best_length)
        return best_path, best_length

    def plot_spanning_tree(self):
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 17) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 17):
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

    def plot_path_and_tree(self, orders):
        """Plot the complete graph and the shortest path for given orders."""
        pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}  # Positions based on grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 17) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 17):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=actual_nodes, node_color='green', node_size=100)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
        nx.draw_networkx_labels(self.graph, pos, {node: node for node in actual_nodes}, font_size=8)

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
        plt.title("Warehouse Complete Graph and Shortest Path")
        plt.axis('off')
        plt.show()

    def plot_spanning_tree(self):
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 17) for x in range(19)]
        for y in range(1, 17):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 17):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, nodelist=actual_nodes, node_color='green', node_size=100)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
        nx.draw_networkx_labels(self.graph, pos, {node: node for node in actual_nodes}, font_size=8)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.legend(['Actual Nodes', 'Dummy Nodes', 'Edge Weights'])
        plt.title("Warehouse Complete Graph with Edge Weights")
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
                    # distance = gurobi.(self.graph, from_node, to_node, weight='weight')
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

    def _divide_orders_by_aisles(self, orders, start_aisle, end_aisle):
        """Divide orders into sub-orders between specified aisles."""
        # aisle_orders = {}
        # aisles = [2, 5, 8, 11, 14, 17]  # Your aisle coordinates
        #
        # for order in orders:
        #     for aisle in aisles:
        #         if order[0] == aisle:  # Assuming the x-coordinate determines the aisle
        #             if aisle not in aisle_orders:
        #                 aisle_orders[aisle] = []
        #             aisle_orders[aisle].append(order)
        #             break
        #
        # return aisle_orders
        """Divide orders into sub-orders between specified aisles."""
        # Aisles are given by their x-coordinates
        aisles = [2, 5, 8, 11, 14, 17]
        start_aisle_index = aisles.index(start_aisle)
        end_aisle_index = aisles.index(end_aisle)
        sub_orders = [order for order in orders if aisles[start_aisle_index] <= order[0] <= aisles[end_aisle_index]]
        return sub_orders

    def dynamic_zone_picking(self, orders, num_pickers):
        M = 6  # Max aisle number
        F = {}  # Dictionary to store the best paths for different aisle-picker combinations

        # Precompute shortest paths for each aisle combination
        RR = {}
        aisles = [2, 5, 8, 11, 14, 17]  # Aisle coordinates
        # for i in range(M):
        #     for j in range(i, M):
        #         sub_orders = self._divide_orders_by_aisles(orders, aisles[i], aisles[j])
        #         _, path_length = self.find_shortest_path(sub_orders)  # Extracting only the path length
        #         RR[(aisles[i], aisles[j])] = path_length

        for i in range(M):
            for j in range(i, M):
                sub_orders = self._divide_orders_by_aisles(orders, aisles[i], aisles[j])
                if sub_orders:  # Check if there are orders in the sub-list
                    _, path_length = self.find_shortest_path(sub_orders)
                    RR[(aisles[i], aisles[j])] = path_length
                else:
                    RR[(aisles[i], aisles[j])] = 0  # No path needed for empty sub-orders

        # Initialize F for one picker
        for i in range(M):
            F[(aisles[i], 1)] = RR[(2, aisles[i])]

        # Compute F for more than one picker
        for i in range(1, M):
            for k in range(2, min(i + 1, num_pickers) + 1):
                temp = float('inf')
                for j in range(k - 1, i):
                    if j + 1 <= i:
                        temp = min(temp, max(F.get((aisles[j], k - 1), float('inf')),
                                             RR.get((aisles[j + 1], aisles[i]), float('inf'))))
                F[(aisles[i], k)] = temp

        return F

# def dynamic_zone_picking(self, max_aisle, pickers, orders):
    #     M = max_aisle  # Maximum aisle number
    #     F = {}  # F(i, k) matrix
    #
    #     # Initialize F matrix
    #     for i in range(1, M + 1):
    #         path, length = self.find_shortest_path(self._divide_orders_by_aisles(orders, 1, i))
    #         F[(i, 1)] = length
    #
    #     for i in range(2, M + 1):
    #         for k in range(2, min(i, pickers) + 1):
    #             temp = float('inf')
    #             for j in range(k - 1, i):
    #                 path, rr_length = self.find_shortest_path(self._divide_orders_by_aisles(orders, j + 1, i))
    #                 F_jk_1_length = F[(j, k - 1)] if (j, k - 1) in F else float('inf')
    #                 temp = min(temp, max(F_jk_1_length, rr_length))
    #             F[(i, k)] = temp
    #
    #     return F

    # def dynamic_zone_picking(self, max_aisle, pickers, orders):
    #     M = max_aisle  # Maximum aisle number
    #     F = {}  # F(i, k) matrix
    #
    #     if not orders:
    #         print("No orders to process.")
    #         return  # Add appropriate handling for empty orders
    #
    #     # Initialize F matrix
    #     for i in range(1, M + 1):
    #     # for i in range(1, M + 1):
    #         path, length = self.find_shortest_path(self._divide_orders_by_aisles(orders, 1, i))
    #         F[(i, 1)] = length
    #
    #     for i in range(2, M + 1):
    #         for k in range(2, min(i, pickers) + 1):
    #             temp = float('inf')
    #             for j in range(k - 1, i):
    #                 path, rr_length = self.find_shortest_path(self._divide_orders_by_aisles(orders, j + 1, i))
    #                 F_jk_1_length = F[(j, k - 1)] if (j, k - 1) in F else float('inf')
    #                 temp = min(temp, max(F_jk_1_length, rr_length))
    #             F[(i, k)] = temp
    #
    #     return F



layout = [['0' for _ in range(19)] for _ in range(18)]  # Initialize layout with '0's
for i in range(19):  # Adjust top and bottom rows for dummy nodes
    layout[0][i] = 'w'
    layout[17][i] = 'w'
for i in range(1, 18):  # Adjust columns for dummy nodes
    for j in [2, 5, 8, 11, 14, 17]:
        layout[i][j] = 'w'

# spanning_tree_solver = WarehouseSpanningTree(layout)
orders = [(16, 14), (1, 5), (1, 8), (7, 13), (13, 9), (1, 13), (4, 11), (4, 5), (7, 4), (7, 7), (16, 4),
          (16, 7)]
# orders = [(1,5),(7,4)]

pickers = 2     # Number of pickers

spanning_tree_solver = WarehouseSpanningTree(layout)
distance_matrix_df = spanning_tree_solver.generate_distance_matrix(orders)
spanning_tree_solver.plot_spanning_tree()
zone_picking_result = spanning_tree_solver.dynamic_zone_picking(orders, pickers)
print(zone_picking_result)
spanning_tree_solver.plot_path_and_tree(orders)
print(distance_matrix_df)
spanning_tree_solver.plot_distance_matrix(distance_matrix_df)
