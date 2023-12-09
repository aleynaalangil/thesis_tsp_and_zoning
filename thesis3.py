import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

class WarehouseSpanningTree:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout
        self.graph = self._create_graph_with_dummy_nodes()

    def _create_graph_with_dummy_nodes(self):
        G = nx.Graph()

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 16) for x in range(19)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 15):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

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
                G.add_edge((i, 0), (i+1, 0), weight=2 / 5)
            if (i, 16) in dummy_nodes:  # Check if the edge endpoint is in dummy_nodes
                G.add_edge((i, 16), (i+1, 16), weight=2 / 5)



        G.add_edge((2,0),(2,1),weight=2 / 5)
        G.add_edge((5,0),(5,1),weight=2 / 5)
        G.add_edge((8,0),(8,1),weight=2 / 5)
        G.add_edge((11,0),(11,1),weight=2 / 5)
        G.add_edge((14,0),(14,1),weight=2 / 5)
        G.add_edge((17,0),(17,1),weight=2 / 5)

        G.add_edge((2,15),(2,16),weight=2 / 5)
        G.add_edge((5,15),(5,16),weight=2 / 5)
        G.add_edge((8,15),(8,16),weight=2 / 5)
        G.add_edge((11,15),(11,16),weight=2 / 5)
        G.add_edge((14,15),(14,16),weight=2 / 5)
        G.add_edge((17,15),(17,16),weight=2 / 5)


        G.add_edge((0, 15), (0, 16), weight=float('inf'))
        G.add_edge((1, 15), (1, 16), weight=float('inf'))
        G.add_edge((3, 15), (3, 16), weight=float('inf'))
        G.add_edge((4, 15), (4, 16), weight=float('inf'))
        G.add_edge((6, 15), (6, 16), weight=float('inf'))
        G.add_edge((7, 15), (7, 16), weight=float('inf'))
        G.add_edge((9, 15), (9, 16), weight=float('inf'))
        G.add_edge((10, 15), (10, 16), weight=float('inf'))
        G.add_edge((12, 15), (12, 16), weight=float('inf'))
        G.add_edge((13, 15), (13, 16), weight=float('inf'))
        G.add_edge((15, 15), (15, 16), weight=float('inf'))
        G.add_edge((16, 15), (16, 16), weight=float('inf'))
        G.add_edge((18, 15), (18, 16), weight=float('inf'))
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
        return G


    # def plot_spanning_tree(self):
    #     mst = nx.minimum_spanning_tree(self.graph)
    #     pos = {node: (node[0], -node[1]) for node in
    #            self.graph.nodes()}  # Position nodes based on their grid coordinates
    #
    #     plt.figure(figsize=(13, 13))
    #     nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)
    #
    #     # Update the dummy nodes to match your graph layout
    #     dummy_nodes = [(x, 0) for x in range(19)] + [(x, 15) for x in range(19)]
    #     for y in range(1, 15):
    #         dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])
    #
    #     actual_nodes = []
    #     for x in range(19):
    #         for y in range(1, 15):
    #             if (x, y) not in dummy_nodes:
    #                 actual_nodes.append((x, y))
    #
    #     # Draw actual nodes in one color (e.g., green)
    #     nx.draw_networkx_nodes(mst, pos, nodelist=actual_nodes, node_color='green', node_size=100)
    #
    #     # Draw dummy nodes in another color (e.g., light blue)
    #     nx.draw_networkx_nodes(mst, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
    #
    #     # Labels for the actual nodes
    #     labels = {node: node for node in actual_nodes}
    #     nx.draw_networkx_labels(mst, pos, labels, font_size=8)
    #
    #     plt.legend(['Actual Nodes', 'Dummy Nodes'])
    #     plt.title("Warehouse Minimum Spanning Tree")
    #     plt.axis('off')  # Turn off the axis
    #     plt.show()

    def find_shortest_path(self, orders):
        """Find the shortest path for TSP, considering each order as a possible starting point."""
        if not orders:
            print("No orders provided.")
            return [], 0

        best_path = []
        best_length = float('inf')

        for start_node in orders:
            # Create a cycle with the start node as the first and last element
            tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]

            # Find the TSP path
            tsp_path = nx.approximation.traveling_salesman_problem(self.graph, nodes=tsp_orders, cycle=True)

            # Calculate the total length of the path
            total_length = sum(
                self.graph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))

            # Check if this is the shortest path so far
            if total_length < best_length:
                best_length = total_length
                best_path = tsp_path

            # Filter out dummy nodes from the best path
        collected_orders_path = [node for node in best_path if self.graph.nodes[node]['is_dummy'] == False]
        print("Collected Orders Path:", collected_orders_path)
        print("Best Starting Point TSP Path:", best_path, "\nLength of the Path:", best_length)
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


    def plot_path_and_tree(self, orders):
        """Plot the spanning tree and the shortest path for given orders."""
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}  # Positions based on grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        # dummy_nodes = [(x, 0) for x in range(13)] + [(x, 16) for x in range(13)]
        # for y in range(1, 16):
        #     dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y)])
        #
        # actual_nodes = []
        # for x in range(13):
        #     for y in range(1, 16):
        #         if (x, y) not in dummy_nodes:
        #             actual_nodes.append((x, y))

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

    # def generate_distance_matrix(self, orders):
    #     """Generate a labeled distance matrix for the given orders."""
    #     num_orders = len(orders)
    #     distance_matrix = [[0 for _ in range(num_orders)] for _ in range(num_orders)]
    #
    #     # Create a dictionary to label orders for easier readability
    #     order_labels = {index: f"Order {index + 1}" for index in range(num_orders)}
    #
    #     for i in range(num_orders):
    #         for j in range(num_orders):
    #             if i != j:
    #                 distance_matrix[i][j] = nx.shortest_path_length(self.graph, orders[i], orders[j], weight='weight')
    #
    #     # Create a pandas DataFrame for better visualization
    #     df = pd.DataFrame(distance_matrix, index=order_labels.values(), columns=order_labels.values())
    #
    #     return df
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

# Example layout based on the provided node positions
# layout = [['0' for _ in range(13)] for _ in range(17)]  # Initialize layout with '0's
# for i in range(13):  # Top and bottom rows are dummy nodes
#     layout[0][i] = 'w'
#     layout[16][i] = 'w'
# for i in range(1, 16):  # Columns 2, 5, 8, and 11 are dummy nodes
#     for j in [2, 5, 8, 11]:
#         layout[i][j] = 'w'

layout = [['0' for _ in range(19)] for _ in range(17)]  # Initialize layout with '0's
for i in range(19):  # Adjust top and bottom rows for dummy nodes
    layout[0][i] = 'w'
    layout[16][i] = 'w'
for i in range(1, 16):  # Adjust columns for dummy nodes
    for j in [2, 5, 8, 11, 14, 17]:
        layout[i][j] = 'w'

# spanning_tree_solver = WarehouseSpanningTree(layout)
orders = [(16, 14), (1, 5), (1, 8), (7, 13), (13, 9), (1, 13), (4, 11), (4, 5), (7, 4), (7, 7), (16, 4),
          (16, 7)]  # Correct format for orders
# original -starts
# spanning_tree_solver = WarehouseSpanningTree(layout)
# spanning_tree_solver.plot_path_and_tree(orders)
# original -ends

spanning_tree_solver = WarehouseSpanningTree(layout)
spanning_tree_solver.plot_spanning_tree()
shortest_path = spanning_tree_solver.find_shortest_path(orders)
spanning_tree_solver.plot_path_and_tree(orders)
distance_matrix_df = spanning_tree_solver.generate_distance_matrix(orders)

# Print the DataFrame for text-based output
print(distance_matrix_df)

# Plotting the distance matrix
spanning_tree_solver.plot_distance_matrix(distance_matrix_df)

