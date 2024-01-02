import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import gc
import sys
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

sys.path.append('./parser.py')
import parser as parser
from datetime import datetime


class WarehouseSpanningTree:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout
        self.graph = self._create_graph_with_dummy_nodes()

    def _create_graph_with_dummy_nodes(self):
        G = nx.Graph()
        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(62)] + [(x, 46) for x in range(62)]
        for y in range(1, 46):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y), (20, y), (23, y), (26, y), (29, y),
                                (32, y), (35, y), (38, y), (41, y), (44, y), (47, y), (50, y), (53, y), (56, y),
                                (59, y)])

        actual_nodes = []
        for x in range(62):
            for y in range(1, 46):
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
        for i in range(0, 62, 3):
            if (i, 0) in dummy_nodes:  # Check if the edge endpoint is in dummy_nodes
                G.add_edge((i, 0), (i + 1, 0), weight=0)
            if (i, 46) in dummy_nodes:  # Check if the edge endpoint is in dummy_nodes
                G.add_edge((i, 46), (i + 1, 46), weight=0)

        for i in range(0, 45, 3):
            if (0, i) in dummy_nodes:
                G.add_edge((0, i), (1, i), weight=float('inf'))

        aisles = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]

        for i in range(len(aisles)):
            G.add_edge((aisles[i], 0), (aisles[i], 1), weight=0)
            G.add_edge((aisles[i], 45), (aisles[i], 46), weight=0)

        non_aisles = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37,
                      39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61]
        for i in range(len(non_aisles)):
            G.add_edge((non_aisles[i], 0), (non_aisles[i], 1), weight=float('inf'))
            G.add_edge((non_aisles[i], 45), (non_aisles[i], 46), weight=float('inf'))

        for i in range(0, 46, 3):
            for j in range(0, 45):
                G.add_edge((i, j), (i + 1, j), weight=float('inf'))

        for i in range(len(non_aisles)):
            for j in range(0, 45):
                G.add_edge((non_aisles[i], j), (non_aisles[i], j + 1), weight=float('inf'))

        for i in range(0, 38):
            for j in range(26, 46):
                G.add_edge((i, j), (i + 1, j), weight=float('inf'))
                G.add_edge((i, j), (i, j + 1), weight=float('inf'))
                dummy_nodes.append((i, j))
        for i in range(0, 37, 3):
            G.add_edge((i, 25), (i + 1, 25), weight=0)

        for i in range(len(aisles) - 7):
            G.add_edge((aisles[i], 24), (aisles[i], 25), weight=0)

        for i in range(0, 39):
            G.add_edge((i, 25), (i + 1, 25), weight=1)

        for i in range(0, 47, 3):
            G.add_edge((i, 0), (i + 1, 0), weight=0)
        for i in range(0, 39, 3):
            G.add_edge((i, 25), (i + 1, 25), weight=0)

        return G

    def find_shortest_path(self, clusters):
        """Find the shortest path for TSP, considering each order as a possible starting point."""
        if not clusters:
            print("No orders provided.")
            return [], 0

        clusters_best_paths = {}
        # print("clusters from tsp: ", clusters)
        # print("number of picker\n", len(clusters))

        for cluster, orders in clusters.items():
            # print("Clusters from first for: ", orders, "in cluster ", cluster)

            # Initialize the best path variables for each cluster
            best_length = float('inf')
            best_path = None

            for start_node in orders:
                # Create a cycle with the start node as the first and last element
                # tsp_orders = [start_node] + [o for o in orders if o != start_node] + [start_node]

                # Find the TSP path
                tsp_path = nx.approximation.traveling_salesman_problem(self.graph, nodes=orders, cycle=True,
                                                                       weight='weight')

                # Calculate the total length of the path
                total_length = sum(
                    self.graph.edges[tsp_path[i], tsp_path[i + 1]]['weight'] for i in range(len(tsp_path) - 1))

                # Compare and update the best path for this cluster
                if total_length < best_length:
                    best_length = total_length
                    best_path = tsp_path

            # After checking all starting points, finalize the best path for this cluster
            if best_path:
                collected_orders_path = [node for node in best_path if not self.graph.nodes[node]['is_dummy']]
                print(f"Best Path for Cluster {cluster} with Length {best_length}: {best_path}")
                # print(f"Collected Orders Path for Cluster {cluster}: {collected_orders_path}")
                clusters_best_paths[cluster] = best_path
                print("Clusters best paths: ", clusters_best_paths)
        self.plot_path_and_tree(clusters_best_paths)
        return clusters_best_paths

    def plot_spanning_tree(self):
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates

        plt.figure(figsize=(13, 13))
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(62)] + [(x, 45) for x in range(62)]
        for y in range(1, 46):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y), (20, y), (23, y), (26, y), (29, y),
                                (32, y), (35, y), (38, y), (41, y), (44, y), (47, y), (50, y), (53, y), (56, y),
                                (59, y)])

        actual_nodes = []
        for x in range(62):
            for y in range(1, 46):
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

    def plot_spanning_tree(self):
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates

        plt.figure(figsize=(60, 60))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(62)] + [(x, 46) for x in range(62)]
        for y in range(1, 46):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y), (20, y), (23, y), (26, y), (29, y),
                                (32, y), (35, y), (38, y), (41, y), (44, y), (47, y), (50, y), (53, y), (56, y),
                                (59, y)])

        actual_nodes = []
        for x in range(62):
            for y in range(1, 46):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))
        for i in range(0, 38):
            for j in range(25, 47):
                dummy_nodes.append((i, j))

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
        """Generate a distance matrix for the given orders with simplified labeling."""
        distance_matrix = {}
        order_mapping = {}

        for i, from_node in enumerate(orders):
            distances = []
            order_mapping[i + 1] = from_node  # Map the simplified label to the actual order coordinates

            for to_node in orders:
                if from_node == to_node:
                    distance = 0
                else:
                    distance = nx.shortest_path_length(self.graph, from_node, to_node, weight='weight')
                distances.append(distance)

            distance_matrix[i + 1] = distances  # Use simplified integer labels

        # Convert the dictionary to a DataFrame with integer labels
        distance_matrix_df = pd.DataFrame(distance_matrix,
                                          index=[i + 1 for i in range(len(orders))])
        # print("Order Mapping:", order_mapping)  # Added for debugging
        # print("type from distance matrix: ",type(order_mapping))
        # Return both the distance matrix and the mapping
        return dict(order_mapping), distance_matrix_df  # this was returning distance_matrix_df before

    def generate_distance_matrix_google(self, orders):
        """Generate a distance matrix for the given orders in the desired dictionary format."""
        distance_matrix = {}
        order_mapping = {}

        for i, from_node in enumerate(orders):
            distances = {}
            order_mapping[i] = from_node  # Map the simplified label to the actual order coordinates

            for j, to_node in enumerate(orders):
                if from_node == to_node:
                    distance = 0
                else:
                    distance = nx.shortest_path_length(self.graph, from_node, to_node, weight='weight')
                distances[j] = distance

            distance_matrix[i] = distances

        return dict(order_mapping), distance_matrix

    def plot_distance_matrix(self, distance_matrix):
        """Plot the distance matrix."""
        plt.figure(figsize=(15, 15))
        sns.heatmap(distance_matrix, annot=True, cmap="YlGnBu", fmt=".2f")  # Using .2f for floating-point numbers
        plt.title("Distance Matrix between Orders")
        plt.show()

    def hierarchical_clustering(self, distance_matrix, order_mapping, max_d=6, n_clusters=1):
        """Perform hierarchical clustering on the distance matrix using Ward's method."""
        # Convert the distance matrix to a condensed format required by the linkage function
        # We only need the upper triangular part of the distance matrix, excluding the diagonal
        # print("order mapping from clustering: ", order_mapping)
        condensed_distance_matrix = distance_matrix.to_numpy()[np.triu_indices(len(distance_matrix), k=1)]
        # Perform hierarchical clustering using Ward's method
        linkage_matrix = linkage(condensed_distance_matrix, method='ward')

        # Plot the dendrogram
        plt.figure(figsize=(10, 8))
        dendrogram(linkage_matrix, labels=distance_matrix.index)
        plt.title("Hierarchical Clustering Dendrogram (Ward's method)")
        plt.xlabel("Order")
        plt.ylabel("Distance")
        plt.show()
        # Determine clusters, either by a given number of clusters or a distance threshold
        if n_clusters is not None:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        elif max_d is not None:
            cluster_labels = fcluster(linkage_matrix, max_d, criterion='distance')
        else:
            raise ValueError("Either max_d or n_clusters must be specified.")

        # Map the cluster labels back to the order coordinates
        clusters = {i: [] for i in range(1, n_clusters + 1)}
        # print("cluster labels: ", cluster_labels)
        # print("clusters from 514: ", clusters)

        for order_label, cluster_label in enumerate(cluster_labels, start=1):
            # print("order label: ", order_label, "and cluster label: ", cluster_label)
            order_coord = order_mapping[order_label]
            # print("order coord: ", order_coord)
            clusters[cluster_label].append(order_coord)

        return clusters, cluster_labels

    def plot_path_and_tree(self, cluster_paths):
        """Plot the paths for each cluster on the graph."""
        pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}  # Position based on grid coordinates

        plt.figure(figsize=(60, 60))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=1)

        # Define and draw nodes
        dummy_nodes = [(x, 0) for x in range(62)] + [(x, 46) for x in range(62)]
        for y in range(1, 46):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y), (20, y), (23, y), (26, y), (29, y),
                                (32, y), (35, y), (38, y), (41, y), (44, y), (47, y), (50, y), (53, y), (56, y),
                                (59, y)])

        for i in range(0, 38):
            for j in range(25, 47):
                dummy_nodes.append((i, j))

        actual_nodes = [node for node in self.graph.nodes() if node not in dummy_nodes]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=actual_nodes, node_color='green', node_size=100)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)
        nx.draw_networkx_labels(self.graph, pos, {node: node for node in actual_nodes}, font_size=8)

        # Draw the best path for each cluster
        colors = ['red', 'blue', 'black', 'purple', 'orange']  # Add more colors if you have more clusters
        for cluster_id, path in cluster_paths.items():
            cluster_color = colors[cluster_id % len(colors)]
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color=cluster_color, width=2)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color=cluster_color, node_size=50)

        plt.legend(['Actual Nodes', 'Dummy Nodes'] + [f'Cluster {i + 1} Path' for i in range(len(cluster_paths))])
        plt.title("Warehouse Graph with Clustered Paths")
        plt.axis('off')
        plt.show()

    def k_means_clustering(self, orders, n_clusters=1):
        """Perform K-Means clustering on the orders and visualize the results."""
        if not orders:
            raise ValueError("No orders provided.")

        # If orders is a list of coordinates, directly convert it to a numpy array
        order_coords = np.array(orders)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(order_coords)
        labels = kmeans.labels_

        # Visualization
        plt.figure(figsize=(10, 8))
        for cluster in range(n_clusters):
            cluster_points = order_coords[labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
        plt.title('K-Means Clustering of Orders')
        plt.xlabel('Coordinate X')
        plt.ylabel('Coordinate Y')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Map the cluster labels back to the order coordinates
        clusters = {i: [] for i in range(n_clusters)}
        for label, order_coord in zip(labels, order_coords):
            clusters[label].append(tuple(order_coord))

        return clusters, labels

    def mds_clustering(self, distance_matrix, order_mapping, n_clusters=3):
        """Perform clustering using Multidimensional Scaling followed by K-Means and visualize the results."""
        # Apply MDS to reduce dimensions
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        reduced_matrix = mds.fit_transform(distance_matrix)

        # Apply K-Means clustering on the reduced data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(reduced_matrix)
        cluster_labels = kmeans.labels_

        # Visualize the clusters
        plt.figure(figsize=(10, 8))
        for cluster in range(n_clusters):
            plt.scatter(reduced_matrix[cluster_labels == cluster, 0],  # x-axis is the first dimension
                        reduced_matrix[cluster_labels == cluster, 1],  # y-axis is the second dimension
                        label=f'Cluster {cluster + 1}')
        plt.title('Cluster Visualization with MDS and K-Means')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Map the cluster labels back to the order coordinates
        clusters = {i: [] for i in range(1, n_clusters + 1)}
        for order_label, cluster_label in enumerate(cluster_labels, start=1):
            order_coord = order_mapping[order_label]
            clusters[cluster_label + 1].append(order_coord)

        return clusters, cluster_labels, reduced_matrix
def solve():
    layout = [['0' for _ in range(62)] for _ in range(46)]  # Initialize layout with '0's
    for i in range(62):  # Adjust top and bottom rows for dummy nodes
        layout[0][i] = 'w'
        layout[45][i] = 'w'
    for i in range(1, 46):  # Adjust columns for dummy nodes
        for j in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]:
            layout[i][j] = 'w'

    barcode_list = parser.openCSV('./barcode2.csv')
    # orders = parser.get_coordinates(barcode_list)
    orders = parser.get_coordinate_pairs(barcode_list)
    print("Orders: \n", orders)

    spanning_tree_solver = WarehouseSpanningTree(layout)
    spanning_tree_solver.plot_spanning_tree()
    order_mapping, distance_matrix_df = spanning_tree_solver.generate_distance_matrix(orders)
    spanning_tree_solver.plot_distance_matrix(distance_matrix_df)
    print("Order Mapping: \n", order_mapping)
    print("Distance Matrix \n", distance_matrix_df)
    spanning_tree_solver.plot_distance_matrix(distance_matrix_df)
    clusters = spanning_tree_solver.mds_clustering(distance_matrix_df, order_mapping)
    print("Clusters: \n", clusters[0])
    cluster_best_paths = spanning_tree_solver.find_shortest_path(clusters[0])
    spanning_tree_solver.plot_path_and_tree(cluster_best_paths)

if __name__ == "__main__":
    start_time = datetime.now()
    print("Start time:", start_time)
    solve()
    end_time = datetime.now()
    print("End time:", end_time)
    execution_time = end_time - start_time
    print("Execution time:", execution_time)
