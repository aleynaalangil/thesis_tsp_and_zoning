import matplotlib.pyplot as plt
import networkx as nx


class WarehouseSpanningTree:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout
        self.graph = self._create_graph_with_dummy_nodes()

    def _create_graph_with_dummy_nodes(self):
        G = nx.Graph()

        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 15) for x in range(19)]
        for y in range(1, 15):
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
                G.add_edge(node, (x + 1, y), weight=0)
            if (x - 1, y) in dummy_nodes:
                G.add_edge(node, (x - 1, y), weight=0)
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

        return G

    # def _create_graph_with_dummy_nodes(self):
    #     G = nx.Graph()
    #
    #     # Define actual and dummy nodes based on the layout provided
    #     dummy_nodes = [(x, 0) for x in range(19)] + [(x, 15) for x in range(19)]
    #     for y in range(1, 15):
    #         dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])
    #
    #     # Add nodes to the graph
    #     for node in dummy_nodes:
    #         G.add_node(node, is_dummy=True)
    #
    #     # Define directions for connectivity (right, left, down, up)
    #     directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    #
    #     # Add edges between dummy nodes
    #     for x, y in dummy_nodes:
    #         for dx, dy in directions:
    #             adjacent_node = (x + dx, y + dy)
    #             if adjacent_node in dummy_nodes:
    #                 weight = 0 if (dx, dy) in [(1, 0), (-1, 0)] else 1
    #                 G.add_edge((x, y), adjacent_node, weight=weight)
    #
    #     # Adding specific edges with different weights
    #     for i in range(2, 17, 3):
    #         G.add_edge((i, 0), (i + 3, 0), weight=2 / 3)
    #         G.add_edge((i, 15), (i + 3, 15), weight=2 / 3)
    #
    #     return G

    def plot_spanning_tree(self):
        mst = nx.minimum_spanning_tree(self.graph)
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates

        plt.figure(figsize=(13, 13))
        # Draw the minimum spanning tree edges
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(13)] + [(x, 16) for x in range(13)]
        for y in range(1, 16):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y)])

        actual_nodes = []
        for x in range(13):
            for y in range(1, 16):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))

        # Draw actual nodes in one color (e.g., green)
        nx.draw_networkx_nodes(mst, pos, nodelist=actual_nodes, node_color='green', node_size=100)

        # Draw dummy nodes in another color (e.g., light blue)
        nx.draw_networkx_nodes(mst, pos, nodelist=dummy_nodes, node_color='lightblue', node_size=50)

        # Labels for the actual nodes
        labels = {node: node for node in actual_nodes}
        nx.draw_networkx_labels(mst, pos, labels, font_size=8)

        plt.legend(['Actual Nodes', 'Dummy Nodes'])
        plt.title("Warehouse Minimum Spanning Tree")
        plt.axis('off')  # Turn off the axis
        plt.show()

    # def find_shortest_path(self, orders):
    #     """Find the shortest path that visits all order items and returns to the start."""
    #     if not orders:
    #         print("No orders provided.")
    #         return [], 0
    #
    #     print("Received Orders:", orders)
    #
    #     # Convert order items to nodes in the graph
    #     order_nodes = [order for order in orders if order in self.graph.nodes()]
    #
    #     print("Order Nodes in Graph:", order_nodes)
    #
    #     if not order_nodes:
    #         print("No matching nodes found in the graph for the given orders.")
    #         return [], 0
    #
    #     shortest_path = []
    #     total_length = 0
    #     picked_order = []  # List to store the order of item picks # Added later
    #     for i in range(len(order_nodes) - 1):
    #         path = nx.shortest_path(self.graph, source=order_nodes[i], target=order_nodes[i + 1], weight='weight')
    #         path_length = nx.shortest_path_length(self.graph, source=order_nodes[i], target=order_nodes[i + 1],
    #                                               weight='weight')
    #         print(f"Path from {order_nodes[i]} to {order_nodes[i + 1]}: {path}, Length: {path_length}")
    #         shortest_path.extend(path[:-1])
    #         total_length += path_length
    #         picked_order.append(order_nodes[i])  # Added later
    #
    #     # Add the path back to the starting node
    #     if order_nodes:
    #         return_to_start_path = nx.shortest_path(self.graph,
    #                                                 source=order_nodes[-1],
    #                                                 target=order_nodes[0],
    #                                                 weight='weight')
    #         return_to_start_length = nx.shortest_path_length(self.graph,
    #                                                          source=order_nodes[-1],
    #                                                          target=order_nodes[0],
    #                                                          weight='weight')
    #         print(
    #             f"Return path to start {order_nodes[0]} from {order_nodes[-1]}: {return_to_start_path}, Length: {return_to_start_length}")
    #         shortest_path.extend(return_to_start_path[1:])  # Skip the first node to avoid duplicates
    #         total_length += return_to_start_length
    #         picked_order.append(order_nodes[-1])  # Add the last order item to the picked order list # Added later
    #
    #     shortest_path.append(order_nodes[0])  # Add the starting node to complete the cycle
    #     picked_order.append(order_nodes[0])  # Added later
    #
    #     print("Calculated Shortest Path:", shortest_path, "\nLength of the Shortest Path:", total_length)
    #     print("Order of Items Picked:", picked_order)  # Added later
    #
    #     return shortest_path  # Added later picked_order

    def find_shortest_path(self, orders):
        """Find a short path to visit all order items, selecting the best starting point."""
        if not orders:
            print("No orders provided.")
            return [], 0

        print("Received Orders:", orders)
        best_path = []
        best_length = float('inf')
        best_order_sequence = []  # To store the order in which items are picked

        for start_node in orders:
            visited = set()
            current_node = start_node
            shortest_path = [current_node]
            total_length = 0
            order_sequence = [current_node]  # To store the sequence for this starting point

            while len(visited) < len(orders):
                visited.add(current_node)
                next_node = None
                min_distance = float('inf')

                for node in orders:
                    if node not in visited:
                        distance = nx.shortest_path_length(self.graph, current_node, node, weight='weight')
                        if distance < min_distance:
                            min_distance = distance
                            next_node = node

                if next_node:
                    total_length += min_distance
                    shortest_path.extend(nx.shortest_path(self.graph, current_node, next_node, weight='weight')[1:])
                    current_node = next_node
                    order_sequence.append(current_node)  # Add to sequence
                else:
                    break

            # Return to start if not already at the starting point
            if current_node != start_node:
                return_to_start_length = nx.shortest_path_length(self.graph, current_node, start_node, weight='weight')
                total_length += return_to_start_length
                shortest_path.extend(nx.shortest_path(self.graph, current_node, start_node, weight='weight')[1:])
                order_sequence.append(start_node)  # Complete the cycle in the sequence

            if total_length < best_length:
                best_length = total_length
                best_path = shortest_path
                best_order_sequence = order_sequence  # Store the best order sequence

        print("Calculated Shortest Path:", best_path, "\nTotal Length:", best_length)
        print("Order of Items Picked:", best_order_sequence)
        return best_path

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

        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 15) for x in range(19)]
        for y in range(1, 15):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])

        actual_nodes = []
        for x in range(19):
            for y in range(1, 15):
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

        plt.legend(['Shortest Path', 'Actual Nodes', 'Dummy Nodes'])
        plt.title("Warehouse Path and Spanning Tree")
        plt.axis('off')
        plt.show()


layout = [['0' for _ in range(19)] for _ in range(16)]  # Initialize layout with '0's
for i in range(19):  # Adjust top and bottom rows for dummy nodes
    layout[0][i] = 'w'
    layout[15][i] = 'w'
for i in range(1, 15):  # Adjust columns for dummy nodes
    for j in [2, 5, 8, 11, 14, 17]:
        layout[i][j] = 'w'

# spanning_tree_solver = WarehouseSpanningTree(layout)
orders = [(1, 13), (4, 11), (4, 5), (7, 4), (7, 7), (7, 13), (13, 9), (16, 14), (16, 7), (16, 4), (1, 5), (1, 8)
          ]  # Correct format for orders

spanning_tree_solver = WarehouseSpanningTree(layout)
spanning_tree_solver.plot_path_and_tree(orders)
