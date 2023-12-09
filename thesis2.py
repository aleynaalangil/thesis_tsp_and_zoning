import matplotlib.pyplot as plt
import networkx as nx


class WarehouseSpanningTree:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout
        self.graph = self._create_graph_with_dummy_nodes()

    def _create_graph_with_dummy_nodes(self):
        G = nx.Graph()

        # Define actual and dummy nodes based on the layout provided
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 15) for x in range(19)]
        for y in range(1, 15):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])
        print("Dummy Nodes from create_graph:", dummy_nodes)
        actual_nodes = []
        for x in range(19):
            for y in range(1, 15):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))
                    G.add_node((x, y), is_dummy=False)  # Add actual nodes to the graph
        print("Actual Nodes from create_graph:", actual_nodes)

        # Add nodes to the graph
        for node in dummy_nodes:
            G.add_node(node, is_dummy=True)

        # Define directions for connectivity (right, left, down, up)
        # directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        #
        # # Add edges between dummy nodes
        # for x, y in dummy_nodes:
        #     for dx, dy in directions:
        #         adjacent_node = (x + dx, y + dy)
        #         if adjacent_node in dummy_nodes:
        #             weight = 0 if (dx, dy) in [(1, 0), (-1, 0)] else 1
        #             G.add_edge((x, y), adjacent_node, weight=weight)

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

        # Adding specific edges with different weights
        for i in range(2, 17, 3):
            G.add_edge((i, 0), (i + 3, 0), weight=2 / 3)
            G.add_edge((i, 15), (i + 3, 15), weight=2 / 3)

        return G

    def plot_spanning_tree(self):
        mst = nx.minimum_spanning_tree(self.graph)
        print("MST Edges from plot_spanning_tree:", mst.edges())
        pos = {node: (node[0], -node[1]) for node in
               self.graph.nodes()}  # Position nodes based on their grid coordinates
        print("Positions from plot_spanning_tree::", pos)

        plt.figure(figsize=(13, 13))
        # Draw the minimum spanning tree edges
        nx.draw_networkx_edges(mst, pos, alpha=0.5, width=1)
        nx.draw_networkx_edge_labels(mst, pos, edge_labels=None, label_pos=0.5, font_size=8)

        # # Define actual and dummy nodes based on the layout provided
        # dummy_nodes = [(x, 0) for x in range(13)] + [(x, 16) for x in range(13)]
        # for y in range(1, 16):
        #     dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y)])
        #
        # actual_nodes = []
        # for x in range(13):
        #     for y in range(1, 16):
        #         if (x, y) not in dummy_nodes:
        #             actual_nodes.append((x, y))

        # Define actual and dummy nodes based on the layout provided
        # Correctly define dummy nodes based on the actual graph layout
        dummy_nodes = [(x, 0) for x in range(19)] + [(x, 15) for x in range(19)]
        for y in range(1, 15):
            dummy_nodes.extend([(2, y), (5, y), (8, y), (11, y), (14, y), (17, y)])
        print("Dummy Nodes from plot_spanning_tree:", dummy_nodes)
        actual_nodes = []
        for x in range(19):
            for y in range(1, 15):
                if (x, y) not in dummy_nodes:
                    actual_nodes.append((x, y))
                    # G.add_node((x, y), is_dummy=False)  # Add actual nodes to the graph
        print("Actual Nodes from plot_spanning_tree:", actual_nodes)
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
    #     """Find a short path to visit all order items, selecting the best starting point."""
    #     if not orders:
    #         print("No orders provided.")
    #         return [], 0
    #
    #     print("Received Orders:", orders)
    #     best_path = []
    #     best_length = float('inf')
    #     best_order_sequence = []  # To store the order in which items are picked
    #
    #     for start_node in orders:
    #         visited = set()
    #         current_node = start_node
    #         shortest_path = [current_node]
    #         total_length = 0
    #         order_sequence = [current_node]  # To store the sequence for this starting point
    #
    #         while len(visited) < len(orders):
    #             visited.add(current_node)
    #             next_node = None
    #             min_distance = float('inf')
    #
    #             for node in orders:
    #                 if node not in visited:
    #                     distance = nx.shortest_path_length(self.graph, current_node, node, weight='weight')
    #                     if distance < min_distance:
    #                         min_distance = distance
    #                         next_node = node
    #
    #             if next_node:
    #                 total_length += min_distance
    #                 shortest_path.extend(nx.shortest_path(self.graph, current_node, next_node, weight='weight')[1:])
    #                 current_node = next_node
    #                 order_sequence.append(current_node)  # Add to sequence
    #             else:
    #                 break
    #
    #         # Return to start if not already at the starting point
    #         if current_node != start_node:
    #             return_to_start_length = nx.shortest_path_length(self.graph, current_node, start_node, weight='weight')
    #             total_length += return_to_start_length
    #             shortest_path.extend(nx.shortest_path(self.graph, current_node, start_node, weight='weight')[1:])
    #             order_sequence.append(start_node)  # Complete the cycle in the sequence
    #
    #         if total_length < best_length:
    #             best_length = total_length
    #             best_path = shortest_path
    #             best_order_sequence = order_sequence  # Store the best order sequence
    #
    #     print("Calculated Shortest Path:", best_path, "\nTotal Length:", best_length)
    #     print("Order of Items Picked:", best_order_sequence)
    #     return best_path

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
            print(f"Trying start_node: {start_node}")
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
                    print(f"Moving to next_node: {next_node} with total_length: {total_length}")
                else:
                    break

            if current_node != start_node:
                return_to_start_length = nx.shortest_path_length(self.graph, current_node, start_node, weight='weight')
                total_length += return_to_start_length
                shortest_path.extend(nx.shortest_path(self.graph, current_node, start_node, weight='weight')[1:])
                order_sequence.append(start_node)  # Complete the cycle in the sequence
                print(f"Returning to start_node: {start_node} with total_length: {total_length}")

            if total_length < best_length:
                best_length = total_length
                best_path = shortest_path
                best_order_sequence = order_sequence  # Store the best order sequence
                print(f"New best path found with length: {best_length}")

        print("Calculated Shortest Path:", best_path, "\nTotal Length:", best_length)
        print("Order of Items Picked:", best_order_sequence)
        return best_path


    def plot_path_and_tree(self, orders):
        # Generate the minimum spanning tree from the graph
        mst = nx.minimum_spanning_tree(self.graph)
        print("MST Edges from plot_path_and_tree:", mst.edges())

        # Define positions for all nodes in the original graph
        pos = {node: (node[0], -node[1]) for node in self.graph.nodes()}
        print("Positions from plot_path_and_tree:", pos)

        plt.figure(figsize=(13, 13))

        # Draw all nodes and edges of the entire graph
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightgrey', node_size=50)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, edge_color='lightgrey')

        # Overlay the MST edges
        nx.draw_networkx_edges(mst, pos, edge_color='blue', width=2)

        # Define and draw the shortest path if it exists
        shortest_path, _ = self.find_shortest_path(orders)
        if shortest_path:
            path_edges = list(zip(shortest_path, shortest_path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='red', width=2)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=shortest_path, node_color='red', node_size=50)

        # Add node labels for the actual nodes
        actual_nodes = [node for node in self.graph.nodes() if not self.graph.nodes[node]['is_dummy']]
        nx.draw_networkx_labels(self.graph, pos, {node: node for node in actual_nodes}, font_size=8)

        # Set plot title and legend
        plt.legend(['Original Graph', 'Minimum Spanning Tree', 'Shortest Path'])
        plt.title("Warehouse Path and Spanning Tree")
        plt.axis('off')  # Turn off the axis
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
spanning_tree_solver.plot_spanning_tree()
spanning_tree_solver.plot_path_and_tree(orders)
