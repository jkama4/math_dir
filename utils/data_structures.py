import math
import random

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Optional

class Graph:
    def __init__(self):
        self.nodes: List[int] = []
        self.edges: Dict[int, Dict[int, int]] = {}
        
    def __str__(self):
        return f"nodes: {self.nodes}, edges: {self.edges}"

    def add_node(self, node_id: int):
        self.nodes.append(node_id)

    def add_edge(self, node1: int, node2: int, cost: int):
        if node1 not in self.edges:
            self.edges[node1] = {}
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node1][node2] = cost
        self.edges[node2][node1] = cost
        
    def generate_graph(self, num_nodes: int, num_edges: int):
        self.nodes = list(range(1, num_nodes + 1))
        
        possible_edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        
        if num_edges > len(possible_edges):
            raise ValueError("Too many edges for the given number of nodes.")
        
        selected_edges = random.sample(possible_edges, num_edges)
        for node1, node2 in selected_edges:
            cost = random.randint(1, 10)
            self.add_edge(node1, node2, cost)
            
    def show_graph(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }
    
    def visualise(self):
        n: List[int] = len(self.nodes)
        positions = {}
        for idx, node in enumerate(self.nodes):
            angle = 2 * math.pi * idx / n
            positions[node] = (math.cos(angle), math.sin(angle))
            
        for node, (x, y) in positions.items():
            plt.scatter(x, y, s=100, c='blue')
            plt.text(x, y, str(node), fontsize=8, ha='center', va='center', color='white')
            
        for node, neighbours in self.edges.items():
            for neighbour, cost in neighbours.items():
                if node < neighbour:
                    x1, y1 = positions[node]
                    x2, y2 = positions[neighbour]
                    plt.plot([x1, x2], [y1, y2], color='black')
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    plt.text(mid_x, mid_y, str(cost), fontsize=10, color='red', ha='center', va='center')
        
        plt.axis('equal')
        plt.show()
        
        return "Closed visualisation"