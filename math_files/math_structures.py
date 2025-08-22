import math

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

class Point:
    def __init__(self, x_val: float, y_val: float):
        """
        Initialize a Point object with the given x and y coordinates.

        Parameters:
            x_val (float): The x-coordinate of the point.
            y_val (float): The y-coordinate of the point.
        """
        self.x_val = x_val
        self.y_val = y_val

    def __str__(self):
        """Return a string representation of the Point object, useful for printing.
        """
        return f"x = {self.x_val}, y = {self.y_val}"
    
    def visualise(self):
        plt.figure(figsize=(10,10))
        
        plt.plot(self.x_val, self.y_val, marker='o')
        
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("Point representation")
        plt.grid(True)
        
        plt.show()
        
        return "Closed visualisation"
    
class LineSegment:
    def __init__(self, point_a: Point, point_b: Point):
        """
        Initialize a LineSegment object with the given two points.

        Parameters:
            point_a (Point): The first point.
            point_b (Point): The second point.
        """
        self.point_a = point_a
        self.point_b = point_b
        self.x_coords: Tuple[Point, Point] = tuple([self.point_a.x_val, self.point_b.x_val])
        self.y_coords: Tuple[Point, Point] = tuple([self.point_a.y_val, self.point_b.y_val])

    def __str__(self):
        """Return a string representation of the LineSegment object, useful for printing.
        """
        return f"Line from {self.point_a} to {self.point_b}"
    
    def calc_length(self):
        """
        Calculate the length of this LineSegment.

        Returns:
            float: The length of the line segment.
        """

        delta_x = self.point_b.x_val - self.point_a.x_val
        delta_y = self.point_b.y_val - self.point_a.y_val

        seg_length = math.sqrt(delta_x**2 + delta_y**2)
        return seg_length
    
    def visualise(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.x_coords, self.y_coords, 'b-', marker='o')
        
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title(str(self))
        plt.grid(True)
        
        plt.show()
        
        return "Closed visualisation"
        
        

    

    


