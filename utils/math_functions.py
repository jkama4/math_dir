import math

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Optional

class LinearFunction:
    def __init__(self, x: float, slope: float, intercept: float):
        """
        Initialize a LinearFunction object with the given x, y, and slope.

        Parameters:
            x (float): The input.
            y (float): The output.
            slope (float): The slope of the function.
        """
        
        self.x = x
        self.slope = slope
        self.intercept = intercept
        self.y = self.evaluate()

    def __str__(self):
        """Return a string representation of the LinearFunction object, useful for printing.
        
        Returns:
            str: A string of the form "f(x) = y".
        """
        return f"f(x) = {self.slope}x + {self.intercept}"
    
    def show_elements(self):
        return {
            "x": self.x,
            "y": self.y,
            "slope": self.slope,
            "intercept": self.intercept
        }
    
    def equation(self):
        """Return the equation of the linear function in slope-intercept form.
        
        Returns:
            str: A string of the form "y = mx + b".
        """
        return f"y = {self.slope}x + {self.intercept}"
    
    def evaluate(self):
        """Evaluate the linear function at x.

        Returns:
            float: The output of the function at x.
        """
        return self.slope * self.x + self.intercept
    
    def derivative(self):
        return f"f'(x) = {self.slope}"
    
    def visualise(self):
        plt.figure(figsize=(10, 10))
        
        x_values: np.linspace = np.linspace(-10, 10, 100)
        y_values: np.linspace = self.slope * x_values + self.intercept
        
        plt.plot(x_values, y_values, label=self.equation())
        
        plt.scatter(self.x, self.y, color='red', zorder=5, label=f"{self.x}, {self.y}")
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(self.equation())
        plt.legend()
        plt.grid(True)
        
        plt.show()
        
        return "Closed visualisation"       
        
class Parabola:
    def __init__(self, x: float, m: float, intercept: float):
        self.x = x
        self.m = m
        self.intercept = intercept
        self.y = self.evaluate()
        
    def __str__(self):
        return f"f(x) = {self.m}x^2 + {self.intercept}"
    
    def show_elements(self):
        return {
            "x": self.x,
            "y": self.y,
            "m": self.m,
            "intercept": self.intercept
        }
    
    def evaluate(self):
        return self.m * (self.x**2) + self.intercept
    
    def derivative(self):
        return f"f'(x) = {2*self.m}x"
    
    def visualise(self):
        plt.figure(figsize=(10, 10))
        
        return "Closed visualation"
        
        