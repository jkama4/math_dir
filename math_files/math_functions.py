import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional

class LinearFunction():
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
        
class Parabola():
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
        return f"f\'(x) = {2*self.m}x"
    
    def visualise(self):
        plt.figure(figsize=(10, 10))
        return "Closed visualation"
        
class SimpleNeuralNetwork():
    """
    Neural Network containing 2 hidden layers with 16 and 8 nodes respectively, and 1 output node"""
    def __init__(self, X: np.array) -> None:
        self.X = X
        self.n = [self.X.shape[0], 16, 8, 1]
        self.W1 = np.random.rand(self.n[0], self.n[1])
        self.b1 = np.random.rand(1, self.n[1])
        self.W2 = np.random.rand(self.n[1], self.n[2])
        self.b2 = np.random.rand(1, self.n[2])
        self.W3 = np.random.rand(self.n[2], self.n[3])
        self.b3 = np.random.rand(1, self.n[3])


    def softmax(self, Z: np.array) -> np.array:
        return 1 / (1 + np.exp(-Z))


    def ReLU(self, Z: np.array) -> np.array:
        return np.maximum(0, Z)


    def cost(y_hat: np.array, y: np.array) -> np.array:
        losses = - ( (y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat) )

        m = y_hat.reshape(-1).shape[0]

        summed_losses = (1 / m) * (np.sum(losses, axis=1))

        return np.sum(summed_losses)


    def forward(self, A0: np.array) -> Tuple[np.array, Dict[str, np.array]]:
        Z1: np.array = np.dot(A0, self.W1) + self.b1
        A1: np.array = self.ReLU(Z=Z1)

        Z2: np.array = np.dot(A1, self.W2) + self.b2
        A2: np.array = self.ReLU(Z=Z2)
        
        Z3: np.array = np.dot(A2, self.W3) + self.b3
        A3: np.array = self.softmax(Z=Z3)

        memory: Dict[str, np.array] = {
            "A0": A0, # input
            "A1": A1,
            "A2": A2,
            "A3": A3 # NN output - y_hat
        }

        return (A3, memory)
    
    
    def backprop_layer_3(y_hat, Y, m, A2, W3):
        A3 = y_hat
        
        dC_dZ3 = (1/m) * (A3 - Y)

        dZ3_dW3 = A2
        dC_dW3 = np.dot(dC_dZ3, dZ3_dW3.T)

        dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)

        dZ3_dA2 = W3 
        dC_dA2 = np.dot(dZ3_dA2.T, dC_dZ3)

        return dC_dW3, dC_db3, dC_dA2
    
    def backprop_layer_2(propagator_dC_dA2, A1: np.array, A2: np.array, W2:np.array):
        dA2_dZ2 = A2 * (1 - A2)
        dC_dZ2 = propagator_dC_dA2 * dA2_dZ2

        dZ2_dW2 = A1
        dC_dW2 = np.dot(dC_dZ2, dZ2_dW2.T)

        dC_db2 = np.sum(dC_dW2, axis=1, keepdims=True)

        dZ2_dA1 = W2
        dC_dA1 = np.dot(dZ2_dA1.T, dC_dZ2)

        return dC_dW2, dC_db2, dC_dA1


    def backprop_layer_1(propagator_dC_dA1, A1, A0, W1):
        dA1_dZ1 = A1 * (1 - A1)
        dC_dZ1 = propagator_dC_dA1 * dA1_dZ1

        dZ1_dW1 = A0
        dC_dW1 = np.dot(dC_dZ1, dZ1_dW1.T)

        dC_db1 = np.sum(dC_dW1, axis=1, keepdims=True)

        return dC_dW1, dC_db1
