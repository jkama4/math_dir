import numpy as np
import matplotlib.pyplot as plt

import math_files.math_structures as ms
import math_files.math_functions as mf
import math_files.data_structures as ds

point_a: ms.Point = ms.Point(x_val=3, y_val=6)
point_b: ms.Point  = ms.Point(x_val=7, y_val=1)

line_segment: ms.LineSegment = ms.LineSegment(point_a=point_a, point_b=point_b)

linear_function: mf.LinearFunction = mf.LinearFunction(x=5, slope=3, intercept=8)
parabola: mf.Parabola = mf.Parabola(x=3, m=5, intercept=4)

graph: ds.Graph = ds.Graph()
graph.generate_graph(num_nodes=9, num_edges=15)

line_segment.visualise()