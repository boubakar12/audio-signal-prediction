#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

def f(x): 
    return 2*(x-1)**2 +3

def f_prine(x):
    return 4*(x-1)

def f_double_prime(x): 
    return 4

def gradient_descent(x_0, alpha, iterations):
    xs = [x_0]
    for i in range(iterations):
        x_new = xs[-1] - alpha*f_prine(xs[-1])
        xs.append(x_new)
    return xs  

def newton_methode(x_0, iterations):
    xs = [x_0]
    for i in range(iterations):
        x_new = xs[-1] - (f_prine(xs[-1])/ f_double_prime([xs[-1]]))
        xs.append(x_new)
    return xs


def plot_function( xs, filename, title):
    x_val = np.linspace(-5, 5, 100)
    y_val = f(x_val)
    plt.plot(x_val, y_val, label = 'f(x)')

    y_traj = [ f(x) for x in xs]
    plt.plot(xs, y_traj, color = 'red', label = 'Trajectory')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == '__main__':
    xs1 = gradient_descent(-3, alpha = .20, iterations = 10)
    plot_function(xs1, filename = 'figure1.pdf', title = 'GD alpha = .20')

    xs2 = gradient_descent(-3, alpha = .45, iterations = 10)
    plot_function(xs2, filename = 'figure2.pdf', title = 'GD alpha =.45')

    xs3 =  gradient_descent(-3, alpha = .60, iterations = 10)
    plot_function(xs3, filename = 'figure3.pdf', title = 'GD alpha = .60')

    xs4 = newton_methode(-3, iterations = 10)
    plot_function(xs4, filename = 'figure4.pdf', title = 'Newton Methode x_0 = -3')


# 2 Optimization

def f2(x):
    return x**4 + (4/3)*x**3 -(5/2)*x**2 -3*x + 2

def f2_prime(x):
    return 4*x**3 + 4*x**2 -5*x -3
def f2_double_prime(x):
    return 12*x**2 + 8*x -5
def gradient_descent_f2( x_0, alpha, iterations):
    xs = [x_0]
    for i in range(iterations):
        x_new = xs[-1] - aplah*f_double_prime(x[-1])
        xs.append(x_new)
    return xs





    

















