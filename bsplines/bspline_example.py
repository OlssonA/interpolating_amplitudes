#!/usr/bin/env python3

from bspline import Bspline
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import testfunctions
from resources import y_to_x_gggH
import math

#Due to using symmetries to compress the training domain the phase space weight
#for gg->Hg needs to be dynamic depending on what the upper value of theta is
def f5_weight(x1, x2, domain = 'validation'):
    min_beta2, max_beta2 = 0.33, 0.99
    beta2 = min_beta2 + (max_beta2 - min_beta2)*x1
    pt_over_mh = 1/(2*math.sqrt(1 - min_beta2)/min_beta2)
    theta_0 = math.asin(pt_over_mh*(2*math.sqrt(1-beta2)/beta2))
    if domain == 'validation':
        theta_H = theta_0 + (math.pi - 2*theta_0)*x2
        J_factor = math.pi - 2*theta_0
    elif domain == 'training':
        theta_H = theta_0 + (math.pi/2 - theta_0)*x2
        J_factor = math.pi/2 - theta_0
    else:
        raise Exception('Invalid phase space domain specified')
    beta2_factor = (1 - 1.0012*beta2)**2/((1 - 0.9802*beta2)*(1 - 0.3357*beta2))
    psw = testfunctions.gggh_ps_weight(beta2, theta_H)
    return psw*beta2_factor*J_factor

if __name__ == "__main__":
    # nr. of data points = (n+1)^dim
    n = 25
    nr_validation_pts = 2000
    dim = 2
    uniform_knots = False #True -> uniform, False -> not-a-knot
    p = 3
    use_weight = 0

    #The bounds of the full parameter space we consider
    validation_bounds = [[0.33, 0.99], [0, np.pi]]
    upper_validation_bounds = [x[1] for x in validation_bounds]
    #The compressed training domain due to utilizing symmetries of gg-->H+j
    training_bounds = [[0.33, 0.99], [0, np.pi/2]]
    upper_training_bounds = [x[1] for x in training_bounds]

    f5_map = testfunctions.f5_map
    phase_space_weight = f5_weight

    #Defined only on the compressed phase space
    def training_function(points):
        amp, full_weight = f5_map(points, bounds = upper_training_bounds)
        if use_weight: return amp*full_weight
        else: return amp

    #Defined on the full phase space domain
    def validation_function(points):
        amp, _ = f5_map(points, bounds = upper_validation_bounds)
        return amp

    bspline = Bspline([p]*dim, [n]*dim, dim, training_function, uniform_knots,
                    check_interpolated_values = True, verbose = True)

    #Remaps a point from the full phase space domain to the training domain and
    #evaluates the approximation. Removes any added weights.
    def interpolate(y):
        x = y_to_x_gggH(y)
        approximation = bspline.evaluate_point(x)
        if use_weight:
            amplitude = approximation/phase_space_weight(*x, domain = 'training')
        else: amplitude = approximation
        return amplitude

    #Uniformly random set of testing points
    validation_pts = np.random.rand(nr_validation_pts, dim)
    true_amplitudes = validation_function(validation_pts)
    interpolated_amplitudes = [interpolate(y) for y in validation_pts]
    abs_differences = [abs(x-y) for x,y in zip(interpolated_amplitudes, true_amplitudes)]
    weights = [phase_space_weight(*x, domain = 'validation') for x in validation_pts]

    num, den = 0, 0
    for abs_error, weight, true_amp, in zip(abs_differences, weights, true_amplitudes):
        num+=np.abs(abs_error*weight)
        den+=np.abs(true_amp*weight)
    
    print(f'Error = {num/den}')
    
    fig = plt.figure(figsize=(15, 9))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    ax1.scatter(validation_pts[:,0], validation_pts[:,1], true_amplitudes, 
                color='tab:blue')
    ax1.set_title('True amplitude')
    ax2.scatter(validation_pts[:,0], validation_pts[:,1], interpolated_amplitudes, 
                color='tab:green')
    ax2.set_title('Interpolated amplitude')

    plt.show()