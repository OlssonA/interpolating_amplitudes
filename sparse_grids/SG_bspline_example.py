#!/usr/bin/env python3
from extended_bspline_SG import SG as extended_bspline_SG
from boundaryless_bspline_SG import SG as boundaryless_bspline_SG
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    #Constructs two SGs, one with extended B-spline basis and one with
    #boundaryless B-spline basis. These are expected to perform similarly.

    sparse_level = 40
    point_threshold = 100
    nr_validation_pts = 250
    dim = 2
    p = 1 #select between 1,3,5
    step_size = 5
    refinement_strategy = 'balanced'
    
    def target_function(point):
        return point[0]**2 + np.exp(point[1]-0.5)
        

    extended_SG = extended_bspline_SG(
                    d = dim, n = sparse_level, function = target_function,
                    point_threshold = point_threshold, saved_grid = None,
                    verbose = False, bspline_degree = p, step_size = step_size,
                    check_interpolated_points=True,
                    refinement_strategy = refinement_strategy)
    
    boundaryless_SG = boundaryless_bspline_SG(
                    d = dim, n = sparse_level, function = target_function,
                    point_threshold = point_threshold, saved_grid = None,
                    verbose = False, bspline_degree = p, step_size = step_size,
                    check_interpolated_points=True,
                    refinement_strategy = refinement_strategy)

    validation_pts = np.random.rand(nr_validation_pts, 2)
    true_values = [target_function(x) for x in validation_pts]
    extended_SG_values = [extended_SG.evaluate(x) for x in validation_pts]
    boundaryless_SG_values = \
        [boundaryless_SG.evaluate(x) for x in validation_pts]
    extended_abs_errors = \
        [x-y for x,y in zip(extended_SG_values, true_values)]
    boundaryless_abs_errors = \
        [x-y for x,y in zip(boundaryless_SG_values, true_values)]
    
    fig = plt.figure(figsize=(15, 9))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    ax3 = fig.add_subplot(1,3,3, projection='3d')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    ax1.scatter(validation_pts[:,0], validation_pts[:,1], 
                true_values, color='tab:blue')
    ax1.set_title('True Values')

    ax2.scatter(validation_pts[:,0], validation_pts[:,1], 
                extended_SG_values, color='tab:green')
    ax2.set_title('Approx: Extended B-spline SG')
    
    ax3.scatter(validation_pts[:,0], validation_pts[:,1], 
                boundaryless_SG_values, color='tab:orange')
    ax3.set_title('Approx: Boundaryless B-spline SG')
    
    print('Extended B-spline 2-norm error:', 
          np.linalg.norm(extended_abs_errors, 2))
    
    print('Boundaryless B-spline 2-norm error:', 
          np.linalg.norm(boundaryless_abs_errors, 2))

    plt.show()