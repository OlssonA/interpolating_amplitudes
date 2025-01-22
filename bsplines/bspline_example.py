#!/usr/bin/env python3

from bspline import Bspline
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    n = 5
    nr_validation_pts = 800
    dim = 2
    uniform_knots = 0
    p = 2

    def target_function(points):
        vals = []
        for p in points:
            vals.append(p[0]**2 + np.exp(p[1]-0.5))
        return vals

    bspline = Bspline([p]*dim, [n]*dim, dim, target_function, uniform_knots,
                    check_interpolated_values = True, verbose = False)

    validation_pts = np.random.rand(nr_validation_pts, dim)
    true_values = target_function(validation_pts)
    interpolated_values = [bspline.evaluate_point(x) for x in validation_pts]
    abs_differences = [x-y for x,y in zip(interpolated_values, true_values)]

    print(f'2-norm error: (n={n})', np.linalg.norm(abs_differences, 2))
    
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

    ax1.scatter(validation_pts[:,0], validation_pts[:,1], true_values, 
                color='tab:blue')
    ax1.set_title('True values')
    ax2.scatter(validation_pts[:,0], validation_pts[:,1], interpolated_values, 
                color='tab:green')
    ax2.set_title('Approximated values')

    plt.show()