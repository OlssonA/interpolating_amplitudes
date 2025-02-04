import numpy as np

def rescale(x, bounds):
    return [(bounds[jj][1]-bounds[jj][0])*point + bounds[jj][0] 
            for jj, point in enumerate(x)]

def from_unit_cube(point, scale):
    return [(scale[jj][1]-scale[jj][0])*point_1D + scale[jj][0] 
            for jj, point_1D in enumerate(point)]

def to_unit_cube(point, scale):
    return [(point_1D-scale[jj][0])/(scale[jj][1]-scale[jj][0]) 
            for jj, point_1D in enumerate(point)]

def y_to_x_ttH(y, test_function):
    y_pi = rescale(y, [[0.1, 0.96], [0, 1], [0, np.pi], [0, np.pi], [0, 2*np.pi]])
    if y_pi[4] > np.pi: y_pi[4] = 2*np.pi - y_pi[4]
    if test_function == 'f2':
        if y_pi[3] > np.pi/2:
            y_pi[2] = np.pi-y_pi[2]
            y_pi[3] = np.pi-y_pi[3]
    else:
        if y_pi[2] > np.pi/2:
            y_pi[2] = np.pi - y_pi[2]
            y_pi[4] = y_pi[4] + np.pi
            if y_pi[4] > np.pi: y_pi[4] = 2*np.pi - y_pi[4]
        if y_pi[3] > np.pi/2:
            y_pi[3] = np.pi - y_pi[3]
            y_pi[4] = y_pi[4] + np.pi
            if y_pi[4] > np.pi: y_pi[4] = 2*np.pi - y_pi[4]
    if test_function == 'f2':
        x = to_unit_cube(y_pi, [[0.1, 0.96], [0, 1], [0, np.pi], [0, np.pi/2], [0, np.pi]])
    else:
        x = to_unit_cube(y_pi, [[0.1, 0.96], [0, 1], [0, np.pi/2], [0, np.pi/2], [0, np.pi]])
    return x

def y_to_x_gggH(y):
    y_pi = rescale(y, [[0.33, 0.99], [0, np.pi]])
    if y_pi[1] > np.pi/2:
        y_pi[1] = np.pi - y_pi[1]
    x = to_unit_cube(y_pi, [[0.33, 0.99], [0, np.pi/2]])
    return x