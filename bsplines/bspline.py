#!/usr/bin/env python3

import numpy as np
import scipy as sp
from bisect import bisect_left

class Bspline:
    def __init__(self, p, n, d, function, uniform_knots,
                 check_interpolated_values = False, verbose = True):
        self.n, self.degrees, self.dim = np.array(n), np.array(p), d
        #Target function. Should accept a list of d-dimensional points and 
        #output a list of function values
        self.function = function
        #Whether to use uniform or not-a-knot knot vectors. True means uniform,
        #False means not-a-knot.
        self.uniform_knots = uniform_knots
        #verify that all data points are interpolated with tolerance 1e-10
        self.verbose = verbose

        if self.verbose:
            print('Computing function values..')
        
        self.grid_lines = [np.linspace(
                0.0001, 1-0.0001, self.n[dim] + 1) for dim in range(self.dim)]
        self.grid = np.vstack(np.meshgrid(
            *tuple(self.grid_lines), indexing='ij')).reshape(self.dim,-1).T
        self.data_values = self.function(self.grid)
        self.points = len(self.data_values)
        if self.verbose:
            print('Done')

        if uniform_knots: self.generate_uniform_knot_vectors()
        else: self.generate_not_a_knot_vectors()
        if self.verbose:
            if uniform_knots: print('Knots are uniform')
            else: print('Not-a-knot construction')
        
        if self.verbose:
            print(f'Constructing bspline of degree {self.degrees[0]}, n={n[0]}..')
        self.compute_control_points()

        if check_interpolated_values: self.check_valid_interpolation()

    #Uniform knot-vector (equidistant knots)
    def generate_uniform_knot_vectors(self):
        self.knot_vectors, self.knot_distances = [], []
        for d in range(self.dim):
            a = 0 - 1e-5
            b = 1 + 1e-5
            p = self.degrees[d]
            m = self.n[d] + p + 1
            low = a - (b-a)*p/(m-2*p)
            high = b + (b-a)*p/(m-2*p)
            self.knot_vectors.append(np.linspace(low, high, m+1))
            self.knot_distances.append(
                self.knot_vectors[-1][1] - self.knot_vectors[-1][0])
    
    #Not-a-knot knot vector (non-equidistant knots)
    def generate_not_a_knot_vectors(self):
        self.knot_vectors, self.knot_distances = [], []
        self.grid_lines = []
        for d in range(self.dim):
            a = 0 - 1e-10
            b = 1
            p = self.degrees[d]
            m = self.n[d] + p + 1
            low = a - (b-a)*p/(m-2*p) #extend knots below 0
            high = b + (b-a)*p/(m-2*p) #extend knots above 1
            exterior_knots = p # on each side (not including boundary)
            interior_knots = m+1 - exterior_knots*2
            exterior_low = list(np.linspace(low, a, exterior_knots+1))
            interior_knots = list(np.linspace(a,b,interior_knots + p-1))
            exterior_high = list(np.linspace(b, high, exterior_knots+1))
            #these knots are included in 'interior knots' already
            del exterior_low[self.degrees[d]]
            del exterior_high[0]
            if p == 2: #even case is not well defined with not-a-knot condition
                del interior_knots[bisect_left(interior_knots, 0.5+1e-5)-1]
            elif p == 3:
                del interior_knots[1] #remove the innermost inner knots
                del interior_knots[-2]
            knots = exterior_low+interior_knots+exterior_high
            self.knot_vectors.append(knots)
        self.knot_vectors = np.asarray(self.knot_vectors)

    #Calculate the control points, the equations simplify for equidistant
    #knot-vectors (not not-a-knot)
    def compute_basis_functions(self, u, dim):
        if self.uniform_knots:
            knot_distance = self.knot_distances[dim]
            knot_vector = self.knot_vectors[dim]
            p = self.degrees[dim]
            non_zero_basis_functions = {}
            k = bisect_left(knot_vector, u) - 1
            non_zero_basis_functions[k] = 1
            for d in range(1,p+1):
                non_zero_basis_functions[k-d] = (knot_vector[k+1] - u)/(knot_distance*d) * non_zero_basis_functions[k-d+1]
                for i in range(k-d+1, k):
                    non_zero_basis_functions[i] = (u-knot_vector[i])/(knot_distance*d) * non_zero_basis_functions[i] +\
                        (knot_vector[i+d+1] - u)/(knot_distance*d) * non_zero_basis_functions[i+1]
                non_zero_basis_functions[k] = (u - knot_vector[k])/(knot_distance*d) * non_zero_basis_functions[k]
            return list(non_zero_basis_functions.keys()), list(non_zero_basis_functions.values())
        else:
            knot_vector = self.knot_vectors[dim]
            p = self.degrees[dim]
            non_zero_basis_functions = {}
            k = bisect_left(knot_vector, u) - 1
            non_zero_basis_functions[k] = 1
            for d in range(1,p+1):
                non_zero_basis_functions[k-d] = (knot_vector[k+1] - u)/(knot_vector[k+1] - knot_vector[k-d+1]) * non_zero_basis_functions[k-d+1]
                for i in range(k-d+1, k):
                    non_zero_basis_functions[i] = (u-knot_vector[i])/(knot_vector[i+d]-knot_vector[i]) * non_zero_basis_functions[i] +\
                        (knot_vector[i+d+1] - u)/(knot_vector[i+d+1] - knot_vector[i+1]) * non_zero_basis_functions[i+1]
                non_zero_basis_functions[k] = (u - knot_vector[k])/(knot_vector[k+d]-knot_vector[k]) * non_zero_basis_functions[k]
            return list(non_zero_basis_functions.keys()), list(non_zero_basis_functions.values())
        
    #Map a self.dim coordinate to an index
    def index_calculator(self, indices):
        return np.sum(
            [indices[d]*np.prod(self.n[:d]+1) for d in range(self.dim)])
    
    #Construct the interpolant
    def compute_control_points(self):
        def find_constraints(point):
            basis_indices, basis_coefficients = [], []
            for dim in range(self.dim):
                indices, coeffs = self.compute_basis_functions(point[dim], dim)
                basis_indices.append(indices)
                basis_coefficients.append(coeffs)
            index_set = np.vstack(
                np.meshgrid(*tuple(basis_indices))).reshape(self.dim,-1).T
            coeff_set = np.vstack(
                np.meshgrid(*tuple(basis_coefficients))).reshape(self.dim,-1).T 
            return index_set, coeff_set
        
        coefficient_array, control_point_index_array, data_index_array = \
            [], [], []
        for i, point in enumerate(self.grid):
            index_set, coefficients = find_constraints(point)
            for indices, coefficient_grid_point in zip(index_set, coefficients):
                coefficient_array.append(np.prod(coefficient_grid_point))
                control_point_index_array.append(self.index_calculator(indices))
                data_index_array.append(i)
        basis_matrix = sp.sparse.csc_matrix(
            (coefficient_array, (data_index_array, control_point_index_array)), 
            shape=[len(self.data_values), len(self.data_values)])
        # from scipy.sparse.linalg import spsolve
        # self.control_points = spsolve(basis_matrix, self.data_values)
        from scipy.optimize import lsq_linear
        self.control_points = lsq_linear(basis_matrix, self.data_values, verbose=0).x
        if self.verbose:
            print(f'Created B-spline with {len(self.control_points)} points')
    
    #Evaluate the B-spline at any point inside of the parameter space
    def evaluate_point(self, point):
        basis_indices, basis_coefficients = [], []
        for dim in range(self.dim):
            indices, coeffs = self.compute_basis_functions(point[dim], dim)
            basis_indices.append(indices)
            basis_coefficients.append(coeffs)
        index_grid = np.vstack(
            np.meshgrid(*tuple(basis_indices))).reshape(self.dim,-1).T
        coeff_grid = np.vstack(
            np.meshgrid(*tuple(basis_coefficients))).reshape(self.dim,-1).T
        result = 0
        for indices, coefficient_grid_point in zip(index_grid, coeff_grid):
            result += np.prod(coefficient_grid_point)*\
                    self.control_points[self.index_calculator(indices)]
        return result
    
    def check_valid_interpolation(self):
        tol = 1e-10
        errors = [abs(val - self.evaluate_point(point)) for point, val in 
                    zip(self.grid, self.data_values)]
        bad_indices = [i for i in range(len(errors)) if errors[i] > tol]
        if not len(bad_indices) == 0:
            print(len(bad_indices), 'data points were not interpolated '
                    'within', tol)
        else:
            print('All data points were interpolated within', tol, 'tolerance')