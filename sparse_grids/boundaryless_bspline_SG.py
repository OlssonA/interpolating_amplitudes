#!/usr/bin/env python3
import numpy as np
from copy import copy, deepcopy
from bisect import bisect_left
from datetime import datetime

class SG:
    def __init__(self, d, n, function, point_threshold, bspline_degree, 
                step_size, saved_grid = False, verbose = True,
                check_interpolated_points = False, 
                refinement_strategy = 'balanced', weight_function = None):
        self.verbose = verbose
        self.dim, self.n, self.p, self.max_refinements = \
            d, n, bspline_degree, step_size
        if verbose: print('Max refinements per step:', self.max_refinements)
        #Target function. Should accept one d-dimensional point and 
        #outputs one function value
        self.function = function 
        #verify that all data points are interpolated with tolerance 1e-10
        self.check_interpolated_points = check_interpolated_points
        #Greedy or balanced (volume-suppressed)
        self.refinement_strategy = refinement_strategy
        #Introduces a bias to the refinement criterion. 
        #Larger weight corresponds to 'worse' prediction. 
        #It can be seen as an error-enchancement factor
        #Should accept a list of points and output a list of weights
        self.weight_function = weight_function
        self.Lambda_e = int(np.ceil(np.log2(self.p+2)))
        self.point_threshold = point_threshold
        
        ######### Setup the grid, look for saved results ########
        if saved_grid:
            self.refinement_candidates = deepcopy(saved_grid.refinement_candidates)
            self.used_points = copy(saved_grid.used_points)
            self.levels, self.indices, self.surpluses, self.values = \
                copy(saved_grid.levels), copy(saved_grid.indices), \
                copy(saved_grid.surpluses), copy(saved_grid.values)
        else:
            #Points that will be further refined, ordered by increasing surplus
            self.refinement_candidates = {
                'points' : [[0.5]*self.dim], 
                'surpluses' : [self.function([0.5]*self.dim)],
                'levels': [[1]*self.dim],
                'indices' : [[1]*self.dim]}
            self.used_points = [[0.5]*self.dim]
            self.levels, self.indices, self.surpluses = \
                [[1]*self.dim], [[1]*self.dim], [self.function([0.5]*self.dim)]
            self.values = [self.surpluses[0]]

        self.adaptive_spatial_refinement()
    
    #Equidistant knot-vector for the polynomial construction. 
    #Not-a-knot construction for the B-splines (with auxilliary knot)
    def knot(self, k, l): 
        if l < self.Lambda_e:
            return k*2**(-l)
        else:
            k -= 1
            if 0 <= k <= self.p: return (k-self.p)*2**(-l)
            elif self.p+1 <= k <= 2**l-2: return (k-(self.p-1)/2)*2**(-l)
            elif 2**l-1 <= k <= 2**l+self.p-1: return (k+1)*2**(-l)

    #Polynomials are used if there are too few points for B-spline of degree p
    def Lagrange_poly_basis(self, x, l, k):
        return np.prod([(x - self.knot(m,l))/ \
                        (self.knot(k,l) - self.knot(m,l))
                        for m in range(1, 2**l) if m != k])

    #Cox de-Boor recursion formula
    def bspline_basis(self, x, l, i, p):
        if p == 0:
            if self.knot(i,l) <= x < self.knot(i+1,l): return 1
            else: return 0
        else:
            a = (x-self.knot(i, l))/(self.knot(i+p,l) - self.knot(i,l))
            b = (self.knot(i+p+1, l) - x)/(self.knot(i+p+1, l) - self.knot(i+1,l))
            return a*self.bspline_basis(x, l, i, p-1)+\
                    b*self.bspline_basis(x, l, i+1, p-1)
    
    #Choose between spline and polynomial depending on level and degree
    def boundaryless_nak_basis(self, x, l, i):
        if l >= self.Lambda_e:
            return self.bspline_basis(x, l, i, self.p)
        else: 
            return self.Lagrange_poly_basis(x, l, i)
    
    #Evaluate the interpolant, this is an inefficient implementation
    #It can be improved by considering the support of the splines
    def evaluate(self, x):
        result = 0
        for i in range(len(self.used_points)):
            basis_product = 1
            for d in range(self.dim):
                basis_product *= \
                    self.boundaryless_nak_basis(x[d],
                                            self.levels[i][d],
                                            self.indices[i][d])
            result += self.surpluses[i]*basis_product
        return result

    #Hierarchization (determining the interpolation coefficients)
    def adaptive_spatial_refinement(self):
        def add_point(point, level, index):
            #Check for and add any missing parents of the new point recursively
            for d in range(self.dim):
                if level[d] == 1: continue
                lower_level = level[d] - 1
                lower_index = int((index[d]+1)/2)
                if not np.mod(lower_index, 2): lower_index -= 1
                parent_point = copy(point)
                parent_point[d] = lower_index*2**(-lower_level)            
                if not parent_point in self.used_points:
                    parent_level = copy(level)
                    parent_index = copy(index)
                    parent_level[d] = lower_level
                    parent_index[d] = lower_index
                    add_point(parent_point, parent_level, parent_index)
            
            self.points_added += 1
            self.levels.append(level)
            self.indices.append(index)
            self.used_points.append(point)
            self.values.append(self.function(point))
        
        #Constrain the b-splines and return the interpolation coefficients
        def hierarchize():
            if self.verbose: 
                print('Current number of points:', len(self.used_points))
                print('Setting up interpolation matrix..')
            matrix = []
            for point in self.used_points:
                row = []
                for level, index in zip(self.levels, self.indices):
                    row.append(np.prod(
                        [self.boundaryless_nak_basis(
                    point[d], level[d], index[d]) for d in range(self.dim)]))
                matrix.append(row)
            if self.verbose: print('Solving linear system..')
            return np.linalg.solve(matrix, self.values)
        
        ######################################################

        ### Main loop, add points until point limit is reached
        while (len(self.used_points) < self.point_threshold):
            if len(self.refinement_candidates['points']) == 0: break
            
            ### Add points in batches and hierarchize in-between
            old_nr_points = len(self.used_points)
            nr_refinements = min(self.max_refinements, 
                                 len(self.refinement_candidates['points']))
            refinements, self.points_added = 0, 0
            while refinements < nr_refinements:
                if not self.refinement_candidates['points']:
                    if self.verbose: print('Ran out of refinement candidates')
                    break
                refinements += 1
                    
                #Last refinement candidate has the largest hierarchical surplus
                worst_point = self.refinement_candidates['points'].pop(-1)
                worst_level = self.refinement_candidates['levels'].pop(-1)
                worst_index = self.refinement_candidates['indices'].pop(-1)
                worst_surplus = self.refinement_candidates['surpluses'].pop(-1)

                #Sparse condition on the grid levels (1-norm is constrained)
                if sum(worst_level) == self.n + self.dim - 1: continue
                
                #Add the children points of one level deeper
                for d in range(self.dim):
                    refined_level = copy(worst_level)
                    refined_level[d] += 1
                    lower_index, upper_index = copy(worst_index), copy(worst_index)
                    lower_index[d] = 2*worst_index[d] - 1
                    upper_index[d] = 2*worst_index[d] + 1
                    for refined_index in [lower_index, upper_index]:
                        refined_point = copy(worst_point)
                        refined_point[d] = refined_index[d]*2**(-refined_level[d])
                        if not refined_point in self.used_points:
                            add_point(refined_point, refined_level, refined_index)
            
            self.surpluses = hierarchize()
                
            nr_added_points = len(self.surpluses) - old_nr_points
            for jj in range(1, nr_added_points+1):
                surplus = self.surpluses[-jj]
                level = self.levels[-jj]
                index = self.indices[-jj]
                point = self.used_points[-jj]

                if self.weight_function:
                    weight = self.weight_function(point)
                else:
                    weight = 1

                #Remember how well the current point was predicted
                #We refine the worst points first
                if self.refinement_strategy == 'greedy':
                        surplus_index = bisect_left(
                                        self.refinement_candidates['surpluses'], 
                                        weight*np.abs(surplus))
                elif self.refinement_strategy == 'balanced':
                    surplus_index = bisect_left(
                                        self.refinement_candidates['surpluses'], 
                                        weight/2**(sum(level))*np.abs(surplus))

                self.refinement_candidates['surpluses'].insert(surplus_index, 
                                                          np.abs(surplus))
                self.refinement_candidates['levels'].insert(surplus_index, 
                                                            level)
                self.refinement_candidates['points'].insert(surplus_index, 
                                                            point)
                self.refinement_candidates['indices'].insert(surplus_index, 
                                                             index)
                        
        self.points = len(self.used_points)
        if self.verbose:
            print('used points:', self.points)
            print('current time:', str(datetime.now()))
        self.refinement_candidates = self.refinement_candidates

        if self.check_interpolated_points:
            tol = 1e-10
            errors = [abs(val - self.evaluate(point)) for point, val in zip(self.used_points, self.values)]
            bad_indices = [i for i in range(len(errors)) if errors[i] > tol]
            if not len(bad_indices) == 0:
                input(f'{len(bad_indices)} bad points..')
            else:
                print('All data points were interpolated within', tol, 'tolerance')