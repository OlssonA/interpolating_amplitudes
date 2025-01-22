#!/usr/bin/env python3

import numpy as np
from copy import copy, deepcopy
from bisect import bisect_left
import ast
from datetime import datetime

class SG:
    def __init__(self, d, n, function, point_threshold,
                saved_grid = None, refinement_strategy = 'greedy',
                modified_basis = True, verbose = True,
                check_interpolated_points = False, weight_function = None):
        #Modified basis extrapolates linearly towards boundary 
        self.modified_basis = modified_basis
        self.verbose = verbose
        self.dim, self.n = d, n #n = maximum grid level
        #Target function. Should accept a list of points and 
        #output a list of function values
        self.function = function 
        #verify that all data points are interpolated with tolerance 1e-10
        self.check_interpolated_points = check_interpolated_points
        self.point_threshold = point_threshold
        #Greedy or balanced (volume-suppressed)
        self.refinement_strategy = refinement_strategy
        #Introduces a bias to the refinement criterion. 
        #Larger weight corresponds to 'worse' prediction. 
        #It can be seen as an error-enchancement factor
        #Should accept a list of points and output a list of weights
        self.weight_function = weight_function
        
        if self.verbose:
            print(f'Training level {n} adaptive sparse grid')
            print(f'Refinement strategy is: {refinement_strategy}')
            print(f'The point threshold is {int(point_threshold)} points')
            print('current time:', str(datetime.now()))

        if saved_grid:
            if self.verbose:
                print(f"Found saved grid with "
                      f"{len(saved_grid.used_points)} points")
                print(f"Importing "
                      f"{len(saved_grid.refinement_candidates['points'])}"\
                    +" refinement candidates from the previous grid")
                print(f"Adding "
                      f"{int(self.point_threshold)-len(saved_grid.used_points)}"
                        " more points to the existing grid")
            self.refinement_candidates = deepcopy(
                saved_grid.refinement_candidates)
            self.used_points = copy(saved_grid.used_points)
            self.level_to_indices = deepcopy(saved_grid.level_to_indices)
            self.values = copy(saved_grid.values)
        else:
            #Points that will be further refined, ordered by increasing surplus
            self.refinement_candidates = {
                'points' : [[0.5]*self.dim], 
                'surpluses' : [self.function([[0.5]*self.dim])[0]],
                'levels': [[1]*self.dim],
                'indices' : [[1]*self.dim]
                }
            #Start in centre
            self.used_points = [[0.5]*self.dim]
            self.values = [self.refinement_candidates['surpluses'][0]]

            #The currently added sparse grid points, organised by grid level
            self.level_to_indices = {
                f"{[1]*self.dim}" : {
                    'indices' : [[1]*self.dim],
                    'surpluses' : [self.refinement_candidates['surpluses'][0]]
                    }}

        self.adaptive_spatial_refinement()

    def evaluate_basis_function(self, x, l, i):
        if self.modified_basis: #Extrapolating basis
            if l == 1 and i == 1: return 1
            elif l > 1 and i == 1:
                if 0 <= x <= 2**(1-l): return 2-x*2**l
                else: return 0
            elif l > 1 and i == 2**l-1:
                if 1-2**(1-l) <= x <= 1: return x*2**l+1-i
                else: return 0
            else: return self.hat_function(x*2**l-i)
        
        else: 
            return self.hat_function(x*2**l-i) #Do not extrapolate
    
    #Simplest possible basis function (aka linear B-spline)
    def hat_function(self, x):
        if x < -1 or x > 1: return 0
        else: return 1 - np.abs(x)

    #Evaluate the interpolant at a self.dim dimensional grid point x
    def evaluate(self, x):
        #Find indices of non-zero basis functions contributing 
        #to the interpolant at x
        def contributing_indices(x, level):
            indices = []
            for d in range(self.dim):
                if x[d] > 1-2**(-level[d]):
                    indices.append(2**level[d]-1)
                elif x[d] < 2**(-level[d]):
                    indices.append(1)
                else:
                    index = 2**level[d]*(x[d]-(np.mod(x[d],2**(-level[d]))))
                    if np.mod(index,2) == 0: indices.append(index+1)
                    else: indices.append(index)
            return indices
        
        result = 0
        for level, index_surplus in self.level_to_indices.items():
            level = ast.literal_eval(level)
            I_l = contributing_indices(x, level)
            if not I_l in index_surplus['indices']: continue
            basis_functions = [self.evaluate_basis_function(
                x[d], level[d], I_l[d]) for d in range(self.dim)]
            result += index_surplus['surpluses'][
                    index_surplus['indices'].index(I_l)]*\
                        np.prod(basis_functions)
        return result

    def adaptive_spatial_refinement(self):
        #Check for and add recursively missing parents of new point 
        def add_point(point, level, index):
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
            #Now this point has all its parents in the grid, we can add it
            self.points_to_add.append(point)
            self.levels_to_add.append(level)
            self.index_to_add.append(index)
            self.used_points.append(point)
            
        while (len(self.used_points) < self.point_threshold) and \
                self.refinement_candidates['points']:
            #Last refinement candidate was the 'worst' predicted point
            worst_point = self.refinement_candidates['points'].pop(-1)
            worst_level = self.refinement_candidates['levels'].pop(-1)
            worst_index = self.refinement_candidates['indices'].pop(-1)
            self.refinement_candidates['surpluses'].pop(-1)

            #Sparse condition on the grid levels (1-norm is constrained)
            #In spatially-adaptive SGs this condition is a bit unncessary
            if sum(worst_level) == self.n + self.dim - 1: continue
            
            #Refine a grid point
            self.points_to_add, self.levels_to_add, self.index_to_add = [],[],[]
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
            
            # if self.verbose and len(self.points_to_add) == 0: 
            #     print('WARNING: DID NOT FIND NEW POINTS')
            if len(self.points_to_add) > 0:
                new_values = self.function(self.points_to_add)
                if self.weight_function == None: 
                    weights = [1]*len(self.points_to_add)
                else:
                    weights = self.weight_function(self.points_to_add)
                for point, level, index, new_val, weight in zip(
                                                        self.points_to_add, 
                                                        self.levels_to_add, 
                                                        self.index_to_add,
                                                        new_values,
                                                        weights):
                    
                    surplus = new_val - self.evaluate(point)
                    self.values.append(new_val)
                    if not str(level) in self.level_to_indices:
                        self.level_to_indices[str(level)] = {'indices':[index], 
                                                        'surpluses' : [surplus]}
                    else:
                        self.level_to_indices[str(level)]['indices'].append(
                            index)
                        self.level_to_indices[str(level)]['surpluses'].append(
                            surplus)

                    #Remember how well the current point was predicted
                    #We refine the 'worst' points first.
                    if self.refinement_strategy == 'greedy':
                        surplus_index = bisect_left(
                            self.refinement_candidates['surpluses'], 
                            weight*np.abs(surplus))
                    elif self.refinement_strategy == 'balanced':
                        surplus_index = bisect_left(
                            self.refinement_candidates['surpluses'],
                            weight/2**(sum(level))*np.abs(surplus))
                    
                    self.refinement_candidates['surpluses'].insert(
                                                            surplus_index,
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

        if self.check_interpolated_points:
            tol = 1e-10
            errors = [abs(val - self.evaluate(point)) for point, val in 
                      zip(self.used_points, self.values)]
            bad_indices = [i for i in range(len(errors)) if errors[i] > tol]
            if not len(bad_indices) == 0:
                print(len(bad_indices), 'data points were not interpolated '
                      'within', tol, 'tolerance')
            else:
                print('All data points were interpolated within', tol, 'tolerance')