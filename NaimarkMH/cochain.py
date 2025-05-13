#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:28:57 2024

Last edited on: May 13 15:43 2025

Author: Afonso Barroso, The University of Manchester
"""


# ----- # ----- #  IMPORTS # ----- # ----- #

import numpy as np
from math import factorial
from collections import defaultdict
import sys
sys.path.append('./')
sys.path.append('../Voronoi_PCC_Analyser')
from matgen.base import Vertex3D, Edge3D, Face3D, Poly, CellComplex
from functools import lru_cache


# ----- # ----- #  CLASSES # ----- # ----- #

class Cochain():
    
    def __init__(self,
                 compl: CellComplex,
                 cochain_dim: int,
                 cochain: np.ndarray):
        
        self._cochain_dim = cochain_dim
        self._values = cochain[:,1:cochain.shape[1]+1].tolist()
        
        if compl is not None:
            chain = compl.get_many(cochain_dim, cochain[:,0].tolist())
            self._chain = chain
            self.structure = compl.structure
            self.is_empty = False
            self.ip = compl.ip
            self._cochain = {chain[i] : cochain[i,1:] for i in range(len(chain))}
            self._cochainarray = cochain
            self.dim = compl.dim
            self.cell_complex = compl
            
        else:
            self._chain = []
            self.structure = None
            self.is_empty = True

                    
    # -- #
    def __repr__(self):
        return self.__str__()
    
    # -- #
    def __str__(self):
        
        try:
            c = self._cochain
            s = f"<class {self.__class__.__name__}>"
            size = len(c.keys())
            if 1 <= size <= 8:
                for i in range(size):
                    j = list(c.keys())[i]
                    if i == 0:
                        s += "\n{" + str(j) + " : " + str(c[j]) + ",\n"
                    elif 0 < i < (size - 1):
                        s += " " + str(j) + " : " + str(c[j]) + ",\n"
                    else:
                        s += " " + str(j) + " : " + str(c[j]) + "}"
            else:
                for i in range(9):
                    if i == 0:
                        j = list(c.keys())[i]
                        s += "\n{" + str(j) + " : " + str(c[j]) + ",\n"
                    elif 1 <= i <= 3:
                        j = list(c.keys())[i]
                        s += " " + str(j) + " : " + str(c[j]) + ",\n"
                    elif i == 4:
                        s += " ...\n"
                    elif 5 <= i <= 7:
                        j = list(c.keys())[i - 9]
                        s += " " + str(j) + " : " + str(c[j]) + ",\n"
                    elif i == 8:
                        j = list(c.keys())[-1]
                        s += " " + str(j) + " : " + str(c[j]) + "}"
        except:
            s = f"<class {self.__class__.__name__}>"
            s += "\nEmpty"
        return s
    
    # -- #
    def __add__(self, other):
        
        if ((self.is_empty == False and other.is_empty == False) and
            (self._cochain_dim != other._cochain_dim or self.dim != other.dim)):
            raise NotImplementedError
            
        elif (self.is_empty == True and other.is_empty == True):
            return Cochain.empty()
        
        try:
            cc = self.cell_complex
        except:
            raise AttributeError("\nThere is no cell complex attributed to the first Cochain. Try swapping the order of addition.")
                
        # Dictionary to store the sum of vectors for each cell index
        cell_dict = defaultdict(lambda: np.zeros((1,3)))
        
        for row in np.vstack((self._cochainarray, other._cochainarray)): # Combine the two cochains
            cell_index = row[0]
            vector = row[1:]
            cell_dict[cell_index] += vector
            
        # Convert the dictionary back to a numpy array
        new_cochain = np.array([[int(key), *value[0]] for key, value in cell_dict.items()])
        
        return Cochain(compl = cc,
                       cochain_dim = self._cochain_dim,
                       cochain = new_cochain)
                
    
    # -- #
    def __sub__(self, other):
        
        if ((self.is_empty == False and other.is_empty == False) and
            (self._cochain_dim != other._cochain_dim or self.dim != other.dim)):
            raise NotImplementedError
            
        elif (self.is_empty == True and other.is_empty == True):
            return Cochain.empty()
        
        try:
            cc = self.cell_complex
        except:
            raise AttributeError("\nThere is no cell complex attributed to the first Cochain. Try swapping the order of addition.")
                
        # Dictionary to store the sum of vectors for each cell index
        cell_dict = defaultdict(lambda: np.zeros((1,3)))
        
        for row in np.vstack((self._cochainarray, other._cochainarray)): # Combine the two cochains
            cell_index = row[0]
            vector = row[1:]
            cell_dict[cell_index] -= vector
            
        # Convert the dictionary back to a numpy array
        new_cochain = np.array([[int(key), *value[0]] for key, value in cell_dict.items()])
        
        return Cochain(compl = cc,
                       cochain_dim = self._cochain_dim,
                       cochain = new_cochain)
    
    # -- #
    def __mul__(self, value: int | float):
        
        if self.is_empty == True:
            return Cochain.empty()
        
        L = len(self)
        if self._cochainarray[:,0].ndim == 1:
            return Cochain(self.cell_complex,
                           self._cochain_dim,
                           np.hstack((np.array([self._cochainarray[:,0]]).reshape((L,1)), self._cochainarray[:,1:] * value)))
        else:
            return Cochain(self.cell_complex,
                           self._cochain_dim,
                           np.hstack((np.array(self._cochainarray[:,0]).reshape((L,1)), self._cochainarray[:,1:] * value)))
    
    # -- #
    def __rmul__(self, value: int | float):
        
        if self.is_empty == True:
            return Cochain.empty()
        
        L = len(self)
        if self._cochainarray[:,0].ndim == 1:
            return Cochain(self.cell_complex,
                           self._cochain_dim,
                           np.hstack((np.array([self._cochainarray[:,0]]).reshape((L,1)), self._cochainarray[:,1:] * value)))
        else:
            return Cochain(self.cell_complex,
                           self._cochain_dim,
                           np.hstack((np.array(self._cochainarray[:,0]).reshape((L,1)), self._cochainarray[:,1:] * value)))

    # -- #
    def __neg__(self):
        
        if self.is_empty == True:
            return Cochain.empty()
        
        L = len(self)
        return Cochain(self.cell_complex,
                       self._cochain_dim,
                       np.hstack((self._cochainarray[:,0].reshape((L,1)), self._cochainarray[:,1:] * (-1))))
    
    # -- #
    def __eq__(self, other) -> bool:
        
        cond1 = (self._cochain_dim == other._cochain_dim)
        cond2 = (self.dim == other.dim)
        cond3 = (self.structure == other.structure)
        try:
            cond4 = (np.allclose(self._cochainarray, other._cochainarray))
        except:
            return False
        return (cond1 and cond2 and cond3 and cond4)
        
    # -- #
    def __len__(self):
        return self._cochainarray.shape[0]
    
    # -- #
    @property
    def show_all(self):
        
        c = self._cochain
        s = f"<class {self.__class__.__name__}>"
        size = len(c.keys())
        if size == 0:
            s += "\nEmpty"
        else:
            for i in range(size):
                j = list(c.keys())[i]
                if i == 0:
                    s += "\n{" + str(j) + " : " + str(c[j]) + ",\n"
                elif 0 < i < (size - 1):
                    s += " " + str(j) + " : " + str(c[j]) + ",\n"
                else:
                    s += " " + str(j) + " : " + str(c[j]) + "}"
        print(s)
        
    # -- #
    def sort(self) -> float:
        
        k = [] ; v = [] ; cochain = self._cochain
        for i in cochain.keys():
            k.append(i.id)
        v = list(cochain.values())
        order = np.argsort(k)
        k = np.array(k)[order]
        v = np.array(v)[order]
        return Cochain(compl = self.cell_complex,
                       cochain_dim = self._cochain_dim,
                       cochain = np.hstack((k.reshape((len(k),1)), v)))
            
    # -- #
    def return_cellcomplex(self) -> CellComplex:
        
        try:
            return self.cell_complex
        except:
            raise AttributeError(f"\nThere is no cell complex attributed to the Cochain\n{self}")
        
    # -- #
    def empty():
        """
        Returns
        -------
        Cochain
            An empty Cochain instance.
            
        Notes
        -----
        By the __init__ method, an empty cochain has .dim = -1 and .cochain_dim = None.        
        """
        
        c = Cochain(compl = None, cochain_dim = 0, cochain=np.array([[1,0]]))
        return c
    
    # -- #
    def clean(self):
                
        to_delete = []
        for row_idx in range(len(self._cochainarray)):
            v = self._cochainarray[row_idx]
            if np.linalg.norm(v[1:]) == 0:
                to_delete.append(v[0])
                del self._chain[row_idx]
        self._cochainarray = np.delete(self._cochainarray, to_delete, axis=0)
        self._values = self._cochainarray[:,1:].tolist()            
                
    # -- #
    @property
    def cochain_dim(self) -> int:
        return self._cochain_dim
    
    # -- #
    @property
    def cochain(self) -> dict:
        return self._cochain
    
    # -- #
    @property
    def chain(self) -> list:
        return self._chain
    
    # -- #
    @property
    def values(self) -> list:
        return self._values.tolist()
    
    # -- #
    def inner_product_2023(self, other = None) -> float:
        """
        Returns
        -------
        float
            The inner product of p-cochains (p in {0,1,2,3}) as defined by K. Berbatov in his PhD Thesis (2023).
            The inner product of two arbitrary cochains of order p is given by
            coef * Sum_over_p-cells (1/measure_of_p-cell * values of the two cochains * Sum (measures of the (d-p)-cells orthogonal to the p-cell))
        """
        
        cochain_dim = self._cochain_dim
        
        # Check that the cochains are of the same order
        if other is None:
            other = self
        elif (cochain_dim != other._cochain_dim):
            raise AttributeError('\nThe method .inner_product_2023() can only be performed between cochains of the same dimension.')
        
        # Set the coefficient to the appropriate cell complex type
        if 'simplicial' in self.structure:
            coef = 1 / (self.dim + 1)
        elif 'quasi-cubical' in self.structure:
            coef = 1 / (2 ** self.dim)
        
        # Define local variables to reduce number of method and attribute calls
        chain = list(set(self._chain) & set(other._chain)) # select the cells on which both cochains are defined (intersection)
        c1 = self._cochain
        c2 = other._cochain
        get_many = self.cell_complex.get_many
        get_one = self.cell_complex.get_one
        sum1 = 0
        
        match cochain_dim:
            case 3:
                for c in chain:
                    # Find the set of orthogonal vertices in the case of 3-cochains
                    ortho = get_many('v', c.v_ids)
                    # Add up the measures of the orthogonal (d-p)-cells (d=3)
                    sum2 = sum([o.measure for o in ortho])
                    # Compute the sum over all p-cells where both cochains take values
                    sum1 += (1/c.measure) * np.dot(c1[c], c2[c]) * sum2
            
            case 2:
                for c in chain:
                    # Find the set of orthogonal edges in the case of 2-cochains
                    polys = get_many('p', c.incident_ids)
                    ortho = []
                    for p in polys:
                        for e in p.e_ids:
                            e = get_one('e', e)
                            if cell_orthogonality_3D(c, e) == 'Orthogonal':
                                ortho.append(e)
                            else: pass
                    # Add up the measures of the orthogonal (d-p)-cells (d=3)
                    sum2 = sum([o.measure for o in ortho])
                    # Compute the sum over all p-cells where both cochains take values
                    sum1 += (1/c.measure) * np.dot(c1[c], c2[c]) * sum2
            
            case 1:
                for c in chain:
                    # Find the set of orthogonal faces in the case of 1-cochains
                    polys = get_many('p', c.incident_polyhedra_ids)
                    ortho = []
                    for p in polys:
                        for f in p.f_ids:
                            f = get_one('f', f)
                            if cell_orthogonality_3D(c, f) == 'Orthogonal':
                                ortho.append(f)
                            else: pass
                    # Add up the measures of the orthogonal (d-p)-cells (d=3)
                    sum2 = sum([o.measure for o in ortho])
                    # Compute the sum over all p-cells where both cochains take values
                    sum1 += (1/c.measure) * np.dot(c1[c], c2[c]) * sum2
                
            case 0:
                for c in chain:
                    # Find the set of orthogonal polyhedra in the case of 0-cochains
                    ortho = get_many('p', c.incident_polyhedra_ids)
                    # Add up the measures of the orthogonal (d-p)-cells (d=3)
                    sum2 = sum([o.measure for o in ortho])
                    # Compute the sum over all p-cells where both cochains take values
                    sum1 += (1/c.measure) * np.dot(c1[c], c2[c]) * sum2
        
        return coef * sum1
                
    # -- #
    def inner_product_2022(self, other = None) -> float:
        """
        Returns
        -------
        float
            The inner product of p-cochains (p in {0,1,2,3}) as defined by K. Berbatov in his 2022 paper.
            The inner product of two arbitrary cochains of order p is given by
            coef * Sum_over_p-cells (1/(measure of p_cell ** 2) Sum_over_0-cells (measure of 0-cell * Sum_over_d-cells (measure of d-cell)))
        """
        
        cochain_dim = self._cochain_dim
        
        # Check that the cochains are of the same order
        if other is None:
            other = self
        elif (cochain_dim != other._cochain_dim):
            raise AttributeError('\nThe method .inner_product_2022() can only be performed between cochains of the same dimension.')
        
        # Set the coefficient to the appropriate cell complex type
        structure = self.structure
        if 'simplicial' in structure:
            coef = 1 / (self.dim + cochain_dim)
        elif 'quasi-cubical' in structure:
            coef = 1 / (2 ** self.dim)
        
        # Define local variables to reduce number of method and attribute calls
        chain = set(self._chain) & set(other._chain) # select the cells on which both cochains are defined (intersection)
        c1 = self._cochain
        c2 = other._cochain
        
        get_many = self.cell_complex.get_many
        sum1 = 0
        for cell in chain:
            sum1 += 1/(cell.measure ** 2) * np.dot(c1[cell], c2[cell]) * innerprod2022_coef(get_many, cell, cochain_dim)
            
        return coef * sum1
    
    # -- #
    def inner_product(self, other = None):
        
        if self.is_empty == True:
            return 0
        elif (other and self.ip != other.ip):
            raise AttributeError("\nWhen taking the .inner_product() between cochains, they both must be configured to the same inner product (2022 or 2023).")
        else:
            if self.ip == 2022:
                return self.inner_product_2022(other)
            elif self.ip == 2023:
                return self.inner_product_2023(other)
        
    # -- #
    def metric(self, other = None):
        
        # Check that the cochains are of the same order
        if other is None:
            other = self
        elif (self._cochain_dim != other._cochain_dim):
            raise AttributeError('\nThe method .metric() can only be performed between cochains of the same dimension.')
        
        # Define local variables to reduce number of method and attribute calls
        cochain_dim = self._cochain_dim
        get_one = self.cell_complex.get_many
        get_many = self.cell_complex.get_many
        selfarray = self._cochainarray
        otherarray = other._cochainarray
        result = np.empty((0, selfarray.shape[1]))
        
        if cochain_dim != 0:
            for row_idx in range(len(selfarray)):
                cell = selfarray[row_idx,0]
                if cell in otherarray[:,0]:
                    coef = 1 / ((get_one(cochain_dim, cell).measure ** 2) * (cochain_dim + 1))
                    for v in get_many(0, cell.v_ids):
                        value = v.measure * coef * np.dot(selfarray[row_idx-1, 1:], otherarray[row_idx-1, 1:])
                        result = np.vstack((result, np.array([v.id, *value])))
                else: pass
        
        else:
            for row_idx in range(len(selfarray)):
                v = selfarray[row_idx,0]
                if v in otherarray[:,0]:
                    vtx = get_one(0, v)
                    coef = 1 / (vtx.measure ** 2)
                    value = vtx.measure * coef * np.dot(selfarray[row_idx-1, 1:], otherarray[row_idx-1, 1:])
                    result = np.vstack((result, np.array([vtx.id, *value])))
                else: pass
        
        return Cochain(self.cell_complex, 0, result)
            
    # -- #
    def cup(self, other, return_test: bool = False):
        
        prod_dim = self._cochain_dim + other._cochain_dim
        
        if prod_dim > self.dim:
            return None
        if (self._cochain_dim == 0 and other._cochain_dim == 0):
            return Cochain.empty()
        
        cc = self.cell_complex
        c1 = self._cochain
        c2 = other._cochain
        
        if return_test == True:
            test_out = np.empty((1,4))
            
        if 'simplicial' in self.structure:
            coef = factorial(self._cochain_dim) * factorial(other._cochain_dim) / factorial(prod_dim + 1)
            
            cup_product = defaultdict(float)
            for cell1, value1 in c1.items():
                
                for cell2, value2 in c2.items():
                    # Because the Vertex3D class does not have the method .v_ids, we have to account for the cases where the cochains
                    # involved are 0-cochains - especially if they are both 0-cochains, in which case the cup product is null. In
                    # particular, in the case where only one of the cochains involved is a 0-cochain, the cup product is only non-zero
                    # if the nodes in that cochain are nodes of the cells in the other cochain.
                    
                    if (self._cochain_dim != 0 and other._cochain_dim != 0):
                        new_simplex = get_new_simplex(cell1.v_ids + cell2.v_ids)
                    elif (self._cochain_dim == 0 and other._cochain_dim != 0):
                        new_simplex = get_new_simplex([cell1.id] + cell2.v_ids)
                    elif (self._cochain_dim != 0 and other._cochain_dim == 0):
                        new_simplex = get_new_simplex(cell1.v_ids + [cell2.id])
                        
                    if new_simplex is not None:
                        target_cell = find_cell(cc, new_simplex)
                        if target_cell is not None:
                            if permutation_parity(new_simplex, target_cell.v_ids) == 0:
                                ori = +1
                            elif permutation_parity(new_simplex, target_cell.v_ids) == 1:
                                ori = -1
                            product = ori * value1 * value2
                            
                            cup_product[target_cell.id] += product
                                
                            if return_test == True:
                                test_out = np.vstack((test_out, np.array([[cell1.id, cell2.id, target_cell.id, product]])))
                                                            
                        else: pass
                    else: pass
                
            if return_test == True:
                test_out = np.delete(test_out, [0], axis=0) ; test_out = test_out.astype(float) ; test_out = test_out[test_out[:,2].argsort()] 
        
        elif 'quasi-cubical' in self.structure:
            coef = 1 / (2 ** prod_dim)
            raise NotImplementedError('\nThe .cup() method of the Cochain class has not been implemented for quasi-cubical complexes.')
        
        result = np.empty((0,2))
        for key, value in cup_product.items():
            value = value * coef
            result = np.vstack((result, np.array([key, *value]))) 
            cup_product = Cochain(compl = cc, cochain_dim = prod_dim, cochain = result)
        
        if return_test == True:
            return cup_product, test_out
        else:
            return cup_product
        
    # -- #
    def star_2023(self):
        """
        Computes the star operator of any cochain according to K. Berbatov's PhD thesis (2023). The star product of a general p-cochain in
        a d-complex takes the following value at particular (d-p)-cells:
            coef * 1/(inner product of the corresponding basis (d-p)-cochain) * Sum_over_orthogonal_p-cells{value of the p-cochain at orthogonal cell}
        The coef is equal to 1 over the number of vertices in a d-cell.
        
        Returns
        -------
        star : Cochain
        """
        
        cc = self.cell_complex
        cochain = self._cochain
        cochain_dim = self._cochain_dim
        star = []
        
        inner_product = Cochain.inner_product_2023
        get_many = self.cell_complex.get_many
            
        if cochain_dim == 3:
            for p in self._chain:
                for v in get_many('v', p.v_ids):
                    i = v.id
                    v = Cochain(compl = cc,
                                cochain_dim = 0,
                                cochain = np.array([[i, 1]]))
                    if 'simplicial' in self.structure:
                        value = cochain[p] / ((self.dim + 1) * inner_product(v))
                    elif 'quasi-cubical' in self.structure:
                        value = cochain[p] / ((2 ** self.dim) * inner_product(v))
                    star.append([i, *value])

            star = Cochain(cc, 0, np.array(star))
                    
        elif cochain_dim == 2:
            for f in self._chain:
                for p in get_many('p', f.incident_ids):
                    for e in get_many('e', p.e_ids):
                        if cell_orthogonality_3D(f, e) == 'Orthogonal':
                            i = e.id
                            e = Cochain(compl = cc,
                                        cochain_dim = 1,
                                        cochain = np.array([[i, 1]]))
                            if 'simplicial' in self.structure:
                                value = cochain[f] / ((self.dim + 1) * inner_product(e))
                            elif 'quasi-cubical' in self.structure:
                                value = cochain[f] / ((2 ** self.dim) * inner_product(e))
                            star.append([i, *value])

            star = Cochain(cc, 1, np.array(star))
                            
        elif cochain_dim == 1:
            for e in self._chain:
                for p in get_many('p', e.incident_polyhedra_ids):
                    for f in get_many('f', p.f_ids):
                        if cell_orthogonality_3D(e, f) == 'Orthogonal':
                            i = f.id
                            f = Cochain(compl = cc,
                                        cochain_dim = 2,
                                        cochain = np.array([[i, 1]]))
                            if 'simplicial' in self.structure:
                                value = cochain[e] / ((self.dim + 1) * inner_product(f))
                            elif 'quasi-cubical' in self.structure:
                                value = cochain[e] / ((2 ** self.dim) * inner_product(f))
                            star.append([i, *value])

            star = Cochain(cc, 2, np.array(star))
                            
        elif cochain_dim == 0:
            for v in self._chain:
                for p in get_many('p', v.incident_polyhedra_ids):
                    i = p.id
                    p = Cochain(compl = cc,
                                cochain_dim = 3,
                                cochain = np.array([[i, 1]]))
                    if 'simplicial' in self.structure:
                        value = cochain[v] / ((self.dim + 1) * inner_product(p))
                    elif 'quasi-cubical' in self.structure:
                        value = cochain[v] / ((2 ** self.dim) * inner_product(p))
                    star.append([i, *value])

            star = Cochain(cc, 3, np.array(star))
                
        return star
    
    # -- #
    def star_2022(self):
        """
        Computes the star operator of any cochain according to K. Berbatov's 2022 paper. The star product of a general p-cochain in
        a d-complex takes the following value at a general (d-p)-cell:
            {coef / inner_product((d-p)-cell)}
        
        Returns
        -------
        star : Cochain
        """
        
        cc = self.cell_complex
        cochain = self._cochain
        cochain_dim = self._cochain_dim
        star = defaultdict(float)
        
        inner_product = Cochain.inner_product_2022
        get_many = self.cell_complex.get_many
            
        if cochain_dim == 3:
            for p in self._chain:
                for v in get_many('v', p.v_ids):
                    i = v.id
                    v = Cochain(compl = cc,
                                cochain_dim = 0,
                                cochain = {i : 1})
                    if 'simplicial' in self.structure:
                        coef = factorial(self.dim - cochain_dim) * factorial(cochain_dim) / factorial(self.dim + 1)
                    elif 'quasi-cubical' in self.structure:
                        coef = 1 / (2 ** cochain_dim)
                    star[i] += cochain[p] / inner_product(v)
            
            for k in star.keys():
                star[k] *= coef
                
            star = Cochain(cc, 0, star)
            
        elif cochain_dim == 2:
            for f in self._chain:
                for p in get_many('p', f.incident_ids):
                    for e in get_many('e', p.e_ids):
                        if cell_orthogonality_3D(f, e) == 'Orthogonal':
                            i = e.id
                            e = Cochain(compl = cc,
                                        cochain_dim = 1,
                                        cochain = {i : 1})
                            if 'simplicial' in self.structure:
                                coef = factorial(self.dim - cochain_dim) * factorial(cochain_dim) / factorial(self.dim + 1)
                            elif 'quasi-cubical' in self.structure:
                                coef = 1 / (2 ** cochain_dim)
                            if i in star.keys():
                                star[i] += cochain[f] / inner_product(e)
                            else:
                                star.update({i : cochain[f] / inner_product(e)})

            for k in star.keys():
                star[k] *= coef

            star = Cochain(cc, 1, star)
                            
        elif cochain_dim == 1:
            for e in self._chain:
                for p in get_many('p', e.incident_polyhedra_ids):
                    for f in get_many('f', p.f_ids):
                        if cell_orthogonality_3D(e, f) == 'Orthogonal':
                            i = f.id
                            f = Cochain(compl = cc,
                                        cochain_dim = 2,
                                        cochain = {i : 1})
                            if 'simplicial' in self.structure:
                                coef = factorial(self.dim - cochain_dim) * factorial(cochain_dim) / factorial(self.dim + 1)
                            elif 'quasi-cubical' in self.structure:
                                coef = 1 / (2 ** cochain_dim)
                            star[i] += cochain[e] / inner_product(f)
            
            for k in star.keys():
                star[k] *= coef
                
            star = Cochain(cc, 2, star)
                            
        elif cochain_dim == 0:
            for v in self._chain:
                for p in get_many('p', v.incident_polyhedra_ids):
                    i = p.id
                    p = Cochain(compl = cc,
                                cochain_dim = 3,
                                cochain = {i : 1})
                    if 'simplicial' in self.structure:
                        coef = factorial(self.dim - cochain_dim) * factorial(cochain_dim) / factorial(self.dim + 1)
                    elif 'quasi-cubical' in self.structure:
                        coef = 1 / (2 ** cochain_dim)
                    star[i] += cochain[v] / inner_product(p)
            
            for k in star.keys():
                star[k] *= coef
                
            star = Cochain(cc, 3, star)
        
        return star
        
    # -- #
    def star(self):
        
        if self.is_empty == True:
            return Cochain.empty()
        else:
            if self.ip == 2022:
                return self.star_2022()
            elif self.ip == 2023:
                return self.star_2023()

    # -- #
    def coboundary(self):
        
        if (self.is_empty == True or self._cochain_dim == 3):
            return Cochain.empty()
        
        cob = np.empty((0,self._cochainarray.shape[1]))
        for cell in self._chain:
            for upper in cell.signed_incident_ids:
                value = self._cochain[cell] * np.sign(upper)
                if abs(upper) not in cob[:,0]:
                    cob = np.vstack((cob, np.array([abs(upper), *value])))
                else:
                    cob[cob[:,0] == abs(upper)][1:] += value
        zero_entries = np.linalg.norm(cob[:,1:], axis=1) == 0
        cob = np.delete(cob, zero_entries, axis=0)
        return Cochain(compl = self.cell_complex,
                      cochain_dim = self._cochain_dim + 1,
                      cochain = cob)
        
    # -- #
    def codifferential(self):
        
        cochain_dim = self._cochain_dim
        
        if (self.is_empty or cochain_dim == 0):
            return Cochain.empty()
        
        cc = self.cell_complex
        chain: list = self._chain
        cochain: dict = self._cochain
        get_one = cc.get_one
        
        def basis_innprod(index, dim):
            return Cochain(cc, dim, np.array([[index,1]])).inner_product()
        
        codiff = np.empty((0, list(cochain.values())[0].shape[0] + 1)) # Add +1 to account for the first column (of indices)
        if cochain_dim == 3:
            for cell in chain:
                numerator = cochain[cell] * basis_innprod(cell.id, cochain_dim)
                lowercells = cc.get_many(cochain_dim-1, cell.f_ids)
                indices = [i.id for i in lowercells]
                denominators = [1/basis_innprod(j.id, cochain_dim-1) for j in lowercells]
                rel_oris = [rel_orientation(get_one(cochain_dim-1, k.id), cell) for k in lowercells]
                for index in range(len(indices)):
                    value = rel_oris[index] * numerator * denominators[index]
                    if indices[index] not in codiff[:,0]:
                        codiff = np.vstack((codiff, np.array([indices[index], *value])))
                    else:
                        codiff[np.argwhere(codiff[:,0] == indices[index])[0,0]][1:] += np.array([*value])
        elif cochain_dim == 2:
            for cell in chain:
                numerator = cochain[cell] * basis_innprod(cell.id, cochain_dim)
                lowercells = cc.get_many(cochain_dim-1, cell.e_ids)
                indices = [i.id for i in lowercells]
                denominators = [1/basis_innprod(j.id, cochain_dim-1) for j in lowercells]
                rel_oris = [rel_orientation(get_one(cochain_dim-1, k.id), cell) for k in lowercells]
                for index in range(len(indices)):
                    value = rel_oris[index] * numerator * denominators[index]
                    if indices[index] not in codiff[:,0]:
                        codiff = np.vstack((codiff, np.array([indices[index], *value])))
                    else:
                        codiff[np.argwhere(codiff[:,0] == indices[index])[0,0]][1:] += np.array([*value])
        elif cochain_dim == 1:
            for cell in chain:
                numerator = cochain[cell] * basis_innprod(cell.id, cochain_dim)
                lowercells = cc.get_many(cochain_dim-1, cell.v_ids)
                indices = [i.id for i in lowercells]
                denominators = [1/basis_innprod(j.id, cochain_dim-1) for j in lowercells]
                rel_oris = [rel_orientation(get_one(cochain_dim-1, k.id), cell) for k in lowercells]
                for index in range(len(indices)):
                    value = rel_oris[index] * numerator * denominators[index]
                    if indices[index] not in codiff[:,0]:
                        codiff = np.vstack((codiff, np.array([indices[index], *value])))
                    else:
                        codiff[np.argwhere(codiff[:,0] == indices[index])[0,0]][1:] += np.array([*value])

        # # Find the (p-1)-cells that will be assigned values. Allow for repetitions because the same cell
        # # might appear N times with 2**N possible sign combinations
        # match cochain_dim:
        #     case 1:
        #         new_chain = cc.get_many(0, [i for cell in chain for i in cell.v_ids])
        #     case 2:
        #         new_chain = cc.get_many(1, [i for cell in chain for i in cell.e_ids])
        #     case 3:
        #         new_chain = cc.get_many(2, [i for cell in chain for i in cell.f_ids])
        
        # codiff = np.empty((0,array.shape[1]))
        # for cell in new_chain:
        #     # Compute the denominator.
        #     denominator = Cochain(cc, cochain_dim-1, np.array([[cell.id, 1]])).inner_product()
        #     # Find the relative orientation between the (p-1)-cell and the p-cells that were included in
        #     # the original cochain.
        #     upper = cell.signed_incident_ids
        #     intersection = set(abs(j) for j in upper) & set(j.id for j in chain)
        #     upper = np.array([i for i in upper if abs(i) in intersection])
        #     if upper.size == 0:
        #         continue
        #     # The numerator is an array where each row corresponds to a contribution from a p-cell.
        #     # The contribution is equal to the value the cochain takes on that p-cell, times the inner
        #     # product of the basis cochain on that p-cell, times the relative orientation between the
        #     # p-cell and the (p-1)-cell.
        #     numerator = np.array([np.sign(u) * array[array[:,0] == abs(u)][0,1:] * Cochain(cc, cochain_dim, np.array([[abs(u), 1]])).inner_product() for u in upper])
        #     value = numerator/denominator
        #     # If the (p-1)-cell has not been visited, we simply add it along with its assigned value to
        #     # the new cochain. If it has been visited, then we need to adjust its assigned value.
            # if cell.id not in codiff[:,0]:
            #     codiff = np.vstack((codiff, np.array([cell.id, *np.sum(value, axis=0)])))
            # else:
            #     codiff[np.argwhere(codiff[:,0] == cell.id)[0,0]][1:] += np.array([*np.sum(value, axis=0)])

        zero_entries = np.linalg.norm(codiff[:,1:], axis=1) == 0
        codiff = np.delete(codiff, zero_entries, axis=0)
        return Cochain(compl = cc,
                        cochain_dim = cochain_dim - 1,
                        cochain = codiff)
    
    # -- #
    def laplacian(self):
        return self.coboundary().codifferential() + self.codifferential().coboundary()


# ----- # ----- #  FUNCTIONS # ----- # ----- #

def rel_orientation(pcell: LowerOrderCell, pplus1cell: Cell) -> int:
    place = np.argwhere(np.abs(pcell.signed_incident_ids) == pplus1cell.id)[0,0]
    return int(pcell.signed_incident_ids[place] / pplus1cell.id)

def cell_orthogonality_3D(c1, c2) -> str:
    """
    Parameters
    ----------
    c1 & c2: Vertex3D, Edge3D, Face3D or Poly

    Returns
    -------
    str
        'Parallel' or 'Orthogonal' or None.

    """
    if ((type(c1) == Poly and type(c2) == Vertex3D)
        or (type(c1) == Vertex3D and type(c2) == Poly)):
        try:
            cond1 = (c1.id in set(c2.incident_polyhedra_ids)) # the vertex is on the 3-cell
        except:
            cond1 = (c2.id in set(c1.incident_polyhedra_ids)) # the vertex is on the 3-cell
        if cond1:
            return 'Orthogonal'
        else:
            return None
    elif ((type(c1) == Face3D and type(c2) == Edge3D)
        or (type(c1) == Edge3D and type(c2) == Face3D)):
        try:
            cond1 = (len(set(c1.incident_ids) & set(c2.incident_polyhedra_ids)) == 1) # they are on the same (p+1)-cell
        except:
            cond1 = (len(set(c1.incident_polyhedra_ids) & set(c2.incident_ids)) == 1) # they are on the same (p+1)-cell
        cond2 = (len(set(c1.v_ids) & set(c2.v_ids)) == 1) # they share only one node
        cond3 = (len(set(c1.v_ids) & set(c2.v_ids)) == 0) # they share no nodes
        if (cond1 and cond2):
            return 'Orthogonal'
        elif (cond1 and cond3):
            return 'Parallel'
        else:
            return None
    elif type(c1) == type(c2) == Edge3D:
        cond1 = (len(set(c1.incident_polyhedra_ids) & set(c2.incident_polyhedra_ids)) == 1) # they are on the same (p+1)-cell
        cond2 = (len(set(c1.v_ids) & set(c2.v_ids)) == 1) # they share only one node
        cond3 = (len(set(c1.v_ids) & set(c2.v_ids)) == 0) # they share no nodes
        if (cond1 and cond2):
            return 'Orthogonal'
        elif (cond1 and cond3):
            return 'Parallel'
        else:
            return None
    else:
        raise ValueError('\nThe cells passed to cell_orthogonality_3D() must both be Edges or their dimensions must add up to 3.')


def cell_intersection(c1, c2) -> tuple:
    """
    Checks if the two cells intersect at a vertex, edge, face or polyhedra and returns a tuple containing an indicative of the order
    of the intersection ('v', 'e', 'f' or 'p') and a list of the indices of the cells in the intersection. Gives priority to higher
    order cells in the intersection (i.e. if two cells intersect at an edge, returns 'e' and the index of that edge, instead of
    'v' and the indices of the nodes that constitute that edge).
    
    Notes
    -----
    Only works for convex polytopes.
    """
    if (type(c1) in [Edge3D, Face3D, Poly] and type(c2) in [Edge3D, Face3D, Poly]):
        n = len(set(c1.v_ids) & set(c2.v_ids))
        if n == 0:
            return None
        elif n == 1:
            inters = list(set(c1.v_ids) & set(c2.v_ids))
            return ('v', inters)
        elif n == 2:
            if (type(c1) is Edge3D and type(c2) in [Face3D, Poly]):
                inters = [c1.id] if c1.id in c2.e_ids else None
            elif (type(c2) is Edge3D and type(c1) in [Face3D, Poly]):
                inters = [c2.id] if c2.id in c1.e_ids else None
            elif (type(c1) is Edge3D and type(c2) is Edge3D):
                inters = [c1.id] if c1.id == c2.id else None
            else:
                inters = list(set(c1.e_ids) & set(c2.e_ids))
            return ('e', inters)
        elif n >= 3:
            if (type(c1) is Face3D and type(c2) is Poly):
                inters1 = [c1.id] if c1.id in c2.f_ids else None
                inters2 = None
            elif (type(c2) is Face3D and type(c1) is Poly):
                inters1 = [c2.id] if c2.id in c1.f_ids else None
                inters2 = None
            elif (type(c1) is Face3D and type(c2) is Face3D):
                inters1 = [c1.id] if c1.id == c2.id else None
                inters2 = None
            elif (type(c1) is Poly and type(c2) is Poly):
                inters1 = None
                inters2 = [c1.id] if c1.id == c2.id else None
            if inters2:
                return ('p', inters2)
            else:
                return ('f', inters1)
    elif (type(c1) == Vertex3D and type(c2) in [Edge3D, Face3D, Poly]):
        n = len({c1.id} & set(c2.v_ids))
        if n == 0:
            return None
        elif n == 1:
            inters = [c1.id]
            return ('v', inters)
    elif (type(c2) == Vertex3D and type(c1) in [Edge3D, Face3D, Poly]):
        n = len({c2.id} & set(c1.v_ids))
        if n == 0:
            return None
        elif n == 1:
            inters = [c2.id]
            return ('v', inters)


def find_cell(cc: CellComplex, vertices: list[int]):
    if 'simplicial' in cc.structure:
        if len(vertices) == 1:
            return cc.get_one( 'v', vertices[0])
        elif len(vertices) == 2:
            v = cc.get_one('v', vertices[0])
            edges = cc.get_many('e', v.incident_ids)
            for e in edges:
                nodes = e.v_ids
                if (nodes[0] in vertices and nodes[1] in vertices):
                    return e
                else: pass
        elif len(vertices) == 3:
            v = cc.get_one('v', vertices[0])
            polys = cc.get_many('p', v.incident_polyhedra_ids)
            for p in polys:
                nodes = p.v_ids
                if (vertices[0] in nodes and vertices[1] in nodes and vertices[2] in nodes):
                    faces = cc.get_many('f', p.f_ids)
                    for f in faces:
                        nodes = f.v_ids
                        if (nodes[0] in vertices and nodes[1] in vertices and nodes[2] in vertices):
                            return f
                        else: pass
                else: pass
        elif len(vertices) == 4:
            v = cc.get_one('v', vertices[0])
            polys = cc.get_many('p', v.incident_polyhedra_ids)
            for p in polys:
                nodes = p.v_ids
                if (nodes[0] in vertices and nodes[1] in vertices and nodes[2] in vertices and nodes[3] in vertices):
                    return p
                else: pass


def is_cyclic_ascending(lst: list[int | float]):
    """
    This function first sorts the list and then checks if the sorted list is a cyclic permutation of the original list by
    iterating over both lists and comparing corresponding elements. If at any point the elements don't match, it returns False.
    Otherwise, it returns True if all elements match.

    Parameters
    ----------
    lst : list[int]

    Returns
    -------
    bool
        True if the input list has its elements sorted in ascending order but in cyclic order, like the numbers on a clock.

    """
    n = len(lst)
    sorted_lst = sorted(lst)
    min_index = lst.index(min(lst)) # find the index of the smallest element
    # Check if the sorted list is a cyclic permutation of the original list
    for i in range(n):
        if lst[(min_index + i) % n] != sorted_lst[i]:
            return False
    return True


def permutation_parity(lst1: list[int | float], lst2: list[int | float]) -> int:
    """
    This function takes in two lists, checks if the sets of their elements are the same, and then returns 0 if the second list
    is an even permutation of the first and returns 1 if the second list is an odd permutation of the first. It leaves the two
    input lists unchanged.
    
    Parameters
    ----------
    lst1 : list[int | float]
    lst2 : list[int | float]

    Raises
    ------
    ValueError
        If the the sets of elements of the two lists are not the same.

    Returns
    -------
    int
        0 if the second list is an even permutation of the first; 1 if the second list is an odd permutation of the first.

    """
    # Check if the sets of elements are the same
    if set(lst1) != set(lst2):
        raise ValueError("\nThe sets of elements in the two lists must be the same.")
    # Make copies of the input lists
    lst1_copy = lst1[:]
    lst2_copy = lst2[:]
    # Count the number of swaps required to sort list1_copy into list2_copy
    swaps = 0
    for i in range(len(lst1_copy)):
        if lst1_copy[i] != lst2_copy[i]:
            j = lst1_copy.index(lst2_copy[i])
            lst1_copy[i], lst1_copy[j] = lst1_copy[j], lst1_copy[i]
            swaps += 1
    # Determine if the permutation is even or odd
    if swaps % 2 == 0:
        return 0  # Even permutation
    else:
        return 1  # Odd permutation


def get_new_simplex(lst: list[int | float]):
    """
    In the context of computing the cup product, a new simplex can be obtained from joining two lists of vertex indices together, but
    we must require that there be one and only one repeated element in the new list, which itself only repeats once - this will be
    the index of the vertex where the two cells involved in the cup product meet. This function returns a list of the unique elements
    of the input list while preserving their order relative to each other; returns None if there is more than one repeating element;
    returns None if there is an element that repeats more than once.
    
    Parameters
    ----------
    lst : list[int | float]

    Returns
    -------
    unique_elements : list[int | float]
    """
    lst_copy = lst[:]  # Create a copy of the input list
    # Count how many times an element shows up in the input list
    counts = {}
    for x in lst_copy:
        counts[x] = counts.get(x, 0) + 1
    # The function returns None if there is not just one repeating element or if an element repeats more than once.
    repeated_elements = 0
    for value in counts.values():
        if value > 1:
            repeated_elements += 1
    if repeated_elements != 1:
        return None
    unique_counts = set(counts.values())
    if len(unique_counts) > 2:
        # Set it as > 2 because we must allow for the count of 1 for each unique element and then one instance of
        # a count of 2 for the one once-repeated element.
        return None
    for x in unique_counts:
        if x > 2:
            return None
    # Create a new list of the uinque elements of the input list without changing their order relative to each other.
    seen = set()
    unique_elements = []
    for item in lst_copy:
        if item not in seen:
            seen.add(item)
            unique_elements.append(item)
    return unique_elements


def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """
    Merges two dictionaries. Where the dictionaries have the same keys, 
    the function adds the corresponding values in the resulting dictionary.
    Where the dictionaries do not have the same keys, the function adds 
    those keys and their corresponding values to the resulting dictionary.

    Parameters:
    dict1 (dict): The first dictionary.
    dict2 (dict): The second dictionary.

    Returns:
    dict: A dictionary resulting from the merging of dict1 and dict2.
    """
    result = dict1.copy()  # Start with a copy of the first dictionary

    for key, value in dict2.items():
        if key in result:
            result[key] += value # Add the values if the key is in both dictionaries
        else:
            result[key] = value # Add the key-value pair if the key is only in the second dictionary

    return result


def subtract_dictionaries(dict1: dict, dict2: dict) -> dict:
    """
    Merges two dictionaries. Where the dictionaries have the same keys,
    the function subtracts the value in the second dictionary from the value
    in the first dictionary in the resulting dictionary. Where the dictionaries 
    do not have the same keys, the function adds those keys and their 
    corresponding values to the resulting dictionary.

    Parameters:
    dict1 (dict): The first dictionary.
    dict2 (dict): The second dictionary.

    Returns:
    dict: A dictionary resulting from the subtraction of dict2 from dict1.
    """
    result = dict1.copy()  # Start with a copy of the first dictionary

    for key, value in dict2.items():
        if key in result:
            result[key] -= value # Subtract the value if the key is in both dictionaries
        else:
            result[key] = -value # Add the key with its negative value if the key is only in the second dictionary

    return result


@lru_cache()
def innerprod2022_coef(get, cell, cochain_dim) -> float:
    
    vertices = get('v', cell.v_ids) if cochain_dim != 0 else [cell]
    sum2 = 0
    for v in vertices:
        polys = get('p', v.incident_polyhedra_ids)
        sum2 += v.measure * sum([p.measure for p in polys])
    
    return sum2


def all_vertices_cochain(cc: CellComplex):
    vertices = cc.vertices
    return Cochain(compl = cc, cochain_dim = 0, cochain = np.array([[i+1,1] for i in range(len(vertices))]))

def all_edges_cochain(cc: CellComplex):
    edges = cc.edges
    return Cochain(compl = cc, cochain_dim = 1, cochain = np.array([[i+1,1] for i in range(len(edges))]))

def all_faces_cochain(cc: CellComplex):
    faces = cc.faces
    return Cochain(compl = cc, cochain_dim = 2, cochain = np.array([[i+1,1] for i in range(len(faces))]))

def all_poly_cochain(cc: CellComplex):
    polyhedra = cc.polyhedra
    return Cochain(compl = cc, cochain_dim = 3, cochain = np.array([[i.id,i.measure] for i in polyhedra]))


# ----- # ----- #  CODE # ----- # ----- #

# if __name__ == '__main__':
    
#     import cellcomplex as cc
    
#     sys.path.append('../')
#     from dccstructure.iofiles import import_complex_data
#     from pathlib import Path
#     from time import time
    
#     t0 = time()
    
#     data_folder = Path('../Built_Complexes/FCC_5x5x5')
    
#     complex_size = [int(str(data_folder)[-5]), int(str(data_folder)[-3]), int(str(data_folder)[-1])] # the number of computational unit cells in each 3D dimension
    
#     nodes, edges, faces, faces_slip, faces_areas, faces_normals, volumes, volumes_vols, nr_cells, A0, A1, A2, A3, B1, B2, B3 = import_complex_data(data_folder)
    
#     edges += 1 ; faces += 1 ; volumes += 1  ;  edges = edges.astype(int)
    
#     faces_slip = [x + 1 for x in faces_slip]

#     A0[:,0:2] += 1 ; A1[:,0:2] += 1 ; A2[:,0:2] += 1 ; A3[:,0:2] += 1 ; B1[:,0:2] += 1 ; B2[:,0:2] += 1 ; B3[:,0:2] += 1

#     Nodes, Edges, Faces, Polyhedra = cc.createAllCells((nodes + 0.5),
#                                                         edges,
#                                                         faces,
#                                                         volumes,
#                                                         faces_normals,
#                                                         A0, A1, A2, A3,
#                                                         B1, B2, B3,
#                                                         faces_areas,
#                                                         volumes_vols)
    
#     del A0, A1, A2, A3, B1, B2, B3
#     del nodes, edges, faces, faces_areas, faces_normals, volumes, volumes_vols
        
#     cc1 = cc.createCellComplex(dim = 3,
#                                 nodes = Nodes,
#                                 edges = Edges,
#                                 faces = Faces,
#                                 polyhedra = Polyhedra,
#                                 ip = 2022)
    
#     cc2 = cc.createCellComplex(dim = 3,
#                                 nodes = Nodes,
#                                 edges = Edges,
#                                 faces = Faces,
#                                 polyhedra = Polyhedra,
#                                 ip = 2023)
    
#     a1 = all_vertices_cochain(cc1)
#     a2 = all_vertices_cochain(cc2)
#     vol1 = all_poly_cochain(cc1)
#     vol2 = all_poly_cochain(cc2)
    
#     x = Cochain(cc1,2,np.array([[i+1, *np.random.randint(low=0, high=10, size=3)] for i in range(int(cc1.facenb/3))]))
#     y = Cochain(cc1,2,np.array([[i+30, *np.random.randint(low=0, high=10, size=3)] for i in range(int(cc1.facenb/3))]))
    
#     print("\n\nInner product (2022) of 0-cochain of all-ones: ⟨1,1⟩ = {:.3f}".format(round(a1.inner_product(), 3)))
#     print("Inner product (2022) of 3-cochain of all-volumes: ⟨vol,vol⟩ = {:.3f}".format(round(vol1.inner_product(), 3)))
#     print("Inner product (2023) of 0-cochain of all-ones: ⟨1,1⟩ = {:.3f}".format(round(a2.inner_product(), 3)))
#     print("Inner product (2023) of 3-cochain of all-volumes: ⟨vol,vol⟩ = {:.3f}".format(round(vol2.inner_product(), 3)))
    # print("\nvol = ☆1 : {} | ☆vol = 1 : {}  (2023)".format(vol2.star().sort() == a2, a2.star().sort() == vol2))

    # test0 = Cochain(cc1, 0, {v.id : 1 for v in cc1.vertices[:500]})
    # test1 = Cochain(cc1, 1, {e.id : 1 for e in cc1.edges[:500]})
    # test2 = Cochain(cc1, 2, {f.id : 1 for f in cc1.faces[:500]})
    # test3 = Cochain(cc1, 3, {p.id : 1 for p in cc1.polyhedra[:500]})

    # t0 = time()
    
    # innerproducts = [test0.inner_product(), test1.inner_product(), test2.inner_product(), test3.inner_product()]
    
    # print(time() - t0)


