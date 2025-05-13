# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:27 2022

Last edited on: 08/02/2024 15:45

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions that correspond to topological and metric operations
on the DCC complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #

import numpy as np
from math import factorial
import sys

sys.path.append('../')
import dccstructure.geometry as geometry

# ----- # ----- # FUNCTIONS # ------ # ----- #



def node_weight(node, cells_3D, cells_0D):
    """
    Parameters
    ----------
    node : int
        The index of a node in the discrete complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_0D: np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.

    Returns
    -------
    The node weight as given in Definition 3.13 of Berbatov, K., et al. (2022). "Diffusion in multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.
    """
    
    unit_sphere = 4 * np.pi
    
    incident_vols =  cells_3D[np.argwhere(cells_3D == node)[:,0]]
    
    denominator = np.sum([geometry.angle_measure(incident_vols[i], node, cells_0D) for i in range(len(incident_vols))])
    
    return unit_sphere / denominator




def metric_tensor(cell, cells_3D, cells_0D, compl):
    """
    Parameters
    ----------
    cell : list OR np array
        A listing of the indices of the constituent nodes of a p-cell in the complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_0D: np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.
    compl: str, optional
        The type of complex. Supported options are 'simplicial' (default) and 'quasi-cubical'.


    Returns
    -------
    The metric tensor for simplicial complexes as given in Definition 3.4 of Berbatov, K., et al. (2022). "Diffusion in
    multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.
    """
    
    chain = np.zeros((np.shape(cells_0D)[0], 1))
    
    if compl == 'simplicial':
        
        denominator = np.shape(cell)[0] * (geometry.geo_measure(cells_0D[cell])) ** 2
        
    elif compl == 'quasi-cubical':
        
        denominator = 2 ** (np.shape(cell)[0] - 1) * (geometry.geo_measure(cells_0D[cell])) ** 2
        
    elif compl not in ['simplicial', 'quasi-cubical']:
        
        print("\nWhen calling the function operations.inner_product(), the 'compl' argument must be set to either 'simplicial' or 'quasi-cubical'.\n")
            
    for node in cell:
        
        chain[node] = node_weight(node, cells_3D, cells_0D)
    
    return chain / denominator




def inner_product(cell, cells_3D, cells_0D, dim=3, compl='simplicial'):
    """
    Parameters
    ----------
    cell : list OR np array
        A listing of the indices of the constituent nodes of a p-cell in the complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_0D: np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.
    dim : int, optional
        The physical dimension of the complex. The default is 3.
    compl: str, optional
        The type of complex. Supported options are 'simplicial' (default) and 'quasi-cubical'.

    Returns
    -------
    The inner product for cell complexes as given in Definition 3.7 and Remark 3.9 of Berbatov, K., et al. (2022). "Diffusion
    in multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.

    """
    
    if type(cell) not in [list, np.ndarray]:
        
        cell = [cell]
    
    numerator = 0
    
    for node in cell:
        
        incident_vols =  cells_3D[np.argwhere(cells_3D == node)[:,0]] # array whose rows list the indices of the constituent nodes of the 3-cells incident on 'node'
        
        numerator += node_weight(node, cells_3D, cells_0D) * np.sum([geometry.geo_measure(cells_0D[i]) for i in incident_vols])
    
    if compl == 'simplicial':
        
        denominator = np.shape(cell)[0] * (dim + 1) * (geometry.geo_measure(cells_0D[cell])) ** 2
        
    elif compl == 'quasi-cubical':
        
        denominator = 2 ** (np.shape(cell)[0] - 1 + dim) * (geometry.geo_measure(cells_0D[cell])) ** 2
        
    elif compl not in ['simplicial', 'quasi-cubical']:
        
        print("\nWhen calling the function operations.inner_product(), the 'compl' argument must be set to either 'simplicial' (default) or 'quasi-cubical'.\n")
    
    return numerator / denominator



def adj_coboundary(cell, cells_3D, cells_2D, cells_1D, cells_0D, v2f, f2e, e2n, dim=3):
    """
    Parameters
    ----------
    cell : list OR np array
        A listing of the indices of the constituent nodes of a p-cell in the complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    cells_1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells_0D : np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.
    v2f : np array
        An array whose rows list the indices of faces which make up one volume, with considerations for relative orientation.
    f2e : np array
        An array whose rows list the indices of edges which make up one face, with considerations for relative orientation.
    e2n : np array
        An array whose rows list the indices of nodes which make up one edge, with considerations for relative orientation.
    dim : int, optional
        The physical dimension of the complex. The default is 3.

    Returns
    -------
    adjcobnd : np array
        A column vector representing the adjoint coboundary of 'cell' as a cochain. Based on Equation (14) of Berbatov, K., et al.
        (2022). "Diffusion in multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.
        
    
    """
    
    # First we need to ensure that 'cell' is a column vector
    
    if type(cell) == list:
        
        cell = np.array(cell).reshape(3,1)
        
    if np.shape(cell)[1] != 1:
        
        cell.reshape(len(cell),1)
        
    # Now we find the adjoint coboundary of 'cell' for each type of p-cell
        
    p = len(cell)
    
    prod = inner_product(cell, cells_3D, cells_0D)
    
    if p == 3:
        
        adjcobnd = 0
        
    elif p == 2:
        
        adjcobnd = v2f * cell
        
        nz_entries = np.argwhere(adjcobnd)[:,0] # identify the non-zero entries of adjcobnd - these will be the indices of the
                                                # (p-1)-cells on the boundary of 'cell'
        
        for i in nz_entries:
            
            adjcobnd[i] *= prod / inner_product(cells_3D[i], cells_3D, cells_0D)
        
    elif p == 1:
        
        adjcobnd = f2e * cell
        
        nz_entries = np.argwhere(adjcobnd)[:,0] # identify the non-zero entries of adjcobnd - these will be the indices of the
                                                # (p-1)-cells on the boundary of 'cell'
        
        for i in nz_entries:
            
            adjcobnd[i] *= prod / inner_product(cells_2D[i], cells_3D, cells_0D)
        
    elif p == 0:
        
        adjcobnd = e2n * cell
        
        nz_entries = np.argwhere(adjcobnd)[:,0] # identify the non-zero entries of adjcobnd - these will be the indices of the
                                                # (p-1)-cells on the boundary of 'cell'
        
        for i in nz_entries:
            
            adjcobnd[i] *= prod / inner_product(cells_1D[i], cells_3D, cells_0D)
            
    return adjcobnd
        


def star_3(cell, cells_3D, cells_0D, dim=3, compl='simplicial'):
    """
    Parameters
    ----------
    cell : list OR np array
        A listing of the indices of the constituent nodes of a p-cell in the complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_0D : np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.
    dim : int, optional
        The physical dimension of the complex. The default is 3.
    compl: str, optional
        The type of complex. Supported options are 'simplicial' (default) and 'quasi-cubical'.

    Returns
    -------
    int
        DESCRIPTION.
    """
    
    result = np.empty((0,2))
    
    if type(cell) == list:
        
        cell = np.array(cell)
        
    if compl == 'simplicial':
        
        for node in cell:
        
            coefficient = (1 / 4) / inner_product(node, cells_3D, cells_0D, dim, compl)
            
            result = np.vstack((result, np.array([int(node), coefficient])))
            
    elif compl == 'quasi-cubical':
        
        for node in cell:
        
            coefficient = (1 / 2**3) / inner_product(node, cells_3D, cells_0D, dim, compl)
            
            result = np.vstack((result, np.array([int(node), coefficient])))
            
    elif compl not in ['simplicial', 'quasi-cubical']:
        
        print("\nWhen calling the function operations.star_3(), the 'compl' argument must be set to either 'simplicial' (default) or 'quasi-cubical'.\n")
        
    return result



def star(cell, cells_3D, cells_2D, cells_1D, cells_0D, v2f, f2e, e2n, dim=3, compl='simplicial'):
    """
    Parameters
    ----------
    cell : list OR np array
        A listing of the indices of the constituent nodes of a p-cell in the complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    cells_1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells_0D : np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.
    v2f : np array
        An array whose rows list the indices of faces which make up one volume, with considerations for relative orientation.
    f2e : np array
        An array whose rows list the indices of edges which make up one face, with considerations for relative orientation.
    e2n : np array
        An array whose rows list the indices of nodes which make up one edge, with considerations for relative orientation.
    dim : int, optional
        The physical dimension of the complex. The default is 3.
    compl: str, optional
        The type of complex. Supported options are 'simplicial' (default) and 'quasi-cubical'.

    Returns
    -------
    int
        DESCRIPTION.
    """
    
    return 1
    