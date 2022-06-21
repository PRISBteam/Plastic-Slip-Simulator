# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:27 2022

Last edited on: 21/06/2022 18:00

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions that correspond to topological and metric operations
on the DCC complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
from math import factorial
import geometry
# from build_complex import build_complex


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




def metric_tensor(cell, cells_3D, cells_0D):
    """
    Parameters
    ----------
    cell : list OR np array
        A listing of the indices of the constituent nodes of a p-cell in the complex.
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one 3-cell.
    cells_0D: np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.

    Returns
    -------
    The metric tensor for simplicial complexes as given in Definition 3.4 of Berbatov, K., et al. (2022). "Diffusion in
    multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.
    """
    
    chain = np.zeros((np.shape(cells_0D)[0], 1))
        
    denominator = np.shape(cell)[0] * (geometry.geo_measure(cells_0D[cell])) ** 2
    
    for node in cell:
        
        chain[node] = node_weight(node, cells_3D, cells_0D)
    
    return chain / denominator




def inner_product(cell, cells_3D, cells_0D, dim=3):
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

    Returns
    -------
    The inner product for simplicial complexes as given in Definition 3.7 and Remark 3.9 of Berbatov, K., et al. (2022). "Diffusion
    in multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.

    """
    
    numerator = 0
    
    for node in cell:
        
        incident_vols =  cells_3D[np.argwhere(cells_3D == node)[:,0]] # array whose rows list the indices of the constituent
                                                                      # nodes of the 3-cells incident on 'node'
        
        numerator += node_weight(node, cells_3D, cells_0D) * np.sum([geometry.geo_measure(cells_0D[i]) for i in incident_vols])
    
    denominator = np.shape(cell)[0] * (dim + 1) * (geometry.geo_measure(cells_0D[cell])) ** 2
    
    return numerator / denominator




def star():
    
    return 1
    