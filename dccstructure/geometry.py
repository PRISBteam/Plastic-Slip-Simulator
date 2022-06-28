# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:14 2022

Last edited on: 28/06/2022 15:45

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions that compute geometric quantities related to the
complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np


# ----- # ----- # FUNCTIONS # ------ # ----- #


def unit_normal(points):
    """
    Parameters
    ----------
    points : np array (N x 3)
        An array whose rows are 3D coordinates of coplanar points.

    Returns
    -------
    The unit normal vector of the plane defined by the points.
    """
    
    # We take two vectors that point along two edges of a face and find their vector cross product. We define the orientation of
    # a face as corresponding to this vector.
    
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    
    normal = np.cross(v1, v2)
    
    # Now we just get the unit normal by dividing it by its length.
    
    normal = normal / np.linalg.norm(normal)
    
    return normal



def vector_angle(vector1, vector2):
    """
    Parameters
    ----------
    vector1 : TYPE
        DESCRIPTION.
    vector2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    
    unit_vectors = [vector1 / np.linalg.norm(vector1) , vector2 / np.linalg.norm(vector2)]
    
    dot = np.dot(unit_vectors[0], unit_vectors[1])
    
    angle = np.arccos(dot)
    
    return angle    



def polygon_area(points):
    """
    Parameters
    ----------
    points : np array (N x 3)
        An array whose rows are 3D coordinates of points.
    
    Returns
    -------
    The area of the polygon defined by the points given.
    
    Notes
    -------
    Based on the answer by Jamie Bull on https://stackoverflow.com/questions/12642256/find-area-of-polygon-from-xyz-coordinates (Accessed 20 Jun 2022).
    """
    
    if len(points) < 3: # not a plane - no area
    
        raise ValueError("/nThe array submitted into geometry.polygon_area() must have at least 3 points to define a plane./n")
        
    total = np.zeros((1,3))
                
    for i in range(len(points)):
        
        reference1 = points[i]
        
        reference2 = points[(i+1) % len(points)]
        
        xprod = np.cross(reference1, reference2)
        
        total += xprod
        
    result = np.dot(total, unit_normal(points))
    
    return float(abs(result/2))



def tetrahedron_volume(points):
    """
    Parameters
    ----------
    points : np array (4 x 3)
        An array whose rows are 3D coordinates of points.
    
    Returns
    -------
    The volume of the tetrahedron defined by the points given.
    
    Notes
    -------
    Based on https://en.wikipedia.org/wiki/Tetrahedron#Volume (Accessed 20 Jun 2022).
    """

    volume = 1/6 * abs(np.dot(points[0] - points[3] ,
                              np.cross(points[1] - points[3] , points[2] - points[3])
                              ))
    
    return volume


def geo_measure(cell):
    """
    Parameters
    ----------
    cell : np array (N x 3)
        An array whose rows are 3D coordinates of points.

    Returns
    -------
    The geometric measure (length, area, volume) of a cell in a simplicial complex.
    """
    
    nr_nodes = np.shape(cell)[0]
    
    if nr_nodes == 1:
        
        return 1
    
    elif nr_nodes == 2:
        
        return np.linalg.norm(cell[1] - cell[0])
    
    elif nr_nodes == 3:
        
        return polygon_area(cell)
    
    elif nr_nodes == 4:
        
        return tetrahedron_volume(cell)



def angle_measure(cell_d, node, cells_0D):
    """
    Parameters
    ----------
    cell_d : list OR np array
        A listing of the indices of the constituent nodes of a top-dimensional cell in the complex.
    node : int
        The index of a node in the discrete complex.
    cells_0D: np array (N x 3)
        A numpy array whose rows list the spatial coordinates of points.

    Returns
    -------
    The angle measure as given in Definition 3.11 of Berbatov, K., et al. (2022). "Diffusion in multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.
        
    Notes
    -------
    Based on https://en.wikipedia.org/wiki/Solid_angle#Solid_angles_for_common_objects (the one from L'Huilier's theorem) (Accessed 21 Jun 2022).
    """
    
    if node not in cell_d:
        
        raise ValueError("\nWhen calling the function geometry.angle_measure(cell_d, node, cells_0D), the node must be a constituent node of cell_d.\n")
        
    other_nodes = np.delete(cell_d, np.argwhere(cell_d == node))
            
    v1 = cells_0D[other_nodes[0]] - cells_0D[node]
    v2 = cells_0D[other_nodes[1]] - cells_0D[node]
    v3 = cells_0D[other_nodes[2]] - cells_0D[node]

    angle1 = vector_angle(v2, v3)
    angle2 = vector_angle(v1, v3)
    angle3 = vector_angle(v1, v2)
    
    s = np.sum([angle1, angle2, angle3]) / 2
    
    solid_angle = 4 * np.arctan(np.sqrt(np.tan(s/2) * np.tan((s - angle1)/2) * np.tan((s - angle2)/2) * np.tan((s - angle3)/2)))
            
    return solid_angle

