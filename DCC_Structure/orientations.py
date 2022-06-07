# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 12:43 2022

Last edited on: 07/06/2022 14:50

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the DCC_Structure package. In here you will find functions that compute an unoriented version of the incidence
matrices, as well as a function which uses that to compute the relative orientation between volumes and faces and between faces
and edges.

"""


# ----- #----- #  IMPORTS # ----- # ----- #


import numpy as np
from itertools import combinations # to avoid nested for loops
from dcc_build import find_equal_rows
from dcc_io import write_to_file


# ----- # ----- # FUNCTIONS # ------ # ----- #


def volumes_to_faces(cells_3D, cells_2D, faces_per_volume=4):
    """
    Parameters
    ----------
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one volume.
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    faces_per_volume: int
        The number of faces which make up one volume. The default is set to 4, as both the BCC and FCC structures use only tetrahedra.

    Returns
    -------
    volumes_as_faces : np array
        An array whose rows list the indices of faces which make up one volume.
    """

    volumes_as_faces = np.empty((0, faces_per_volume))

    for vol in cells_3D:
        
        # First we take the nodes in a volume and arrange them in combinations of N nodes, where N is the number of nodes per face.
        # These represent possible faces in the selected volume.
        
        faces_in_vol = np.sort(np.array([i for i in combinations(vol, np.shape(cells_2D)[1])]))
        
        # Now we take these combinations and see which ones match up to actual faces.
        
        faces_in_vol = find_equal_rows(cells_2D, faces_in_vol)[:,0]
        
        faces_in_vol = np.sort(faces_in_vol)
                                        
        volumes_as_faces = np.vstack((volumes_as_faces, faces_in_vol))
    
    return volumes_as_faces.astype(int)




def faces_to_edges(cells_2D, cells_1D, edges_per_face=3):
    """
    Parameters
    ----------
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    cells_1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    edges_per_face: int
        The number of edges which make up one face. The default is set to 3, as both the BCC and FCC structures use only tetrahedra.

    Returns
    -------
    faces_as_edges : np array
        An array whose rows list the indices of edges which make up one face.
    """

    faces_as_edges = np.empty((0, edges_per_face))

    for face in cells_2D:
        
        # First we take the nodes in a face and arrange them in combinations of 2 nodes. These represent possible edges in
        # the selected face.
        
        edges_in_face = np.sort(np.array([i for i in combinations(face, 2)]))
        
        # Now we take these combinations and see which ones match up to actual edges.
        
        edges_in_face = find_equal_rows(cells_1D, edges_in_face)[:,0]
        
        edges_in_face = np.sort(edges_in_face)
                                        
        faces_as_edges = np.vstack((faces_as_edges, edges_in_face))
    
    return faces_as_edges.astype(int)




def find_relative_orientations(cells_3D, cells_2D, cells_1D, cells_0D, v2f, f2e):
    """
    Parameters
    ----------
    cells_3D : np array
        An array whose rows list the indices of nodes which make up one volume.
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    cells_1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells_0D : np array
        An array whose rows list the coordinates of the nodes.
    v2f : np array
        An array whose rows list the indices of faces which make up one volume.
    f2e : np array
        An array whose rows list the indices of edges which make up one face.

    Returns
    -------
    modified 'v2f' : np array
        An array whose rows list the indices of faces which make up one volume, with considerations for relative orientation.
    modified 'f2e' : np array
        An array whose rows list the indices of edges which make up one face, with considerations for relative orientation.
    """
    
    
    # Relative orientation between volumes and faces.
        
    for vol_index in range(0, np.shape(cells_3D)[0]):
        
        Array = cells_0D[cells_2D[v2f[vol_index]]]
        
        # This 'Array' will be a 3D array where each index along axis 0 corresponds to a face of the volume cells_3D[vol_index], each
        # index along axis 1 corresponds to a node of a face of the volume, and each index along axis 2 corresponds to a coordinate
        # of a node of a face of the volume.
        
        # The orientation of every volume is taken as positive and interpreted as pointing outward of the volume. To find relative
        # orientations, we need to consider the orientations of the faces. We take two vectors that point along two edges of a face
        # and find their vector cross product. We define the orientation of a face as corresponding to this vector. Then, we find that
        # a volume and one of its faces have a positive relative orientation if the inner product between the face's orientation and
        # a vector pointing from the centroid of the volume to the centroid of the face is positive.
        
        # Step 1. Find two vectors along two edges of each face and then the orientations of the faces.
        
        v1 = Array[:,1] - Array[:,0]
        v2 = Array[:,2] - Array[:,0]
        
        orientation_face = np.cross(v1, v2)
        
        # Step 2. Find the orientation vectors from the centroid of the volume to the centroids of each face.
        
        orientation_volume = np.average(Array, axis=1) - np.average(cells_0D[cells_3D[vol_index]], axis=0)
        
        # Step 3. Calculate the inner product between the two orientations and attribute relative orientations. We do this by
        # adding a minus sign in the array 'v2f' front of a face which is relatively negatively oriented wrt the volume.
        
        for i in range(0, np.shape(v2f[vol_index])[0]): # for each face in the current volume
            
            if np.inner(orientation_face[i], orientation_volume[i]) < 0:
                
                v2f[vol_index, i] *= -1
                
            else: pass
        
    # relative orientation between faces and edges
    
    for face_index in range(0, np.shape(cells_2D)[0]):
    
        Array = cells_0D[cells_1D[f2e[face_index]]]
        
        # This 'Array' will be a 3D array where each index along axis 0 corresponds to an edge of the face cells_2D[face_index], each
        # index along axis 1 corresponds to a node of an edge of the face, and each index along axis 2 corresponds to a coordinate
        # of a node of an edge of the face.
        
        # The orientation of every edge [A,B] is considered to correspond to a vector which points from node A to node B. For the
        # faces, the orientations are defined the same as above. For each face and each of its edges, we find the vector pointing
        # from the centroid of the face to the centroid of the edge, and compute its cross product with the orientation of the edge.
        # Then, we say that the face and the edge have a positive relative orientation if the inner product between this cross product
        # and the orientation of the face is negative.
        
        # Step 1. Find vectors along the edges of each face and then the orientations of the faces.
        
        orientation_edges = Array[:,1] - Array[:,0]
        
        orientation_face = np.cross(orientation_edges[0], orientation_edges[1])
        
        # Step 2. Find vectors pointing from the centroid of the face to the centroid of each edge.
        
        vectors = np.average(Array, axis=1) - np.average(cells_0D[cells_2D[face_index]], axis=0)
        
        # Step 3. Find the cross product between orientation_face and each of the 'vectors', and compute the inner product between
        # this cross product and the orientation_edges. Finally, define the relative orientations appropriately.
        
        for i in range(0, np.shape(f2e[face_index])[0]): # for each edge in the current face
            
            x_product = np.cross(vectors[i], orientation_edges[i])
            
            if np.inner(x_product, orientation_face) < 0:
                
                f2e[face_index, i] *= -1
                
            else: pass
        
    return v2f, f2e
