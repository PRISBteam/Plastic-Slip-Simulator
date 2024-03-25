# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2022

Last edited on: Mar 13 12:00 2024

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions that compute an unoriented version of the incidence
matrices, as well as a function which uses that to compute the relative orientation between volumes and faces and between faces
and edges.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
from itertools import combinations # to avoid nested for loops
from functools import partial
import multiprocessing as mp
# from pathlib import Path
import os
import time

import sys
sys.path.append('../')
sys.path.append('./')

from dccstructure.build import find_equal_rows
# from dccstructure_mp.iofiles_mp import write_to_file



# ----- # ----- # FUNCTIONS # ------ # ----- #



def vol_to_faces(cell: np.ndarray,
                 cells2D: np.ndarray,
                 faces_per_volume = 4):
    """
    Parameters
    ----------
    cell : np array
        An array whose rows list the indices of nodes which make up one volume (3-cell).
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    faces_per_volume: int
        The number of faces which make up one volume. The default is set to 4, as both the BCC and FCC structures use only tetrahedra.

    Returns
    -------
    volumes_as_faces : np array
        An array whose rows list the indices of faces which make up one volume.
    """
        
    # First we take the nodes in a volume and arrange them in combinations of N nodes, where N is the number of nodes per face.
    # These represent possible faces in the selected volume.
    
    faces_in_vol = np.sort(np.array([i for i in combinations(cell, np.shape(cells2D)[1])]))
    
    # Now we take these combinations and see which ones match up to actual faces. It would be too expensive to search the whole array
    # 'cells2D', so we try to limit the search to the faces containing nodes in 'cell'.
    
    possible_faces = []
    
    for x in cell:
        possible_faces = possible_faces + list(np.where(cells2D[:,0] == x)[0])
        
    faces_in_vol = find_equal_rows(faces_in_vol, cells2D[possible_faces])[:,1]
    
    # The indices in the array 'faces_in_vol' now refer to the array 'possible_faces', so we need to convert those indices into indices
    # of the rows of 'cells2D'.
    
    faces_in_vol = np.array(possible_faces)[faces_in_vol]
                                            
    return faces_in_vol.astype(int)




def face_to_edges(cell: np.ndarray,
                  cells1D: np.ndarray,
                  edges_per_face = 3):
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
            
    # First we take the nodes in a face and arrange them in combinations of 2 nodes. These represent possible edges in
    # the selected face.
    
    edges_in_face = np.sort(np.array([i for i in combinations(cell, 2)]))
    
    # Now we take these combinations and see which ones match up to actual edges. It would be too expensive to search the whole array
    # 'cells1D', so we try to limit the search to the edges containing nodes in 'cell'.
    
    possible_edges = []
    
    for x in cell:
        possible_edges = possible_edges + list(np.where(cells1D[:,0] == x)[0])
    
    edges_in_face = find_equal_rows(edges_in_face, cells1D[possible_edges])[:,1]
    
    # The indices in the array 'edges_in_face' now refer to the array 'possible_edges', so we need to convert those indices into indices
    # of the rows of 'cells1D'.
    
    edges_in_face = np.array(possible_edges)[edges_in_face]
    
    return edges_in_face.astype(int)




def find_relative_orientations_v2f(cell: list,
                                   cells2D: np.ndarray,
                                   cells0D: np.ndarray):
    """
    Parameters
    ----------
    cell : list
        An ordered triplet whose first element is an array of the constitutent nodes of a volume, second element is
        that volume's index, and third element is an array of the constituent (unoriented) faces of that volume.
    cells2D : np array
        An array whose rows list the indices of nodes which make up one face.
    cells0D : np array
        An array whose rows list the coordinates of the nodes.

    Returns
    -------
    modified cell[2] : np array
        An array whose rows list the indices of faces which make up one volume, with considerations for relative orientation.
    """
    
    
    # Relative orientation between volumes and faces.
                
    Array = cells0D[cells2D[cell[2]]]
    
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
    
    orientation_volume = np.average(Array, axis=1) - np.average(cells0D[cell[0]], axis=0)
    
    # Step 3. Calculate the inner product between the two orientations and attribute relative orientations. We do this by
    # adding a minus sign in the array 'oriented_v2f' (defined below) front of a face which is relatively negatively oriented
    # with respect to the volume.
    
    oriented_v2f = np.copy(cell[2])
    
    for i in range(0, np.shape(cell[2])[0]): # for each face in the current volume
        
        if np.inner(orientation_face[i], orientation_volume[i]) < 0:
            
            oriented_v2f[i] *= -1
            
        else: pass
        
    return oriented_v2f.astype(int)




def find_relative_orientations_f2e(cell: list,
                                   cells1D: np.ndarray,
                                   cells0D: np.ndarray):
    """
    Parameters
    ----------
    cell : tuple
        An ordered triplet whose first element is an array of the constitutent nodes of a volume, second element is
        that volume's index, and third element is an array of the constituent (unoriented) faces of that volume.
    cells1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells0D : np array
        An array whose rows list the coordinates of the nodes.
    f2e : np array
        An array whose rows list the indices of edges which make up one face.

    Returns
    -------
    modified 'f2e' : np array
        An array whose rows list the indices of edges which make up one face, with considerations for relative orientation.
    """

        
    # relative orientation between faces and edges
        
    Array = cells0D[cells1D[cell[2]]]
    
    # This 'Array' will be a 3D array where each index along axis 0 corresponds to an edge of the face cells_2D[cell[1]], each
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
    
    vectors = np.average(Array, axis=1) - np.average(cells0D[cell[0]], axis=0)
    
    # Step 3. Find the cross product between orientation_face and each of the 'vectors', and compute the inner product between
    # this cross product and the orientation_edges. Finally, define the relative orientations appropriately.
    
    oriented_f2e = np.copy(cell[2])

    for i in range(0, np.shape(cell[2])[0]): # for each edge in the current face
        
        x_product = np.cross(vectors[i], orientation_edges[i])
        
        if np.inner(x_product, orientation_face) < 0:
            
            oriented_f2e[i] *= -1
            
        else: pass
        
    return oriented_f2e.astype(int)




def get_orientations(cells0D: np.ndarray,
                     cells1D: np.ndarray,
                     cells2D: np.ndarray,
                     cells3D: np.ndarray,
                     faces_per_volume: int,
                     edges_per_face: int):
    """
    This function summarises the entire "orientations" part of the DCCStructure package.
    This function takes in the complex data (nodes, edges, faces) and returns modified versions of the 'cells1D' and 'cells2D'
    arrays. The modification is an attachment of a minus sign next to any edge or face index that is oriented incoherently with
    respect to its incident face or volume (respectively).
    Parameters
    ----------
    cells0D : np array
        An array whose rows list the coordinates of the nodes.
    cells1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells2D : np array
        An array whose rows list the indices of nodes which make up one face.
    faces_per_volume: int
        The number of faces which make up one volume. The default is set to 4, as both the BCC and FCC structures use only tetrahedra.
    edges_per_face: int
        The number of edges which make up one face. The default is set to 3, as both the BCC and FCC structures use only tetrahedra.

    Returns
    -------
    An array defining the volumes (3-cells) as a collection of faces (2-cells) and an array defining faces as a collection of edges
    (1-cells) with considerations for the relative orientations of p-cells and (p-1)-cells.
    """
    
    """ --- VOLUMES TO FACES --- """
    
    v2f = []  ;  f2e = []
    
    part_v2f_1 = partial(vol_to_faces,
                         cells2D = cells2D,
                         faces_per_volume = faces_per_volume)
    
    part_f2e_1 = partial(face_to_edges,
                         cells1D = cells1D,
                         edges_per_face = edges_per_face)
    
    print("\n\\\\--- 1. Converting volumes to faces ---//\n")
    
    t0 = time.time()
    
    with mp.Pool() as pool:
        for result in pool.map(part_v2f_1, cells3D, chunksize = int(len(cells3D)/os.cpu_count())):
            v2f.append(result)
        
    print(f"Time elapsed: {time.time() - t0} s.\n")
    
    
    """ --- FACES TO EDGES --- """
    
    print("\\\\--- 2. Converting faces to edges ---//\n")
    
    t0 = time.time()
            
    with mp.Pool() as pool:
        for result in pool.map(part_f2e_1, cells2D, chunksize = int(len(cells2D)/os.cpu_count())):
            f2e.append(result)
        
    print(f"Time elapsed: {time.time() - t0} s.\n")
    
    del pool, result, t0
    
    v2f = np.array(v2f).astype(int)  ;  f2e = np.array(f2e).astype(int)
    
    
    """ --- FIND RELATIVE ORIENTATIONS --- """
    
    part_v2f_2 = partial(find_relative_orientations_v2f,
                          cells2D = cells2D,
                          cells0D = cells0D)
    
    part_f2e_2 = partial(find_relative_orientations_f2e,
                          cells1D = cells1D,
                          cells0D = cells0D)
    
    print("\\\\--- 3. Orienting volumes and faces ---//\n")
        
    t0 = time.time()
        
    triplets = []
    
    for i in range(0, len(cells3D)):
        triplets.append([cells3D[i], i, v2f[i]])
        
    v2f = []
    
    with mp.Pool() as pool:
        for result in pool.map(part_v2f_2, triplets, chunksize = int(len(triplets)/os.cpu_count())):
            v2f.append(result)
            
    print(f"Time elapsed: {time.time() - t0} s.\n")
    
    print("\\\\--- 4. Orienting faces and edges ---//\n")
    
    t0 = time.time()
    
    triplets = []
    
    for i in range(0, len(cells2D)):
        triplets.append([cells2D[i], i, f2e[i]])
        
    f2e = []
    
    with mp.Pool() as pool:
        for result in pool.map(part_f2e_2, triplets, chunksize = int(len(triplets)/os.cpu_count())):
            f2e.append(result)
            
    print(f"Time elapsed: {time.time() - t0} s.\n")
    
    del triplets, pool, result, t0
    
    v2f = np.array(v2f).astype(int)  ;  f2e = np.array(f2e).astype(int)
    
    return v2f, f2e


"""
----------------------------------------------------------------------------------------------------------------------------
"""



# def import_complex_data(data_folder: Path):
#     """
#     Parameters
#     ----------
#     data_folder : str
#         The path name of the folder where the data files are located.

#     Returns
#     -------
#     np.array
#         Extracts information about the complex from the data files in data_folder.
#     """

#     # Import cell complex data files
            
#     #directory = os.getcwd() # str type
            
#     #os.chdir(data_folder)
    
#     data = []
                
#     if (data_folder / 'nodes.txt').is_file():
                    
#         with open(data_folder / 'nodes.txt') as file:
#             nodes = np.genfromtxt(file, delimiter = ' ')
            
#         data.append(nodes)
    
#     if (data_folder / 'edges.txt').is_file():
                    
#         with open(data_folder / 'edges.txt') as file:
#             edges = np.genfromtxt(file, delimiter = ' ').astype(int)
            
#         data.append(edges)

#     if (data_folder / 'faces.txt').is_file(): 
                    
#         with open(data_folder / 'faces.txt') as file:
#             faces = np.genfromtxt(file, delimiter = ' ').astype(int)
            
#         data.append(faces)
                
#     if (data_folder / 'faces_slip.txt'):
                    
#         with open(data_folder / 'faces_slip.txt') as file:
#             faces_slip = list(np.genfromtxt(file, delimiter = ' ').astype(int))
            
#         data.append(faces_slip)
        
#     if (data_folder / 'volumes.txt').is_file():
                    
#         with open(data_folder / 'volumes.txt') as file:
#             volumes = np.genfromtxt(file, delimiter = ' ').astype(int)
            
#         data.append(volumes)
        
#     if (data_folder / 'nr_cells.txt'):
                    
#         with open(data_folder / 'nr_cells.txt') as file:
#             nr_cells = list(np.genfromtxt(file, delimiter = ' ').astype(int))
            
#         data.append(nr_cells)
              
#     # os.chdir(directory)
            
#     # return data



# if __name__ == '__main__':
    
#     struc = 'fcc'
#     size = [13,13,13]
#     lattice = np.array([[1,0,0], [0,1,0], [0,0,1]])
#     origin = np.array([0,0,0])
#     multi = True
    
#     # Access the data folder
    
#     cwd = os.getcwd()
#     data_folder = Path(cwd)
#     data_folder = data_folder.parent
#     data_folder = data_folder / "Built Complexes"
#     data_folder = data_folder / "FCC_13x13x13"
    
#     # Retrieve the data
    
#     nodes, edges, faces, faces_slip, volumes, nr_cells = import_complex_data(data_folder)        
    
#     v2f, f2e = get_orientations(cells0D = nodes,
#                                 cells1D = edges,
#                                 cells2D = faces,
#                                 cells3D = volumes,
#                                 faces_per_volume = 4,
#                                 edges_per_face = 3)    
        
#     write_to_file(f2e, 'faces_to_edges',
#                   v2f, 'volumes_to_faces',
#                   new_folder = True)



