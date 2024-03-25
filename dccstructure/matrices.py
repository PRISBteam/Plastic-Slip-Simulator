# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2022

Last edited on: Mar 13 12:00 2024

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions that find the degree node distribution, incidence
matrices, adjacency matrices, and other combinatorial forms, given a cell complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
import time
from functools import partial
from scipy import sparse
from pathlib import Path ; import os
import multiprocessing as mp

import sys
sys.path.append('../')
sys.path.append('./')

from dccstructure.build import check_uniqueness
from dccstructure.iofiles import write_to_file


# ----- # ----- # FUNCTIONS # ------ # ----- #


def node_degree(index: int, cells1D: np.ndarray) -> int:
    """
    Parameters
    ----------
    index : int
        The index of the node in the complex for which one wishes to find the degree.
    cells1D : np array
        An array whose rows list the indices of nodes which make up one edge.

    Returns
    -------
    node_degrees : list
        A list containing the degree of each node; the index in the list corresponds to the index of the node..
    """
            
    node_degree = np.count_nonzero(cells1D == index)
        
    return node_degree




def adjacency_0cells(index: int, cells1D: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    index : int
        The index of the node in the complex for which one wishes to find the neighbouring nodes. A neighbouring node is defined as
        one which is the other endpoint of an edge which is incident on the node 'index'.
    cells1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    
    Returns
    -------
    An array with three columns: the first one lists 'index' repeatedly, the second one lists the adjacent nodes, the third one lists
    the degree of adjacency (=1 for regular cell complexes).
    """
                
    nbs = cells1D[np.where(cells1D == index)[0]] # selected rows of 'cells1D' containing node 'index'; "nbs" short for "neighbours"
    
    nbs = nbs[nbs != index] # 1D array of the neighbours of node 'index'
    
    # Now we need to put the information into the desired format.
    
    output = np.hstack((np.array([index] * len(nbs)).reshape((len(nbs), 1)),
                        nbs.reshape((len(nbs), 1)),
                        np.array([1] * len(nbs)).reshape((len(nbs), 1))))
    
    return output




def adjacency_1cells(index: int, cells1D: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    index : int
        The index of the edge in the complex for which one wishes to find the neighbouring edges. A neighbouring edge is defined as
        one which shares at least one node with the edge 'index'.
    cells1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    
    Returns
    -------
    An array with three columns: the first one lists 'index' repeatedly, the second one lists the adjacent edges, the third one lists
    the degree of adjacency (=1 for regular cell complexes).
    """
                                            
    # The neighbours of an edge are the edges with which it shares a border. So, we need to find its
    # endpoints and then find which other edges are also incident on them.
    
    nbs = np.argwhere(cells1D == cells1D[index][0])[:,0] # edges incident on first endpoint; "nbs" short for "neighbours"
    
    nbs = np.hstack((nbs,
                     np.argwhere(cells1D == cells1D[index][1])[:,0])) # adding the edges incident on second endpoint
    
    nbs = np.delete(nbs, np.argwhere(nbs == index)) # remove the current edge: obtain list of neighbours by edge index
                                                
    # Now we need to put the information into the desired format.
    
    output = np.hstack((np.array([index] * len(nbs)).reshape((len(nbs), 1)),
                        nbs.reshape((len(nbs), 1)),
                        np.array([1] * len(nbs)).reshape((len(nbs), 1))))
    
    return output




def adjacency_2cells(index: int, faces_to_edges: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    index : int
        The index of the face in the complex for which one wishes to find the neighbouring faces. A neighbouring face is defined as
        one which shares at least one edge with the face 'index'.
    faces_to_edges : np array
        An array whose row indices correspond to face indices and which, for each face, lists the indices of the edges on its boundary.
        
    Notes
    -----
    The array passed to 'faces_to_edges' must contain only positive integers.
    
    Returns
    -------
    An array with three columns: the first one lists 'index' repeatedly, the second one lists the adjacent faces, the third one lists
    the degree of adjacency (=1 for regular cell complexes).
    """

    # The neighbours of a face are the faces with which it shares a border. So, we need to find its
    # constituent edges and then find which other faces also contain those edges.
    
    faces_to_edges = np.copy(np.abs(faces_to_edges))
    
    nbs = np.argwhere(faces_to_edges == faces_to_edges[index][0])[:,0] # faces incident on first edge; "nbs" short for "neighbours"
    
    for k in range(1, np.size(faces_to_edges[index])): # for each edge in the face
        
        nbs = np.hstack((nbs,
                         np.argwhere(faces_to_edges == faces_to_edges[index][k])[:,0])) # faces incident on remaining edges
    
    nbs = np.delete(nbs, np.argwhere(nbs == index)) # remove the current face: obtain list of neighbours by face index
                                                
    # Now we need to put the information into the desired format.
    
    output = np.hstack((np.array([index] * len(nbs)).reshape((len(nbs), 1)),
                        nbs.reshape((len(nbs), 1)),
                        np.array([1] * len(nbs)).reshape((len(nbs), 1))))
    
    return output




def adjacency_3cells(index: int, volumes_to_faces: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    index : int
        The index of the volume in the complex for which one wishes to find the neighbouring volumes. A neighbouring volume is
        defined as one which shares at least one face with the volume 'index'.
    volumes_to_faces : np array
        An array whose row indices correspond to volume indices and which, for each volume, lists the indices of the faces on its
        boundary.
        
    Notes
    -----
    The array passed to 'volumes_to_faces' must contain only positive integers.
    
    Returns
    -------
    An array with three columns: the first one lists 'index' repeatedly, the second one lists the adjacent faces, the third one lists
    the degree of adjacency (=1 for regular cell complexes).
    """
    
    # The neighbours of a volume are the volumes with which it shares a border. So, picking a volume, we need to find its
    # constituent faces and then find which other volumes also contain those faces.
    
    # Take the first listed face in the boundary of the volume 'index'.
    
    volumes_to_faces = np.copy(np.abs(volumes_to_faces))
    
    nbs = np.argwhere(volumes_to_faces == volumes_to_faces[index][0])[:,0] # volumes incident on first face; "nbs" short for "neighbours"
    
    # Then consider the remaining faces.
    
    for k in range(1, np.size(volumes_to_faces[index])): # for every remaining face in the volume
        
        nbs = np.hstack((nbs,
                         np.argwhere(volumes_to_faces == volumes_to_faces[index][k])[:,0])) # faces incident on remaining edges
    
    nbs = np.delete(nbs, np.argwhere(nbs == index)) # remove the current face: obtain list of neighbours by face index
            
    # Now we need to put the information into the desired format.
    
    output = np.hstack((np.array([index] * len(nbs)).reshape((len(nbs), 1)),
                        nbs.reshape((len(nbs), 1)),
                        np.array([1] * len(nbs)).reshape((len(nbs), 1))))
    
    return output




def incidence_1cells(cell: tuple) -> np.ndarray:
    """
    Parameters
    ----------
    cell : tuple (index, edge)
        An ordered pair whose first value is the index of the edge in the complex for which one wishes to find the nodes on which
        it is incident and whose second element is an array listing the indices of NODES which make up the EDGE, WITHOUT
        considerations for relative orientations.
        
    Returns
    -------
    A list of the indices of the nodes on which 'edge' is incident.
    """
    
    # The array edge aready carries information about which nodes the cell is incident on, so we only need to give it orientation
    # considerations and put the information into the desired format.
    
    # We define an edge-node pair to be coherently oriented if the edge points away from the node.
    
    nodes = np.hstack((cell[1].reshape((len(cell[1]),1)), np.array([[1],[-1]])))
                
    # Now we need to put the information into the desired format.
    
    output = np.hstack((np.array([cell[0]] * len(nodes)).reshape((len(nodes), 1)),
                        nodes))
    
    return output




def incidence_2cells(cell: tuple) -> np.ndarray:
    """
    Parameters
    ----------
    cell : tuple (index, face_as_edge)
        An ordered pair whose first value is the index of the face in the complex for which one wishes to find the edges on which
        it is incident and whose second element is an array listing the indices of EDGES which make up the FACE, WITH considerations
        for relative orientations.
        
    Returns
    -------
    A list of the indices of the edges on which 'face' is incident.
    """
    
    # The array face aready carries information about which nodes the cell is incident on, so we only need to give it orientation
    # considerations and put the information into the desired format.
    
    output = np.empty((0,3))
    
    for edge in cell[1]:
        
        row = np.array([[cell[0], np.abs(edge), np.sign(edge)]])
        
        # We have to consider the fact that 'edge' might be 0, in which case np.sign(0) = 0 but we would want a 1 there.
        
        if row[0,2] == 0: row[0,2] = 1
        
        output = np.vstack((output, row))
    
    return output.astype(int)




def incidence_3cells(cell: tuple) -> np.ndarray:
    """
    Parameters
    ----------
    cell : tuple (index, volume_as_face)
        An ordered pair whose first value is the index of the volume in the complex for which one wishes to find the faces on which
        it is incident and whose second element is an array listing the indices of FACES which make up the VOLUME, WITH considerations
        for relative orientations.
        
    Returns
    -------
    A list of the indices of the faces on which 'volume' is incident.
    """
    
    # The array face aready carries information about which nodes the cell is incident on, so we only need to give it orientation
    # considerations and put the information into the desired format.
    
    output = np.empty((0,3))
    
    for face in cell[1]:
        
        row = np.array([[cell[0], np.abs(face), np.sign(face)]])
        
        # We have to consider the fact that 'face' might be 0, in which case np.sign(0) = 0 but we would want a 1 there.
        
        if row[0,2] == 0: row[0,2] = 1
        
        output = np.vstack((output, row))
    
    return output.astype(int)




def get_matrices(cells0D: np.ndarray,
                 cells1D: np.ndarray,
                 cells2D: np.ndarray,
                 cells3D: np.ndarray,
                 faces_to_edges: np.ndarray,
                 volumes_to_faces: np.ndarray):
    """
    This function summarises the entire "matrices" part of the DCCStructure package.
    This function takes in the complex data (nodes, edges, faces, volumes) as well as data on the relative orientations (faces_to_edges,
    volumes_as_faces) and returns the adjacency matrices of degrees 0, 1, 2 and 3, the adjacency matrices of degrees 1, 2 and 3, as well
    as a list of the node degrees by node index.
    
    Parameters
    ----------
    cells0D : np array
        An array whose rows list the coordinates of the nodes.
    cells1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells2D : np array
        An array whose rows list the indices of nodes which make up one face.
    cells3D : np array
        
    faces_to_edges : np array
        An array whose row indices correspond to face indices and which, for each face, lists the indices of the edges on its boundary.
    volumes_to_faces : np array
        An array whose row indices correspond to volume indices and which, for each volume, lists the indices of the faces on its
        boundary.

    Returns
    -------
    The adjacency matrices of degrees 0, 1, 2 and 3, the adjacency matrices of degrees 1, 2 and 3, as well as a list of the node
    degrees by node index.
    """

    
    A0 = [] ; A1 = [] ; A2 = [] ; A3 = [] ; B1 = [] ; B2 = [] ; B3 = []
    
    part_A0 = partial(adjacency_0cells, cells1D = cells1D)
    part_A1 = partial(adjacency_1cells, cells1D = cells1D)
    part_A2 = partial(adjacency_2cells, faces_to_edges = np.abs(faces_to_edges))
    part_A3 = partial(adjacency_3cells, volumes_to_faces = np.abs(volumes_to_faces))
    
    part_B1 = partial(incidence_1cells)
    part_B2 = partial(incidence_2cells)
    part_B3 = partial(incidence_3cells)
    
    status = []
    
    """ --- ADJACENCY MATRICES --- """
    
    print("\n\\\\--- 1. Assembling adjacency matrices ---//\n")
    
    t0 = time.time()
    
    with mp.Pool() as pool:
        for result in pool.map(part_A0, list(range(len(cells0D))), chunksize = int(len(cells0D)/os.cpu_count())):
            A0.extend(result)
        
    with mp.Pool() as pool:
        for result in pool.map(part_A1, list(range(len(cells1D))), chunksize = int(len(cells1D)/os.cpu_count())):
            A1.extend(result)
            
    with mp.Pool() as pool:
        for result in pool.map(part_A2, list(range(len(cells2D))), chunksize = int(len(cells2D)/os.cpu_count())):
            A2.extend(result)
    
    with mp.Pool() as pool:
        for result in pool.map(part_A3, list(range(len(cells3D))), chunksize = int(len(cells3D)/os.cpu_count())):
            A3.extend(result)
            
    print(f"Time elapsed: {time.time() - t0} s.\n")
    
    del pool, result, t0
        
    """ --- INCIDENCE MATRICES --- """
    
    print("\\\\--- 2. Assembling incidence matrices ---//\n")
    
    t0 = time.time()
    
    cells = []
    for i in range(len(cells1D)): 
        cells.append([i, cells1D[i]])
    
    with mp.Pool() as pool:
        for result in pool.map(part_B1, cells, chunksize = int(len(cells)/os.cpu_count())):
            B1.extend(result)
    
    cells = []
    for i in range(len(faces_to_edges)):
        cells.append([i, faces_to_edges[i]])
    
    with mp.Pool() as pool:
        for result in pool.map(part_B2, cells, chunksize = int(len(cells)/os.cpu_count())):
            B2.extend(result)
            
    cells = []
    for i in range(len(volumes_to_faces)):
        cells.append([i, volumes_to_faces[i]])
            
    with mp.Pool() as pool:
        for result in pool.map(part_B3, cells, chunksize = int(len(cells)/os.cpu_count())):
            B3.extend(result)
                
    print(f"Time elapsed: {time.time() - t0} s.\n")
    
    del cells, pool, result, t0
    
    A0 = np.array(A0)
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    B1 = np.array(B1)
    B2 = np.array(B2)
    B3 = np.array(B3)
    
    status = [check_uniqueness(A0),
              check_uniqueness(A1),
              check_uniqueness(A2),
              check_uniqueness(A3),
              check_uniqueness(B1),
              check_uniqueness(B2),
              check_uniqueness(B3)]
    
    if np.all(status):
        
        print('-> SUCCESS!\n')
        status.append(True)
        
    else:
        
        matrices = np.array(['A0','A1','A2','A3','B1','B2','B3'])
        matrices = np.delete(matrices, status)
        print(f'The matrices {matrices} did not compute correctly.\n')
        print('-> FAILURE!\n')
        
    node_degrees = []
    for j in range(len(cells0D)):
        node_degrees.append(node_degree(j, cells1D))
        
    return A0, A1, A2, A3, B1, B2, B3, node_degrees


def convert_to_cscmatrix(array):
    """
    Parameters
    ----------
    array : np array (N x 3)
        An array whose columns list the row, column and value entries of a sparse matrix.

    Returns
    -------
    The sparse matrix defined by 'array' but in sparse.csc_matrix format.
    """
    
    sparse_m = sparse.csc_matrix((array[:,2], (array[:,0], array[:,1])), shape = (np.max(array[:,0])+1, np.max(array[:,1])+1))
    
    return sparse_m





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
        
#     if (data_folder / 'volumes_to_faces.txt'):
                    
#         with open(data_folder / 'volumes_to_faces.txt') as file:
#             v2f = np.genfromtxt(file, delimiter = ' ').astype(int)
            
#         data.append(v2f)
        
#     if (data_folder / 'faces_to_edges.txt'):
                    
#         with open(data_folder / 'faces_to_edges.txt') as file:
#             f2e = np.genfromtxt(file, delimiter = ' ').astype(int)
            
#         data.append(f2e)
    
#     return data



# if __name__ == '__main__':
    
#     struc = 'bcc'
#     size = [13,13,13]
#     lattice = np.array([[1,0,0], [0,1,0], [0,0,1]])
#     origin = np.array([0,0,0])
#     multi = True
    
    # # Access the data folder
    
    # cwd = os.getcwd()
    # data_folder = Path(cwd)
    # data_folder = data_folder.parent
    # data_folder = data_folder / "Built Complexes"
    
    # if struc == 'bcc':
    #     data_folder = data_folder / "BCC_13x13x13"
        
    # elif struc == 'fcc':
    #     data_folder = data_folder / "FCC_13x13x13"
    
    # del cwd
    
    # # Retrieve the data
    
    # nodes, edges, faces, faces_slip, volumes, nr_cells, v2f, f2e = import_complex_data(data_folder)
    
    # # Process the data
    
#     A0, A1, A2, A3, B1, B2, B3, node_degrees = get_matrices(nodes, edges, faces, volumes, f2e, v2f)
    
#     write_to_file(A0, 'A0',
#                   A1, 'A1',
#                   A2, 'A2',
#                   A3, 'A3',
#                   B1, 'B1',
#                   B2, 'B2',
#                   B3, 'B3',
#                   node_degrees, 'node_degrees',
#                   new_folder = True)
