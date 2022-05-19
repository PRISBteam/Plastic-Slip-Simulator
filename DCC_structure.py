# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 15:22:00 2022

Last edited on: 04/05/2022 17:10

Author: Afonso Barroso, 9986055

This code intends to build a discrete combinatorial cubic complex and from it obtain all the essential objects needed
to describe its structure, such as incidence matrices and the Laplacian and its spectrum.

"""

# OUTUPUT INTO A FILE
# LAPLACIAN
# readme file
# nr nodes, edges, faces, volumes
# output inc and adj matrices as [row, column, value]
# optimisation?
# check function parameters


# ----- #----- #  IMPORTS # ----- #----- #

import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations # to avoid nested for loops
from scipy import sparse
import time
import matplotlib.pyplot as plt
# from matplotlib.colors import LightSource


# ----- # ----- # ESSENTIALS # ------ # ----- #

# external input file

VALID_STRUCTURES = ['simple cubic', 'bcc', 'fcc', 'hcp']
STRUC = 'bcc'
LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]]) # lattice basis vectors
SIZE = [12] * 3 # nr unit cells in each direction
DIM = 3


# ----- # ----- # FUNCTIONS # ------ # ----- #


def check_input_lattice_vectors(lattice_vector_1, lattice_vector_2, lattice_vector_3, dim):
    
    try:
        
        # This function checks that the lattice basis vectors as given by the user have valid inputs and, if given as blank,
        # take their default values
        
        if dim == 2:
        
            if lattice_vector_1 == []:
                lattice_vector_1 = np.array([[1, 0, 0]])
            else:
                lattice_vector_1 = np.array([lattice_vector_1])
                
            
            if lattice_vector_2 == []:
                lattice_vector_2 = np.array([[0, 1, 0]])
            else:
                lattice_vector_2 = np.array([lattice_vector_2])
                
            lattice_vector_3 = np.array([[0, 0, 0]])
        
        elif dim == 3:
            
            if lattice_vector_1 == []:
                lattice_vector_1 = np.array([[1, 0, 0]])
            else:
                lattice_vector_1 = np.array([lattice_vector_1])
                
            
            if lattice_vector_2 == []:
                lattice_vector_2 = np.array([[0, 1, 0]])
            else:
                lattice_vector_2 = np.array([lattice_vector_2])
                
            
            if lattice_vector_3 == []:
                lattice_vector_3 = np.array([[0, 0, 1]])            
            else:
                lattice_vector_3 = np.array([lattice_vector_3])
                
                
        LATTICE = np.vstack((lattice_vector_1, lattice_vector_2, lattice_vector_3))
            
        return LATTICE

    except:
        
            print("\nWARNING: Something went wrong with the function check_input_lattice_vectors().\n")



def transl_copy(array, vector, nr_copies, axis = None):
    """
    Parameters
    ----------
    array : np array (N x 3)
        An array of points in 3D space.
    vector : np array OR list (1 x 3)
        Defines the direction in which we want to translate-copy the input array.
    nr_copies: int
        Sets the number of times we want to translate-copy the input array.
    axis : int in {0,1,2}
        The base coordinate (x, y or z) along which we want to translate-copy the input array by the amount specified in
        the vector's corresponding component. The default is 'None'.
    
    Returns
    -------
    An array of new (translated) points vstacked onto the input array.
    """
    
    try:
        
        if axis == None:
            
            new_array = np.zeros((1,3)) # Will use this as basis for vstack
            
            # Create a new array with the same # rows as the input array that we'll modify to be the translated points
            
            add_points = np.zeros(( np.size(array[:,0]), 3 )) 
            
            for i in range(1, nr_copies):
                
                add_points[:,0] = array[:,0] + i * vector[0]
                add_points[:,1] = array[:,1] + i * vector[1]
                add_points[:,2] = array[:,2] + i * vector[2]
                
                new_array = np.vstack((new_array, add_points))
                
                # The new_array now contains a point (0,0,0) plus the translated points - all that's left is to delete that
                # first point and add the original array
                
            new_array = np.delete(new_array, 0, 0)
            new_array = np.vstack((array, new_array))
                
        elif axis == 0:
            
            new_array = np.zeros((1,3)) # Will use this as basis for vstack
            
            # Create a new array with the same # rows as the input array that we'll modify to be the translated points
            add_points = np.zeros(( np.size(array[:,0]), 3 )) 
    
            
            for i in range(1, nr_copies):
                
                add_points[:,0] = array[:,0] + i * vector[0]
                add_points[:,1] = array[:,1]
                add_points[:,2] = array[:,2]
                
                new_array = np.vstack((new_array, add_points))
                
                # The new_array now contains a point (0,0,0) plus the translated points - all that's left is to delete that
                # first point and add the original array
                
            new_array = np.delete(new_array, 0, 0)
            new_array = np.vstack((array, new_array))
                
        elif axis == 1:
            
            new_array = np.zeros((1,3)) # Will use this as basis for vstack
            
            # Create a new array with the same # rows as the input array that we'll modify to be the translated points
            add_points = np.zeros(( np.size(array[:,0]), 3 )) 
    
            
            for i in range(1, nr_copies):
                
                add_points[:,0] = array[:,0]
                add_points[:,1] = array[:,1] + i * vector[1]
                add_points[:,2] = array[:,2]
                
                new_array = np.vstack((new_array, add_points))
                
                # The new_array now contains a point (0,0,0) plus the translated points - all that's left is to delete that
                # first point and add the original array
                
            new_array = np.delete(new_array, 0, 0)
            new_array = np.vstack((array, new_array))
                
        elif axis == 2:
            
            new_array = np.zeros((1,3)) # Will use this as basis for vstack
            
            # Create a new array with the same # rows as the input array that we'll modify to be the translated points
            add_points = np.zeros(( np.size(array[:,0]), 3 )) 
    
            
            for i in range(1, nr_copies):
                
                add_points[:,0] = array[:,0]
                add_points[:,1] = array[:,1]
                add_points[:,2] = array[:,2] + i * vector[2]
                
                new_array = np.vstack((new_array, add_points))
                
                # The new_array now contains a point (0,0,0) plus the translated points - all that's left is to delete that
                # first point and add the original array
                
            new_array = np.delete(new_array, 0, 0)
            new_array = np.vstack((array, new_array))
                                
        return new_array

    except:
        
        print("\nWARNING: Something went wrong with the function transl_copy().\n")



def find_equal_rows(array, points):
    """
    Parameters
    ----------
    array : np array (M x N)
        An array of points in 3D space.
    points : np array (L x N)
        An array of points in 3D space

    Returns
    -------
    row_indices : np array
        Returns a (K x 2) array where the first column gives indices of the input 'array' and the second column gives the
        indices of the 'points' array. Paired indices give the rows in each argument that are equal.
        
    DISCLAIMER: this function returns inaccurate results for repeated rows in either 'array' or 'points'.
    """
    # Inspired by Daniel's reply on https://stackoverflow.com/questions/18927475/numpy-array-get-row-index-searching-by-a-row (Accessed 09/02/2022)
    
    try:
    
        row_indices = np.empty((0,2))
        
        for row in points:
        
            if list(row) in array:
                
                try:
            
                    row_index = np.where(np.all(array == row, axis=1))[0][0] # Returns a tuple of arrays, so need to include [0][0]
                                                                             # to get the value we need
                    
                    matching_rows = np.array([[ row_index,
                                                np.where(np.all(points == row, axis=1))[0][0]
                                             ]])
                    
                    row_indices = np.vstack((row_indices, matching_rows))
                
                except:
                    
                    pass
                
            else:
                # print('\nThe array does not contain the point ' + list(row) + '.\n\n')
                pass
            
        return row_indices.astype(int)

    except:
        
        print("\nWARNING: Something went wrong with the function find_equal_rows().\n")



def create_nodes(structure=None, origin=None, lattice=None, size=None, dim=None, axis=None):
    """
    Parameters
    ----------
    structure : str
        A descriptor of the basic structure of the lattice.
    origin : numpy array (1 x 3)
        An array with the spatial coordinates of the origin point for the space.
    lattice : np array OR list (3x3 or 2x3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    size : list
        Lists the number of unit cells in each spatial coordinate.
    dim: int
        The dimension of the space.
    axis: int
        The created lattice lies on the plane perpendicular to the axis, where 0 = x, 1 = y and 2 = z.

    Returns
    -------
    Numpy arrays of positions in 3D space. In all cases the arrays returned include a 'nodes' array which contains all the nodes in the complex. For SC
    structure, that is the only return. For BCC structure, a 3-tuple (nodes_sc, nodes_bcc, nodes_virtual) is also returned. For FCC structure, a 2-tuple
    (nodes_sc, nodes_fcc) is also returned.
    """
    
    try:
        
        ##----- SC -----##
    
        if structure == 'simple cubic':
            
            """ inputs needed: structure, origin, lattice, size, dim """
    
            nodes = transl_copy(transl_copy(transl_copy(origin,
                                                          lattice[0],  size[0] + 1,  axis=0),
                                              lattice[1],  size[1] + 1,  axis=1),
                                  lattice[2],  size[2] + 1,  axis=2)
            
            return nodes
        
        ##----- BCC -----##
            
        elif structure == 'bcc':
            
            """ inputs needed: structure, origin, lattice, size, dim, axis (2D only) """
            
            # For a bcc structure, we build the lattice by starting with the first node, which will be at the origin (0,0,0) and use the
            # transl_copy() function to form the bottom line of the sample. Then, we set a new node at the centre of the first unit cell
            # and use the transl_copy() function to create a second lline of nodes. For the third line, we origin with a node at position
            # (0,0,1) and repeat the procedure we did for the first line. And so on.
    
            if dim == 2:
                
                ax1 = (axis + 1) % 3
                ax2 = (axis + 2) % 3
                            
                # First line:
                
                nodes = transl_copy(origin, lattice[ax1], size[ax1] + 1, axis=ax1)
                
                nodes_bcc = np.empty((0,3))
                
                # Subsequent lines:
                    
                for i in range(1, size[ax2] * 2 + 1): # Need the +1 bc range() is exclusive at the top value
                    
                    # The bottom line of the crystal is done, and now we need to make the subsequent lines of atoms. For each layer of
                    # unit cells (in an axis+1 = const plane), we need to add two new lines, the first corresponding to the proper bcc atoms, and the
                    # second corresponding to the otherwise regular simple cubic atoms
                    #
                    #   i = 4  axis+1 = 2   *   *   *   *   *   *  <-- usual simple cubic atom line
                    #   i = 3                 *   *   *   *   *    <-- proper bcc atom line
                    #   i = 2  axis+1 = 1   *   *   *   *   *   *  <-- usual simple cubic atom line
                    #   i = 1                 *   *   *   *   *    <-- proper bcc atom line
                    #          axis+1 = 0   *   *   *   *   *   *  <-- bottom line
                    
                    
                    start_node = (origin +
                                  (i % 2) * (i - 1) / 2 * lattice[ax2] +
                                  (i % 2) * (lattice[ax1] + lattice[ax2]) / 2 +
                                  ((i + 1) % 2) * i / 2 * lattice[ax2])
                    
                    if i % 2 == 1: # proper bcc nodes
                        
                        new_nodes = transl_copy(start_node, lattice[ax1],  size[ax1],  axis=ax1)
                        
                        nodes_bcc = np.vstack((nodes_bcc, new_nodes))
                        
                    elif i % 2 == 0: # regular simple cubic nodes
                        
                        new_nodes = transl_copy(start_node, lattice[ax1],  size[ax1] + 1,  axis=ax1)
                        
                    nodes = np.vstack((nodes, new_nodes))
                    
                nodes_bcc_ind = find_equal_rows(nodes, nodes_bcc)[:,0]
                    
                nodes_sc = np.delete(nodes, nodes_bcc_ind, axis=0)
                
                return nodes, (nodes_sc, nodes_bcc)
                
            elif dim == 3:
            
                # First sheet:
                
                nodes = transl_copy(transl_copy(origin,
                                                lattice[0], size[0] + 1, axis=0),
                                    lattice[1], size[1] + 1, axis=1)
                
                nodes_bcc = np.empty((0,3))
                
                # Subsequent sheets:
                    
                for i in range(1, size[2] * 2 + 1): # Need the +1 bc range() is exclusive at the top value
                    
                    # The bottom surface of the crystal is done, and now we need to make the subsequent sheets of atoms. For each layer of
                    # unit cells (in a z = const plane), we need to add two new sheets, the first corresponding to the proper bcc atoms, and the
                    # second corresponding to the otherwise regular simple cubic atoms
                    #
                    #   i = 4  z = 2   *   *   *   *   *   *  <-- usual simple cubic atom sheet
                    #   i = 3            *   *   *   *   *    <-- proper bcc atom sheet
                    #   i = 2  z = 1   *   *   *   *   *   *  <-- usual simple cubic atom sheet
                    #   i = 1            *   *   *   *   *    <-- proper bcc atom sheet
                    #          z = 0   *   *   *   *   *   *  <-- bottom surface
                    
                    
                    start_node = (origin +
                                  (i % 2) * (i - 1) / 2 * lattice[2] +
                                  (i % 2) * (lattice[0] + lattice[1] + lattice[2]) / 2 +
                                  ((i + 1) % 2) * i / 2 * lattice[2])
                    
                    if i % 2 == 1: # proper bcc nodes
                        
                        new_nodes = transl_copy(transl_copy(start_node,
                                                              lattice[0],  size[0],  axis=0),
                                                lattice[1],  size[1],  axis=1)
                        
                        nodes_bcc = np.vstack((nodes_bcc, new_nodes))
                        
                    elif i % 2 == 0: # regular simple cubic nodes
                        
                        new_nodes = transl_copy(transl_copy(start_node,
                                                              lattice[0],  size[0] + 1,  axis=0),
                                                lattice[1],  size[1] + 1,  axis=1)
                        
                    nodes = np.vstack((nodes, new_nodes))
                                    
                # Finally, we need the virtual fcc nodes that are necessary to describe the slip systems in bcc.
                
                x, nodes_virtual = create_nodes(structure = 'fcc',         # This function will return an array x for which we have no use,
                                                origin = origin,           # and a tuple (nodes, nodes_sc, nodes_fcc) of which we are only
                                                lattice = lattice,         # interested in the second element.
                                                size = size,
                                                dim = dim)
                
                nodes_virtual = x[nodes_virtual[1]]
                
                nodes = np.vstack((nodes, nodes_virtual))
                
                nodes = np.unique(nodes, axis = 0)
                
                nodes_bcc_ind = find_equal_rows(nodes, nodes_bcc)[:,0]
                nodes_virtual_ind = find_equal_rows(nodes, nodes_virtual)[:,0]
                
                nodes_bcc_and_virtual_ind = np.hstack((nodes_bcc_ind, nodes_virtual_ind))
                    
                nodes_sc = np.delete(nodes, nodes_bcc_and_virtual_ind, axis=0)

                nodes = nodes[nodes[:,0].argsort()]                 # sort by x value
                nodes = nodes[nodes[:,1].argsort(kind='mergesort')] # sort by y value
                nodes = nodes[nodes[:,2].argsort(kind='mergesort')] # sort by z value
                
                # Sorting technique taken from https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted%20by%20a%20single,the%20values%20in%20an%20array%20in%20ascending%20value. (Accessed 08/04/22)

                nodes_sc = list(find_equal_rows(nodes, nodes_sc)[:,0])
                nodes_sc.sort()
                
                nodes_bcc = list(nodes_bcc_ind)
                nodes_bcc.sort()
                
                nodes_virtual = list(nodes_virtual_ind)
                nodes_virtual.sort()                
                
                return nodes, (nodes_sc, nodes_bcc, nodes_virtual)
            
        ##----- FCC -----##
        
        elif structure == 'fcc':
            
            """ inputs needed: structure, origin, lattice, size, dim """
            
            if dim == 2:
                
                pass
            
            elif dim == 3:
            
                # Notice that an fcc structure consists of 3 2D bcc lattices, one in each coordinate direction. We start with the nodes on the
                # faces perpendicular to the z axis, then to the y axis, and finally to the x axis.
                
                z_nodes = create_nodes(structure='bcc', origin=origin, lattice=lattice, size=size, dim=2, axis=2)[0]
                
                z_nodes = transl_copy(z_nodes, lattice[2], size[2] + 1)
                
                y_nodes = create_nodes(structure='bcc', origin=origin, lattice=lattice, size=size, dim=2, axis=1)[0]
                
                y_nodes = transl_copy(y_nodes, lattice[1], size[1] + 1)
        
                x_nodes = create_nodes(structure='bcc', origin=origin, lattice=lattice, size=size, dim=2, axis=0)[0]
                
                x_nodes = transl_copy(x_nodes, lattice[0], size[0] + 1)
        
                nodes = np.vstack((z_nodes, y_nodes, x_nodes))
    
                # These will contain repeated points, so let's delete those.
                
                nodes = np.unique(nodes, axis = 0)
                
                # Now we sort the array by z-value, then y-value, then x-value, so that the indexing order increases along the x-axis, then along the y-axis
                # and finally along the z-axis.
                
                nodes = nodes[nodes[:,0].argsort()]                 # sort by x value
                nodes = nodes[nodes[:,1].argsort(kind='mergesort')] # sort by y value
                nodes = nodes[nodes[:,2].argsort(kind='mergesort')] # sort by z value
                
                # Sorting technique taken from https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted%20by%20a%20single,the%20values%20in%20an%20array%20in%20ascending%20value. (Accessed 08/04/22)
    
                # Finally, we single out the SC and FCC nodes by their indices.
                
                nodes_sc = create_nodes(structure = 'simple cubic', origin = origin, lattice = lattice, size = size, dim = dim)
                
                nodes_fcc = np.delete(nodes, find_equal_rows(nodes, nodes_sc)[:,0], axis=0)
                
                nodes_sc = list(find_equal_rows(nodes, nodes_sc)[:,0])
                                
                nodes_fcc = list(find_equal_rows(nodes, nodes_fcc)[:,0])
    
            return nodes, (nodes_sc, nodes_fcc)

    except:
        
        print("\nWARNING: Something went wrong with the function create_nodes().\n")



def find_neighbours(cells_0D, lattice, structure, dim, special_0D=None):
    """
    Parameters
    ----------
    cells_0D : np array OR list (N x 3)
        The set of points for which we want to create a regular complex.
    lattice : np array OR list (3x3 or 2x3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    structure: str, optional
        A descriptor of the basic structure of the lattice. The default is 'simple cubic'.
    special_0D: tuple of numpy arrays
        Contains numpy arrays containing nodes of different types.
    dim: int
        The spatial dimension of the lattice.

    Returns
    -------
    An array of pairs of indices of the input 'cells_0D' which relate neighbouring points.
    """
    
    try:
    
        # We will proceed by filtering out the pairs of points in the input 'cells_0D' which are farther apart than one lattice
        # constant, in order to obtain the neighbours of each point. Then, calculate the distances between each point and its
        # neighbours to see which are the closest. Finally, we create pairs connecting these points.
    
        ##----- SC -----##
    
        if structure == 'simple cubic':
            
            neighbours = np.empty((0,2))
            
            counter = 1 # This will keep track of the index of the next point as listed in 'cells_0D'
        
            for row in cells_0D:
                
                # Calculate only distances between the current point (row) and all other points which have not been considered
                # yet and are not the very point we're considering (so, all points with indices >= counter).
                
                dist = cdist(
                             np.array([row]),                        # Need 'row' to be a 2D array in cdist()
                             cells_0D[counter : np.shape(cells_0D)[0], :])
                
                # Consider as neighbours only the points which are one lattice constant away. Need to add 'counter' to each
                # element, because the i-th entry in 'dist' is actually the distanece between 'row' and the (i + counter)-th node.
                
                close_points = np.argwhere( dist <= np.max(lattice) )[:,1] + counter
                
                # The array close_points is a list of indices of 'cells_0D' corresponding to points which are up to one lattice
                # constant away from 'row', the point we are considering now. The next step is to pair these indices with the
                # index of 'row', so as to have an organised list of neighbours. The index of the current 'row' will be equal to
                # 'counter' - 1.
                
                # In order to pair the index of the current row with the indices in 'close_points', we need them both to be 3x1 numpy
                # arrays so we can use np.hstack().
                
                index = [counter - 1] * np.size(close_points)
                
                index = np.array(index).reshape(np.size(close_points), 1)
                
                current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
                
                neighbours = np.vstack((neighbours, current_neighbours))
                
                # This way, each row of 'neighbours' will be a horizontal pairing between a point and one of its neighbours.
                            
                counter += 1
                
            return neighbours.astype(int)
                
        ##----- BCC -----##
    
        elif structure == 'bcc':
            
            """ Required: special_0D = (nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_virtual]). """
            
            if dim == 2:
                
                neighbours = np.empty((0,2))
                
                return neighbours.astype(int)
            
            elif dim == 3:
                                        
                # We will go through each type of node and connect it to the appropriate nodes.
                
                # First off, we have SC nodes. All the simple cubic nodes will connect to other simple cubic nodes a maximum distance of one lattice constant away.
                # For that, we follow a similar algorithm to the one above.      
                
                neighbours = np.empty((0,2))
    
                counter = 1 # This will keep track of the index of the next point as listed in special_0D[0]
    
                for point in special_0D[0]: # for an SC point
                    
                    # Calculate only distances between the current point and all other points which have not been considered
                    # yet and are not the very point we're considering (so, all points with indices >= counter).
                    
                    dist = cdist(np.array([point]),                                 # Need 'point' to be a 2D array in cdist()
                                 special_0D[0][counter : np.shape(cells_0D)[0], :])
                    
                    # Consider as neighbours only the points which are one lattice constant away. Need to add 'counter' to each
                    # element, because the i-th entry in 'dist' is actually the distanece between 'row' and the (i + counter)-th node.
                    
                    close_points = np.argwhere( dist <= np.max(lattice) )[:,1] + counter
                    
                    # The array close_points is a list of indices of 'special_0D[0]' corresponding to points which are up to one lattice
                    # constant away from 'point', the point we are considering now. The next step is to pair these indices with the
                    # index of 'point', so as to have an organised list of neighbours. The index of the current 'point' will be equal to
                    # 'counter' - 1.
                    
                    # There's a catch, though, because nothing in life is simple. The indices we are working with are those of the array special_0D[0], but the
                    # indices we really want are relative to the array 'cells_0D'. So, we need to convert them over.
                    
                    # In order to pair the index of the current row with the indices in 'close_points', we need them both to be 3x1 numpy
                    # arrays so we can use np.hstack().
                    
                    index = list(find_equal_rows(cells_0D, np.array([special_0D[0][counter - 1]]))[:,0]) # the inputs of find_equal_rows() must be 2D arrays
                                                                                                   
                    index = index * np.size(close_points)
                    
                    index = np.array(index).reshape(np.size(close_points), 1)
                    
                    close_points = find_equal_rows(cells_0D, special_0D[0][close_points])[:,0]
                    
                    current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
                    
                    neighbours = np.vstack((neighbours, current_neighbours))
                    
                    counter += 1
                    
                # Now we move on to the BCC nodes. These need to be connected to every other node in the unit cell, i.e. every node a maximum distance away equal
                # to half a cubic diagonal.
                
                for point in special_0D[1]: # for a BCC point
                    
                    # We will make use of the geometry of the unit cell in this context and of the index ordering of the nodes as given by the function
                    # create_nodes(). When calculating the distance between the current BCC node and other points, there is no need to cycle through EVERY
                    # other node when we know, in terms of indices relative to the array 'cells_0D', the lowest and highest nodes we may consider for neighbours.
                    
                    min_node_index = find_equal_rows(cells_0D, np.array([point - np.sum(lattice, axis=1) / 2]))[0,0]
                    
                    max_node_index = find_equal_rows(cells_0D, np.array([point + np.sum(lattice, axis=1) / 2]))[0,0]
                    
                    points_considered = cells_0D[min_node_index : max_node_index + 1] # Need a +1 because it is exclusive; note that this will include 'point'
                    
                    dist = cdist(np.array([point]),    # Need 'point' to be a 2D array in cdist()
                                 points_considered)
                
                    # We consider as neighbours only the points which are at most half a cubic diagonal away. But we need to be careful that the column indices
                    # in 'dist' will correspond to indices in points_considered, not indices in 'cells_0D', which is what we want. But we know that the nodes in
                    # points_considered have indices ranging in order from min_node_index to max_node_index, so to the indices below we just need to add the former.
                
                    close_points = np.argwhere( dist <= np.linalg.norm(np.sum(lattice, axis=1)) / 2 )[:,1] + min_node_index
                    
                    # Note that close_points actually includes the very 'point' we are considering, since we did not exclude it from 'dist' above. we just need to
                    # remove it. The index of 'point' in 'cells_0D' is given by:
                    
                    index = list(find_equal_rows(cells_0D, np.array([point]))[:,0]) # the inputs of find_equal_rows() must be 2D arrays
                    
                    # And so we just need to:
                    
                    close_points = np.delete(close_points, np.argwhere(close_points == index)[0], axis=0)
    
                    # In order to pair the index of the current node with the indices in 'close_points', we need them both to be 3x1 numpy
                    # arrays so we can use np.hstack().
                                                                                                   
                    index = index * np.size(close_points)
                    
                    index = np.array(index).reshape(np.size(close_points), 1)
                                    
                    current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
                    
                    neighbours = np.vstack((neighbours, current_neighbours))
                    
                # Lastly, we move on to the virtual FCC nodes. Since the connections between these and the BCC nodes are already included above, we only need to
                # establish new connections between an FCC node and the SC nodes on the same face as itself.
                
                for point in special_0D[2]: # for an FCC point
                    
                        # We consider as neighbours only the SC points which are at most half a square diagonal away. To allow for the lattice basis vectors
                        # to have asymmetrical dimensions, that "half a square diagonal" is actually better understood as half a cubic diagonal multiplied by
                        # the cosine of the angle between (the vector from one of the SC nodes on the same face as the virtual node to the BCC node in the
                        # same unit cell) and (the vector from one of the SC nodes on the same face as the virtual node to the virtual node).
                        
                        # So, the first step is to find an SC node on the same plane as the virtual node we are considering. Note that, if the plane is
                        # of the type xi = const (where xi is a coordinate), then 'point' and the SC nodes will have the same xi coordinate. There will, of course,
                        # be several SC nodes that fulfil this condition, so we simply take the first one.
                        
                        same_plane_SC_node = special_0D[0][np.argwhere(special_0D[0] == point)[0,0]]
                        
                        xi = np.argwhere(special_0D[0] == point)[0,1] # This is the coordinate that stays constant along the plane
                        
                        same_u_cell_BCC_node = point + lattice[xi] / 2
                        
                        vector1 = same_plane_SC_node - same_u_cell_BCC_node ; vector1_norm = np.linalg.norm(vector1)
                        
                        vector2 = same_plane_SC_node - point ; vector2_norm = np.linalg.norm(vector2)
                        
                        max_dist = np.linalg.norm(np.sum(lattice, axis=1)) / 2 * (np.inner(vector1, vector2) / (vector1_norm * vector2_norm))
                        
                        # Now, there is a problem here with precision. The variable max_dist ends up having more precision than the function cdist() below, and so
                        # no points will actually satisfy the condition for close_points. To fix that, we add a little uncertainty.
                        
                        max_dist = max_dist + 0.000000000000001
                        
                        # When calculating the distance function next, we consider only SC nodes.
                        
                        dist = cdist(np.array([point]), # Need 'point' to be a 2D array in cdist()
                                     special_0D[0])
            
                        close_points = np.argwhere( dist <= max_dist )[:,1]
                        
                        # The indices in close_points will be relative to the column indices of 'dist', which are in turn relative to the indice of the SC nodes.
                        # So, we just need to convert them to indices relative to 'cells_0D'.
                        
                        close_points = find_equal_rows(cells_0D, special_0D[0][close_points])[:,0]
                        
                        index = list(find_equal_rows(cells_0D, np.array([point]))[:,0]) # the inputs of find_equal_rows() must be 2D arrays
                                                                                                       
                        index = index * np.size(close_points)
                        
                        index = np.array(index).reshape(np.size(close_points), 1)
                                            
                        current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
                        
                        neighbours = np.vstack((neighbours, current_neighbours))
                                        
                    
                ####### neighbours = np.unique(neighbours, axis = 0) # To prevent repeated entries.
                
                # Now we sort the array by values on the second column and then by values on the first column, so that the index order goes: connections from node 0
                # in order of increasing index of the other endpoint, connections from node 1 in order of increasing index of the other endpoint, etc.
                
                neighbours = np.sort(neighbours.astype(int))
                neighbours = neighbours[neighbours[:,1].argsort()] # sort by value on column 1
                neighbours = neighbours[neighbours[:,0].argsort(kind='mergesort')] # sort by value on column 0
                
                # Sorting technique adapted from https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted%20by%20a%20single,the%20values%20in%20an%20array%20in%20ascending%20value. (Accessed 08/04/22)
    
                neighbours = neighbours.astype(int)
                
                # Finally, we single out the SC, BCC and virtual FCC edges.
                
                sc_neighbours = []; bcc_neighbours = []; virtual_neighbours = [];
                
                for i in range(0, np.shape(neighbours)[0]):
                    
                    # if neighbours[i] connects two SC nodes
                    sc_condition = (list(cells_0D[neighbours[i][0]]) in special_0D[0].tolist() and list(cells_0D[neighbours[i][1]]) in special_0D[0].tolist())
    
                    # elif neighbours[i] is incident on a BCC node
                    bcc_condition = (list(cells_0D[neighbours[i][0]]) in special_0D[1].tolist() or list(cells_0D[neighbours[i][1]]) in special_0D[1].tolist())
                    
                    # elif neighbours[i] is incident on a virtual node (but bcc_condition takes priority)
                    virtual_condition = (list(cells_0D[neighbours[i][0]]) in special_0D[2].tolist() or list(cells_0D[neighbours[i][1]]) in special_0D[2].tolist())
                                    
                    if sc_condition:
                        
                        sc_neighbours.append(i)
                        
                    elif bcc_condition:
                        
                        bcc_neighbours.append(i)
                        
                    elif virtual_condition and not bcc_condition:
                        
                        virtual_neighbours.append(i)
                    
                return neighbours, (sc_neighbours, bcc_neighbours, virtual_neighbours)
    
        ##----- FCC -----##
    
        elif structure == 'fcc':
            
            """ Required input: special_0D = (nodes_sc, nodes_bcc, nodes_fcc). """
            
            neighbours = np.empty((0,2))
            
            if dim == 2:
                                
                return neighbours.astype(int)
            
            elif dim == 3:
                    
                # In the FCC structure, we again need SC, BCC and FCC nodes. The SC nodes will connect to each other and to the nearest
                # FCC nodes; adittionally, the FCC nodes will connect to each other and to the BCC node.
                
                counter = 0 # this will keep track of the *local* index of the node we are considering
                
                # Let's deal with the SC-SC connections first.
                                
                for point in cells_0D[special_0D[0]]:
                    
                    index = find_equal_rows(cells_0D, np.array([point]))[0,0] # this is the *global* index
                    
                    # Calculate only distances between the current point and all other SC nodes which have not been considered
                    # yet and are not the very point we're considering (so, all points with indices >= counter + 1).
                    
                    dist = cdist(np.array([point]),                        # Need 'point' to be a 2D array in cdist()
                                 cells_0D[special_0D[0]][counter + 1 : ])
                    
                    # Consider as neighbours only the points which are one lattice constant away. Need to add 'counter' to each
                    # element, because the i-th entry in 'dist' is actually the distance between 'point' and the (i + counter)-th node
                    # in cells_0D[special_0D[0]].
                    
                    close_points = np.argwhere( dist <= np.max(lattice) )[:,1] + counter + 1
                    
                    # The array close_points is a list of indices of cells_0D[special_0D[0]] corresponding to points which are up to one lattice
                    # constant away from 'point', the point we are considering now. We need to scale these indices back as indices of 'cells_0D'.
                    
                    close_points = find_equal_rows(cells_0D, cells_0D[special_0D[0]][close_points])[:,0]
                    
                    close_points = np.delete(close_points, close_points == index, axis = 0) # to avoid repetitions
                    
                    # The next step is to pair these indices with the index of 'point', so as to have an organised list of neighbours.
                    # The index of the current 'point' will be equal to 'counter'.
                    
                    # In order to pair the index of the current point with the indices in 'close_points', we need them both to be 3x1 numpy
                    # arrays so we can use np.hstack().
                    
                    index = [index] * np.size(close_points)
                    
                    index = np.array(index).reshape(np.size(close_points), 1)
                    
                    current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
                    
                    neighbours = np.vstack((neighbours, current_neighbours))
                    
                    # This way, each row of 'neighbours' will be a horizontal pairing between a point and one of its neighbours.
                                
                    counter += 1
                
                # Now we deal with all other connections.
                
                counter = 0
                
                for point in cells_0D:

                    # We consider as neighbours only the points which are at most half a square diagonal away. To allow for the lattice basis vectors
                    # to have asymmetrical dimensions, we will consider three distinct square diagonals (which are actually rectangular diagonals, but that's
                    # just semantics) and, for simplicity, take the largest one as the neighbour distance cut-off (this asusmes that there isn't a huuuge
                    # difference between the three diagonals).
                    
                    diag1 = np.sqrt(np.linalg.norm(lattice[0]) ** 2 + np.linalg.norm(lattice[1]) ** 2)
                    diag2 = np.sqrt(np.linalg.norm(lattice[1]) ** 2 + np.linalg.norm(lattice[2]) ** 2)
                    diag3 = np.sqrt(np.linalg.norm(lattice[2]) ** 2 + np.linalg.norm(lattice[0]) ** 2)
                    
                    max_dist = np.max(np.array([diag1, diag2, diag3])) / 2

                    # # Now, there is a problem here with precision. The variable max_dist ends up having more precision than the function cdist() below, and so
                    # # no points will actually satisfy the condition for close_points. To fix that, we add a little uncertainty.
                    
                    # max_dist = max_dist + 0.000000000000001
                    
                    # To save on computation, we consider only the distance between the current point and all other nodes that have not been considered yet.
                                            
                    dist = cdist(np.array([point]), # Need 'point' to be a 2D array in cdist()
                                 cells_0D[counter + 1 : ])
        
                    close_points = np.argwhere( dist <= max_dist )[:,1]
                    
                    # Because we only considered the nodes after the current 'point', we need to scale the indices presented in 'close_points' to match the
                    # indices of 'cells_0D'.
                    
                    close_points = close_points + counter + 1

                    # In order to pair the index of the current point with the indices in 'close_points', we need them both to be 3x1 numpy
                    # arrays so we can use np.hstack().
                    
                    index = [counter] * np.size(close_points)
                    
                    index = np.array(index).reshape(np.size(close_points), 1)
                    
                    current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
                    
                    neighbours = np.vstack((neighbours, current_neighbours))
                    
                    counter += 1
                
                # Now we sort the array.
                
                neighbours = neighbours[neighbours[:,1].argsort()] # sort by value on column 1
                neighbours = neighbours[neighbours[:,0].argsort(kind='mergesort')] # sort by value on column 0
                
                # Sorting technique adapted from https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted%20by%20a%20single,the%20values%20in%20an%20array%20in%20ascending%20value. (Accessed 08/04/22)
                    
                neighbours = neighbours.astype(int)
                                    
                # Finally, we differentiate edges connecting two SC nodes, edges connecting two FCC nodes and edges connecting one FCC and one SC node.
                
                sc_neighbours = []; bcc_fcc_neighbours = []; fcc2_neighbours = []; fcc_sc_neighbours = [];
                
                for i in range(0, np.shape(neighbours)[0]):
                    
                    # if neighbours[i] connects two SC nodes
                    sc_condition = (neighbours[i][0] in special_0D[0] and neighbours[i][1] in special_0D[0])
                    
                    # if neighbours[i] connects one BCC and one FCC node
                    bcc_fcc_condition = ((neighbours[i][0] in special_0D[1] and neighbours[i][1] in special_0D[2]) or
                                        (neighbours[i][0] in special_0D[2] and neighbours[i][1] in special_0D[1]))
                    
                    # if neighbours[i] connects two FCC nodes
                    fcc2_condition = (neighbours[i][0] in special_0D[2] and neighbours[i][1] in special_0D[2])
    
                    # if neighbours[i] connects one FCC and one SC node
                    fcc_sc_condition = ((neighbours[i][0] in special_0D[0] and neighbours[i][1] in special_0D[2]) or
                                        (neighbours[i][0] in special_0D[2] and neighbours[i][1] in special_0D[0]))
                    
                    if sc_condition:
                        
                        sc_neighbours.append(i)
                        
                    elif bcc_fcc_condition:
                        
                        bcc_fcc_neighbours.append(i)
                        
                    elif fcc2_condition:
                        
                        fcc2_neighbours.append(i)
                        
                    elif fcc_sc_condition:
                        
                        fcc_sc_neighbours.append(i)
                                            
                return (neighbours, sc_neighbours, bcc_fcc_neighbours, fcc2_neighbours, fcc_sc_neighbours)

        ##----- HCP -----##

        elif structure == 'hcp':
            
            pass
        
    except:
        
        print("\nSomething went wrong with the function find_neighbours().\n")



def degree_distribution(cells_1D, cells_0D):
    """
    Parameters
    ----------
    cells_1D : np array
        An array whose rows list the indices of nodes which make up one edge.
    cells_0D : np array
        An array whose rows list the coordinates of the nodes.

    Returns
    -------
    node_degrees : list
        A list containing the degree of each node; the index in the list corresponds to the index of the node..
    """
    
    node_degrees = []
    
    for i in range(0, np.shape(cells_0D)[0]):
        
        node_degrees.append(np.count_nonzero(cells_1D == i))
        
    return node_degrees



def create_faces(edges, structure, cells_0D=None):
    """
    Parameters
    ----------
    edges : np array (M x 2)
        A numpy array of pairs of indices as given by the create_edges function.
    special_edges: tuple of numpy arrays
        Contains numpy arrays containing edges of different types.
    structure: str, optional
        A descriptor of the basic structure of the lattice.
        
    Returns
    -------
    A numpy array where each row contains the indices of poits in 'array' that delineate a face.
    """
    
    try:
    
        ##----- SC -----##

        if structure == 'simple cubic':
                    
        # Proceed as follows: take a point in an edge in the input 'edges'. Find out if any two neighbours of that point have another
        # neighbour in common (which is not the original point). Then, the face will be described by the original point, those
        # two points and finally the second common neighbour.
        
            faces = np.empty((0,4))
        
            neighbours = np.sort(edges) # Sorts each row from lowest value to highest value (e.g. [10, 1] to [1, 10]).
        
        # Because the first column of 'neighbours' will have several repeated entries in sequence, instead of cycling through
        # each row, we can just cycle through each 'cells_0D' index and find all its neighbours in one go.
            
            # For each point:
    
            for i in range(0, np.max(neighbours[:,1]) + 1):  # Need to add +1 because range() is exclusive at the top value
            
            # We select the rows in 'neighbours', i.e. the edges, that contain that point on the first column.
            
                rows = np.where(neighbours[:,0] == i)
                
                neighbour_points = neighbours[rows][:,1]
                
                # This last array will be a list of the neighbours of point i. Now we need to find the neighbours of those
                # neighbours.
                
                # For each pair of neighbours of point i, we check which ones have a second neighbour in common (which is not i).
                
                for [x,y] in combinations(neighbour_points, 2):
                    
                    # To deconstruct the neighbours of x and y, we jave to work with both columns of 'neighbours', which requires
                    # more tact than the simple procedure above.
                    
                    # First, we identify the rows in 'neighbours' where x and y are mentioned.
                    
                    rows_x = neighbours[np.where(neighbours == x)[0]]
                    
                    rows_y = neighbours[np.where(neighbours == y)[0]]
                    
                    # These arrays contain pairs of x/y and their neighbours, including i. So, secondly, we need to remove any 
                    # mention of i, and then any mention of x/y.
                    
                    rows_x_sans_i = rows_x[np.where(rows_x != i)]
                    
                    neighbours_x = rows_x_sans_i[np.where(rows_x_sans_i != x)]
                    
                    rows_y_sans_i = rows_y[np.where(rows_y != i)]
                    
                    neighbours_y = rows_y_sans_i[np.where(rows_y_sans_i != y)]
                    
                    # Now we need to find the common neighbours of x and y that are not i.
                    
                    common_neighbours = np.intersect1d(neighbours_x, neighbours_y)
                                            
                    if np.size(common_neighbours) != 0: # If there is, in fact, a second common neighbour
                        
                        face = np.array([[ i, x, y, common_neighbours[0] ]]) # Needs to be 2D to vstack
                            
                        faces = np.vstack((faces, face))
                            
                    else:
                        pass
                    
            return faces.astype(int)

        ##----- BCC -----##

        elif structure == 'bcc':
            
            """ Requires input: cells_0D=nodes """
            
            # For a BCC structure, the network of faces is composed solely of triangular faces. The procedure here will be: take an edge, take its
            # endpoints, find which other node(s) have those endpoints as neighbours, make faces.
            
            faces = np.empty((0,3))
            
            for edge in edges:
                
                # First we find the neighbours of the two endpoints.
                
                nb_endpoint_1 = edges[ np.where( edges == edge[0] )[0] ] # "nb" short for neighbours
    
                nb_endpoint_1 = nb_endpoint_1[ nb_endpoint_1 != edge[0] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
                nb_endpoint_1 = nb_endpoint_1[ nb_endpoint_1 != edge[1] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
    
                nb_endpoint_2 = edges[ np.where( edges == edge[1] )[0] ] # "nb" short for neighbours
                
                nb_endpoint_2 = nb_endpoint_2[ nb_endpoint_2 != edge[0] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
                nb_endpoint_2 = nb_endpoint_2[ nb_endpoint_2 != edge[1] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
    
                common_neighbours = np.intersect1d(nb_endpoint_1, nb_endpoint_2)
    
                for point in common_neighbours:
                    
                    new_face = np.array([[edge[0], edge[1], point]]) # Needs to be 2D to vstack
                    
                    faces = np.vstack((faces, new_face))
                            
            # Now we sort the array 'faces' by order of the indices of the constituent nodes.
            
            faces = np.sort(faces)
            
            faces = np.unique(faces.astype(int), axis = 0)
            
            # The last step is to identify those faces which lie on a side of the cubic unit_cell, since they will ultimately not be useful
            # to us when describing slip planes. So, it is useful to keep a record of them just so we can disregard them later. The way we
            # can identify them is by the fact that all three constituent nodes will share one and the same coordinate, since all the cubic
            # faces lie on a plane of the type xi = const., where xi is a coordinate.
            
            faces_sc = []
            
            for face in faces:
                
                centroid = np.average(cells_0D[face], axis=0)
                
                # We want to check if all three points in 'face' have a coordinate value in common with the centroid of the face.
                
                condition = ((cells_0D[face][:,0] == centroid[0]).all() or
                             (cells_0D[face][:,1] == centroid[1]).all() or
                             (cells_0D[face][:,2] == centroid[2]).all())
                
                 # If the above is True, then all cells_0D[face] lie on a plane xi = const., meaning they lie on a side of the cubic unit-cell.
                
                if condition:
                    
                    faces_sc.append(find_equal_rows(faces, np.array([face]))[0,0]) # Need 'face' to be a 2D array.
                
            return faces, faces_sc

        ##----- FCC -----##

        elif structure == 'fcc':
            
            """ Requires input: cells_0D = (nodes, nodes_sc, nodes_bcc, nodes_fcc) """
            
            # Since all faces are triangles, like in BCC, we can take the algorithm for the BCC case and adapt it.
            
            # For an FCC structure, the network of faces is composed solely of triangular faces. The procedure here will be: take an edge, take its
            # endpoints, find which other node(s) have those endpoints as neighbours, make faces.
            
            faces = np.empty((0,3))
            
            faces_slip = []
                        
            for edge in edges:
                
                # First we find the neighbours of the two endpoints.
                
                nb_endpoint_1 = edges[ np.where( edges == edge[0] )[0] ] # "nb" short for neighbours
    
                nb_endpoint_1 = nb_endpoint_1[ nb_endpoint_1 != edge[0] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
                nb_endpoint_1 = nb_endpoint_1[ nb_endpoint_1 != edge[1] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
    
                nb_endpoint_2 = edges[ np.where( edges == edge[1] )[0] ] # "nb" short for neighbours
                
                nb_endpoint_2 = nb_endpoint_2[ nb_endpoint_2 != edge[0] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
                nb_endpoint_2 = nb_endpoint_2[ nb_endpoint_2 != edge[1] ] # to avoid the endpoints of sc_edge appearing in the array of neighbours
    
                common_neighbours = np.intersect1d(nb_endpoint_1, nb_endpoint_2)
    
                for point in common_neighbours:
                    
                    new_face = np.sort(np.array([[edge[0], edge[1], point]]).astype(int)) # Needs to be 2D to vstack
                    
                    if np.size(find_equal_rows(faces, new_face)) == 0: # to avoid repetitions
                    
                        faces = np.vstack((faces, np.sort(new_face))).astype(int)
                        
                        # Here we change the algorithm slightly. We want to keep a log of the faces which will constitute slip planes.
                        # This does not include faces on the sides of the cubic unit cell, and faces on the inside of the inner octahedron.
                        
                        new_face = new_face[0] # Need the slice [0] because new_face was left as a 2D array above
                        
                        # Now, faces on slip plaenes are recognisable by being constituted of 3 FCC nodes, or 2 FCC and 1 SC nodes.
                        
                        condition = ((new_face[0] in cells_0D[3] and new_face[1] in cells_0D[3] and new_face[2] in cells_0D[3])
                                  or (new_face[0] in cells_0D[1] and new_face[1] in cells_0D[3] and new_face[2] in cells_0D[3])
                                  or (new_face[0] in cells_0D[3] and new_face[1] in cells_0D[1] and new_face[2] in cells_0D[3])
                                  or (new_face[0] in cells_0D[3] and new_face[1] in cells_0D[3] and new_face[2] in cells_0D[1]))
                        
                        if condition:
        
                            faces_slip.append(find_equal_rows(faces, np.array([new_face]))[0,0])
                        
                    else:
                        
                        pass
                                                                
            return (faces, faces_slip)

    except:
        
        print("\nWARNING: Something went wrong with the function create_faces().\n")



def create_volumes(lattice, structure, cells_0D=None, cells_2D=None):
    """
    Parameters
    ----------
    cells_0D : np array
        A numpy array whose rows list the spatial coordinates of points.
    lattice : np array OR list (3 x 3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    structure : str, optional
        A descriptor of the basic structure of the lattice.
    special_edges: tuple of numpy arrays
        Contains numpy arrays containing nodes of different types.

    Returns
    -------
    A numpy array whose rows list the indices of points in 3D space that define a 3-cell of a 3D complex.
    """

    try:

        ##----- SC -----##

        if structure == 'simple cubic':
            
            """ Need input 'cells_0D = nodes' """
            
        # The most direct way is to take each node and build the volume for which it is a corner (corresponding to the origin
        # of the unit cell). This does not work for the nodes on surfaces corresponding to xi = L, where xi is a spatial coordinate
        # (such as x1, x2, x3) and L**3 is the size of the (simple) cubic complex, so we will need to remove these.
                
            volumes = np.empty((0, 8))
            
            L1 = np.max(cells_0D[:,0])
            L2 = np.max(cells_0D[:,1])
            L3 = np.max(cells_0D[:,2])
            
            nodes_to_remove = list(np.argwhere(cells_0D[:,0] == L1)[:,0]) # All the nodes on the surface x1 = L1
            
            for i in np.argwhere(cells_0D[:,1] == L2)[:,0]:
                
                nodes_to_remove.append(i) # All the nodes on the surface x2 = L2
                
            for i in np.argwhere(cells_0D[:,2] == L3)[:,0]:
            
                nodes_to_remove.append(i) # All the nodes on the surface x3 = L3
            
            # The list nodes_to_remove will contain some repeated elements corresponding to the corners of the complex. We can
            # remove these easily by making use of sets, which are unordered collections of unique elements.
            # Inspired by poke's reply on https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists (Accessed 16/02/2022)
            
            nodes_to_remove = list(set(nodes_to_remove)); nodes_to_remove.sort()
            
            relevant_nodes = np.delete(cells_0D, nodes_to_remove, axis=0)
            
            # To account for the possibility of a 2D or 1D complex, consider 2 cases:
                
            if np.shape(lattice)[0] == 3:
            
                for node in relevant_nodes:
                    
                    nodes_in_volume = np.vstack((node,
                                                 node + lattice[0],
                                                 node + lattice[1],
                                                 node + lattice[2],
                                                 node + lattice[0] + lattice[1],
                                                 node + lattice[0] + lattice[2],
                                                 node + lattice[1] + lattice[2],
                                                 node + lattice[0] + lattice[1] + lattice[2]))
                    
                    nodes_in_volume_indices = np.sort(find_equal_rows(cells_0D, nodes_in_volume)[:,0]) # Use np.sort() just to organise
                    
                    volumes = np.vstack((volumes, nodes_in_volume_indices))
                    
            elif np.shape(lattice)[0] < 3:
                
                pass # There are no volumes in a 2D complex
        
            return volumes.astype(int)
                
        ##----- BCC -----##

        elif structure == 'bcc':
            
            """ Need input 'cells_2D = faces' """
            
            # In the bcc structure, each unit cell contains 16 volumes which are not "shared" with any other unit cell. They are non-regular
            # tetrahedra, so will contain 4 nodes and 4 faces. There are 16 because each cubic unit cell is divided into 4 equal square pyramids
            # and then those are divided into 4 equal non-regular tetrahedra.
            
            # So, here's how we're gonna do this. We take a face and consider every other face which shares its nodes. Then, we take the nodes of
            # those faces and see which one pops up the most times. recall that faces in this structure are triangles.
                    
            volumes = np.empty((0,4))
            
            counter = 0 # will keep track of which face we are currently on
            
            for face in cells_2D:
                
                nb_faces_0 = np.argwhere(cells_2D == face[0])[:,0]           # first find the faces that have the same node face[0], 
                nb_faces_0 = np.delete(nb_faces_0, nb_faces_0 == counter)    # then remove the current face to avoid issues
                
                nb_faces_1 = np.argwhere(cells_2D == face[1])[:,0]           # first find the faces that have the same node face[0],
                nb_faces_1 = np.delete(nb_faces_1, nb_faces_1 == counter)    # then remove the current face to avoid issues
    
                nb_faces_2 = np.argwhere(cells_2D == face[2])[:,0]           # first find the faces that have the same node face[0],
                nb_faces_2 = np.delete(nb_faces_2, nb_faces_2 == counter)    # then remove the current face to avoid issues
                
                # Now we consider intersections between these 3 arrays. Somewhere in there we will find the 4th node to complete the volume.
                
                int_1 = np.intersect1d(nb_faces_0, nb_faces_1)
                
                int_2 = np.intersect1d(nb_faces_0, nb_faces_2)
                
                int_3 = np.intersect1d(nb_faces_1, nb_faces_2)
                
                last_node = np.hstack((int_1, int_2, int_3)) # gather the 'faces' indices that were left from the intersections
                
                last_node = faces[last_node].reshape(1, np.size(faces[last_node]))[0] # transform into node indices and reshape into one row
                
                last_node = np.delete(last_node, np.where(last_node == face[0])) # remove node face[0] to avoid double counting
                last_node = np.delete(last_node, np.where(last_node == face[1])) # remove node face[1] to avoid double counting
                last_node = np.delete(last_node, np.where(last_node == face[2])) # remove node face[2] to avoid double counting
                
                # What we have left in the array last_node is a collection of points which might be the final 4th node needed to complete
                # the volume. We can identify it by the fact that it will be the most common one in last_node. So,
                
                unique, counts = np.unique(last_node, return_counts=True)
                
                last_node = unique[counts == max(counts)]
                
                for i in range(np.size(last_node)):
                
                    new_volume = np.hstack((face, last_node[i]))
                                
                    if list(find_equal_rows(volumes, np.array([new_volume]))) == []:  # If the volume does not exist already
                        
                        volumes = np.vstack((volumes, new_volume))
                    
                    else: # Otherwise it already exists and so we skip it to avoid repetition
                    
                        pass
                
                counter += 1
                
            volumes = np.sort(volumes)
            
            volumes = np.unique(volumes.astype(int), axis = 0)
            
            return volumes.astype(int)
            
        ##----- FCC -----##

        elif structure == 'fcc':
                        
            # The 3-cells in this structure will be tetrahedra, just like in the BCC case, except in a different configuration.#
            # We can simply use that algorithm again.
            
            volumes = create_volumes(lattice, structure = 'bcc', cells_2D = cells_2D)
            
            return volumes
        
    except:
        
        print("\nSomething went wrong with the function create_volumes().\n")



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



def combinatorial_form(degree, cells_3D, cells_2D, cells_1D, cells_0D, v2f, f2e):
    """
    Parameters
    ----------
    degree: int
        The degree of the desired combinatorial form.
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
    
    """
    
    e2n = cells_1D * [1, -1] # this defines the relative orientation between edges and nodes, where an edge-node pair is considered
                             # to be positively relatively oriented if the edge points away from the node.
    
    """ DEGREE 0 """
    
    if degree == 0:
        
        # In this case, the combinatorial form could simply be the identity form, but it is more interesting to make it so
        # that it also works as an adjacency matrix; that is, the combinatorial form of degree 0 acting on a p-cell C returns
        # the p-cells D,E,F,... that are adjacent to (*share a border with*) C.
        
        ### --- NODES --------------------
        
        # Of course, nodes have no border, so instead in this case we define the "neighbours as nodes that have the same
        # incident edges.
        
        rows_0 = []; cols_0 = []; values_0 = []
        
        for i in range(0, np.shape(cells_0D)[0]): # for each indexed node
            
            nbs = cells_1D[np.where(cells_1D == i)[0]] # selected rows of 'cells_1D' containing i; "nbs" short for "neighbours"
            
            nbs = nbs[nbs != i] # 1D array of the neighbours of 'i'
                                        
            cols_0.extend(list(nbs)) # the cols correspond to the nodes
            
            rows_0.extend([i] * np.size(nbs)) # the rows correspond to the neighbour nodes

            values_0.extend([1] * len(nbs))        

        comb_form_0 = sparse.csc_matrix((values_0, (rows_0, cols_0)),
                                        shape = (np.size(cells_0D[:,0]), np.size(cells_0D[:,0])))
        
        ### --- EDGES --------------------
        
        rows_1 = []; cols_1 = []; values_1 = []
                    
        for i in range(0, np.shape(cells_1D)[0]): # for each indexed edge
        
            # The neighbours of an edge are the edges with which it shares a border. So, picking an edge, we need to find its
            # endpoints and then find which other edges are also incident on them.
            
            nbs = np.argwhere(cells_1D == cells_1D[i][0])[:,0] # edges incident on first endpoint; "nbs" short for "neighbours"
            
            nbs = np.hstack((nbs,
                             np.argwhere(cells_1D == cells_1D[i][1])[:,0])) # edges incident on second endpoint
            
            nbs = np.delete(nbs, np.argwhere(nbs == i)) # remove the current edge: obtain list of neighbours by edge index
            
            rows_1.extend([i] * np.size(nbs))
            
            cols_1.extend(list(nbs))
        
            values_1.extend([1] * len(nbs))        
                    
        comb_form_1 = sparse.csc_matrix((values_1, (rows_1, cols_1)),
                                        shape = (np.size(cells_1D[:,0]), np.size(cells_1D[:,0])))
        
        ### --- FACES --------------------

        rows_2 = []; cols_2 = []; values_2 = []

        for i in range(0, np.shape(cells_2D)[0]): # for each indexed face
        
            # The neighbours of a face are the faces with which it shares a border. So, picking a face, we need to find its
            # constituent edges and then find which other faces also contain those edges.
                            
            nbs = np.argwhere(f2e == f2e[i][0])[:,0] # faces incident on first edge; "nbs" short for "neighbours"
            
            for k in range(1, np.size(f2e[i])): # for each edge in the face
                
                nbs = np.hstack((nbs,
                                 np.argwhere(f2e == f2e[i][k])[:,0])) # faces incident on remaining edges
            
            nbs = np.delete(nbs, np.argwhere(nbs == i)) # remove the current face: obtain list of neighbours by face index
            
            rows_2.extend([i] * np.size(nbs))
            
            cols_2.extend(list(nbs))

            values_2.extend([1] * len(nbs))        
        
        comb_form_2 = sparse.csc_matrix((values_2, (rows_2, cols_2)),
                                        shape = (np.size(cells_2D[:,0]), np.size(cells_2D[:,0])))
        
        ### --- VOLUMES --------------------
        
        rows_3 = []; cols_3 = []; values_3 = []

        for i in range(0, np.shape(cells_3D)[0]): # for each indexed volume
        
        # The neighbours of a volume are the volumes with which it shares a border. So, picking a volume, we need to find its
        # constituent faces and then find which other volumes also contain those faces.
                        
            nbs = np.argwhere(abs(v2f) == abs(v2f[i][0]))[:,0] # volumes incident on first face; "nbs" short for "neighbours"
            
            for k in range(1, np.size(v2f[i])): # for every other face in the volume
                
                nbs = np.hstack((nbs,
                                 np.argwhere(abs(v2f) == abs(v2f[i][k]))[:,0])) # faces incident on remaining edges
            
            nbs = np.delete(nbs, np.argwhere(nbs == i)) # remove the current face: obtain list of neighbours by face index
            
            rows_3.extend([i] * np.size(nbs))
            
            cols_3.extend(list(nbs))

            values_3.extend([1] * len(nbs))        
        
        comb_form_3 = sparse.csc_matrix((values_3, (rows_3, cols_3)),
                                        shape = (np.size(cells_3D[:,0]), np.size(cells_3D[:,0])))
        
        
        combinatorial_form = [comb_form_0, comb_form_1, comb_form_2, comb_form_3]
    
    
    """ DEGREE 1 """
    
    if degree == 1:
        
        ### --- NODES --------------------
        
        # This is a trivial case, as nodes are the least-dimensional cells. The combinatorial form is a matrix of zeros, so to
        # save spave we're not even going to define it.
        
        ### --- EDGES --------------------
        
        # This matrix transforms edges into their constituent nodes, taking into account relative orientations. The rows correspond
        # to nodes and the columns correspond to edges.
                
        cols_1 = list(range(0, np.shape(cells_1D)[0])) * 2
        cols_1.sort()
        
        rows_1 = list(cells_1D.flatten())
        
        values_1 = np.sign(e2n.flatten())
        
        values_1[np.argwhere(values_1 == 0)] = 1
        
        comb_form_1 = sparse.csc_matrix((values_1, (rows_1, cols_1)),
                                        shape = (np.size(cells_0D[:,0]), np.size(cells_1D[:,0])))
        
        ### --- FACES --------------------
        
        # This matrix transforms faces into their constituent edges, taking into account relative orientations. The rows correspond
        # to edges and the columns correspond to faces.
                
        cols_2 = list(range(0, np.shape(cells_2D)[0])) * np.shape(f2e)[1]
        cols_2.sort()
        
        rows_2 = list(abs(f2e).flatten())
        
        values_2 = np.sign(f2e.flatten())
        
        values_2[np.argwhere(values_2 == 0)] = 1
        
        comb_form_2 = sparse.csc_matrix((values_2, (rows_2, cols_2)),
                                        shape = (np.size(cells_1D[:,0]), np.size(cells_2D[:,0])))
        
        ### --- VOLUMES --------------------
        
        # This matrix transforms volumes into their constituent faces, taking into account relative orientations. The rows correspond
        # to faces and the columns correspond to volumes.
                
        cols_3 = list(range(0, np.shape(cells_3D)[0])) * np.shape(v2f)[1]
        cols_3.sort()
        
        rows_3 = list(abs(v2f).flatten())
        
        values_3 = np.sign(v2f.flatten())
        
        values_3[np.argwhere(values_3 == 0)] = 1
        
        comb_form_3 = sparse.csc_matrix((values_3, (rows_3, cols_3)),
                                        shape = (np.size(cells_2D[:,0]), np.size(cells_3D[:,0])))
        
        combinatorial_form = [0, comb_form_1, comb_form_2, comb_form_3]

        
    """ DEGREE 2 """
    
    if degree == 2:
        
        ### --- NODES --------------------
        
        # This is a trivial case, as nodes are the least-dimensional cells. The combinatorial form is a matrix of zeros, so to
        # save space we're not even going to define it.
        
        ### --- EDGES --------------------
        
        # This is a trivial case, as there are no (1 - 2)-dimensional cells. The combinatorial form is a matrix of zeros, so to
        # save space we're not even going to define it.
        
        ### --- FACES --------------------
        
        # This matrix transforms faces into their constituent nodes, taking into account relative orientations. The rows correspond
        # to nodes and the columns correspond to faces.
                
        cols_2 = list(range(0, np.shape(cells_2D)[0])) * np.shape(cells_2D)[1]
        cols_2.sort()
        
        rows_2 = list(cells_2D.flatten())
        
        values_2 = [1] * np.size(cells_2D)
        
        comb_form_2 = sparse.csc_matrix((values_2, (rows_2, cols_2)),
                                        shape = (np.size(cells_0D[:,0]), np.size(cells_2D[:,0])))
        
        ### --- VOLUMES --------------------
        
        # This matrix transforms volumes into their constituent edges, taking into account relative orientations. The rows correspond
        # to edges and the columns correspond to volumes.
        
        if (STRUC == 'bcc' or STRUC == 'fcc'):
            
            v2e = np.empty((0,6)) # Need to find which edges constitute which volumes.
            
        elif STRUC == 'simple cubic':
            
            v2e = np.empty((0,12)) # Need to find which edges constitute which volumes.
        
        for volume in cells_3D:
            
            pairs = np.sort(np.array([i for i in combinations(volume, 2)])) # These represent possible edges in the selected volume
            
            true_pairs = find_equal_rows(cells_1D, pairs).astype(int)
            
            edges_in_volume = np.sort(true_pairs[:,0]) # Gives indices of edges that constitute the volume
            
            v2e = np.vstack((v2e, edges_in_volume)).astype(int)
        
        cols_3 = list(range(0, np.shape(cells_3D)[0])) * np.shape(v2e)[1]
        cols_3.sort()
        
        rows_3 = list(v2e.flatten())
        
        values_3 = [1] * np.size(v2e)
                
        comb_form_3 = sparse.csc_matrix((values_3, (rows_3, cols_3)),
                                        shape = (np.size(cells_1D[:,0]), np.size(cells_3D[:,0])))
        
        combinatorial_form = [0, 0, comb_form_2, comb_form_3]
        
    """ DEGREE 3 """
    
    if degree == 3:
        
        ### --- NODES --------------------
        
        # This is a trivial case, as nodes are the least-dimensional cells. The combinatorial form is a matrix of zeros, so to
        # save space we're not even going to define it.
        
        ### --- EDGES --------------------
        
        # This is a trivial case, as there are no (1 - 3)-dimensional cells. The combinatorial form is a matrix of zeros, so to
        # save space we're not even going to define it.
        
        ### --- FACES --------------------
        
        # This is a trivial case, as there are no (2 - 3)-dimensional cells. The combinatorial form is a matrix of zeros, so to
        # save space we're not even going to define it.
        
        ### --- VOLUMES --------------------
        
        # This matrix transforms volumes into their constituent nodes. The rows correspond to nodes and the columns correspond
        # to volumes.
                
        cols_3 = list(range(0, np.shape(cells_3D)[0])) * np.shape(cells_3D)[1]
        cols_3.sort()
        
        rows_3 = list(cells_3D.flatten())
        
        values_3 = [1] * np.size(cells_3D)
        
        comb_form_3 = sparse.csc_matrix((values_3, (rows_3, cols_3)),
                                        shape = (np.size(cells_0D[:,0]), np.size(cells_3D[:,0])))
        
        combinatorial_form = [0, 0, 0, comb_form_3]

        
    return combinatorial_form



def Laplacian(cobndr_below=None, adj_cobndr=None, adj_cobndr_above=None, cobndr=None):
    
    laplacian = np.matmul(cobndr_below, adj_cobndr) + np.matmul(adj_cobndr_above, cobndr)
    
    return laplacian



def write_to_file(file, *args):
    """
    Parameters
    ----------
    file : .txt file
        A file on which we want to write the output of the code.
    *args : np.array AND str
        The input arguments should be an alternating listing of np.arrays and strings, the first specifying an internal structure
        of the complex, namely nodes, edges, faces, and volumes, and the second simply stating the name of the preceding variable,
        e.g. *args = (nodes, 'nodes', edges, 'edges', faces, 'faces', volumes, 'volumes').

    Returns
    -------
    Several .txt files with the input *args written on them.
    """
    
    # This function will create one .txt file with the *args cleanly presented on it, easy for a human to read, as well as other
    # files, with the *args more messily printed on them, but such that they are easier to read with another Python script, for example.
    
    # First file.
    
    with open(file, 'w') as f1:
            
        for i in range(0, len(args)):
            
            # The point of the *args is that each i%2 == 0 is some variable np.array and each i%2 == 1 is a str, detailing the
            # name of the preceding variable.

            if i %2 == 0:
                
                for row in args[i]:
                    
                    if np.all(row == args[i][0]): # first line
                        
                        f1.write(str(args[i + 1]) + ' = np.array([' + str(row) + ',\n')
                                                                    
                    elif np.all(row == args[i][-1]): # last line
                        
                        f1.write(' ' * (len(args[i + 1]) + 13) + str(row) + '])\n\n')
                        
                    else:
                        
                        f1.write(' ' * (len(args[i + 1]) + 13) + str(row) + ',\n')
                        
            else:
                pass
            
    # Other files.
    
        for i in range(0, len(args)):
            
            # The point of the *args is that each i%2 == 0 is some variable np.array and each i%2 == 1 is a str, detailing the
            # name of the preceding variable.

            if i %2 == 0:
                
                file_name = 't_' + args[i+1] + '.txt'
                
                np.savetxt(file_name, args[i], fmt = '%1.2f', delimiter = ',', header = args[i+1].upper(), comments = '# ')



def graph(cells_0D, structure, azimuth = 8, cells_1D=None, cells_2D=None, text=True):
    """
    Parameters
    ----------
    cells_0D : np array OR tuple of np arrays
        The spatial positions of points in 3D space.
    structure : str, optional
        A descriptor of the basic structure of the lattice.
    cells_1D : numpy array (M x 2) OR tuple of numpy arrays
        An array of edges connecting the points in 'cells_0D'. The default is None.

    Returns
    -------
    Plots the complex with or without edges between the nodes and saves it as a .png file.

    """
    
    def alpha_param(full_array, part_array):
        """
        Parameters
        ----------
        full_array : np array
            The x values of the nodes.
        part_array : np array
            The x values of the nodes of the edge/face we are considering.

        Returns
        -------
        a : float
            The appropriate alpha value for the edge/face we are considering.
        """
        
        max_value = np.max(full_array)
        
        a = np.average(part_array) / max_value
        
        if a == 0:
            
            a = 0.1
            
        elif 0 < a <= 0.9:
            
            a += 0.1
            
        elif a > 1:
                
            a = 1
        
        return a
            

    fig_name = STRUC + '_' + str(np.prod(SIZE))
    
    f_size = (10, 10)
    
    plt.figure(figsize = f_size)
    
    ax = plt.axes(projection = '3d')
        
    #---- SC ----#
    
    if structure == 'simple cubic':
                
        # Plot nodes
        
        ax.view_init(elev = 15, azim = azimuth)
                    
        ax.scatter3D(cells_0D[:,0], cells_0D[:,1], cells_0D[:,2], s = 70, c = 'black')
        
        if cells_1D != None:
            
            fig_name = fig_name + '_edges'

            for edge in cells_1D:
                
                edge = cells_0D[edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], 'green', alpha = alpha_param(edges))
        
    #---- BCC ----#
    
    elif structure == 'bcc':

        """ Required: cells_0D = (nodes, nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_virtual]);
                      cells_1D = (edges, edges[edges_sc], edges[edges_bcc], edges[edges_virtual]) """

        # Plot nodes
                
        ax.view_init(elev = 11, azim = azimuth)
        
        ax.scatter3D(cells_0D[1][:,0], cells_0D[1][:,1], cells_0D[1][:,2], s = 70, c = 'black')     # regular simple cubic nodes
        ax.scatter3D(cells_0D[2][:,0], cells_0D[2][:,1], cells_0D[2][:,2], s = 70, c = 'cyan')      # proper bcc nodes
        ax.scatter3D(cells_0D[3][:,0], cells_0D[3][:,1], cells_0D[3][:,2], s = 70, c = 'green')     # virtual fcc nodes

        # Plot edges
        
        if cells_1D != None:
            
            fig_name = fig_name + '_edges'
            
            for edge in cells_1D[1]: # SC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c='black', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[2]: # BCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c='blue', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[3]: # Virtual FCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], 'g--', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
            
        # Plot faces
        
        try:
            
            if cells_2D.any() != None:
            
                tri = np.empty((0,3))

                fig_name = fig_name + '_faces'
                
                # chosen_faces = [11,  9, 10, 39, 31, 33, 34, 43]
                # chosen_nodes = np.unique(cells_2D[chosen_faces])
                
                chosen_nodes = np.array([0,1,2,3,4,7])
                
                x = cells_0D[0][chosen_nodes][:,0]
                y = cells_0D[0][chosen_nodes][:,1]
                z = cells_0D[0][chosen_nodes][:,2]
                
                points = np.array(tuple(zip(x,y,z)))
                
                triangles = combinations(list(range(0,np.size(x))), 3)
                
                for i in triangles:
                    
                    possible_face = find_equal_rows(cells_0D[0], points[list(i)])[:,0]
                    
                    possible_face = list(find_equal_rows(cells_2D, np.array([np.sort(possible_face)]))[:,0])
                    
                    if possible_face != []: # If the face exists
                        
                        a = np.array([i[0], i[1], i[2]])
                        
                        tri = np.vstack((tri, a))
                    
                    else:
                        
                        pass
                                    
                ax.plot_trisurf(x, y, z, triangles = tri.astype(int), color='pink', edgecolor='purple', alpha=0.5, linewidth=3)
                                        
        except:
            pass
        
    #---- FCC ----#
    
    elif structure == 'fcc':
        
        """ Required: cells_0D = (nodes, nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_fcc]);
                      cells_1D = (edges, edges[edges_sc], edges[edges_bcc_fcc], edges[edges_fcc2], edges[edges_fcc_sc]) """

        # Plot nodes
                
        ax.view_init(elev = 11, azim = azimuth)
        
        ax.scatter3D(cells_0D[1][:,0], cells_0D[1][:,1], cells_0D[1][:,2], s = 70, c = 'black')     # regular simple cubic nodes
        ax.scatter3D(cells_0D[2][:,0], cells_0D[2][:,1], cells_0D[2][:,2], s = 70, c = 'cyan')      # proper bcc nodes
        ax.scatter3D(cells_0D[3][:,0], cells_0D[3][:,1], cells_0D[3][:,2], s = 70, c = 'green')     # virtual fcc nodes

        # Plot edges
        
        if cells_1D != None:
            
            fig_name = fig_name + '_edges'
            
            for edge in cells_1D[1]: # SC-SC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'black', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[2]: # BCC-FCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'blue', ls = '--', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[3]: # FCC-FCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'turquoise', ls = '--', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[4]: # FCC-SC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'green', ls = '-.', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
            
        # Plot faces
        
        try:
                        
            fig_name = fig_name + '_faces'
            
            tri = np.empty((0,3))
            
            chosen_faces = [33, 58, 56, 61, 21, 10, 17, 37]
            
            points = np.unique(cells_2D[chosen_faces])
            
            points = cells_0D[0][points]
                                    
            for face in chosen_faces:
                
                t = cells_0D[0][cells_2D[face]]
                
                t = find_equal_rows(points, t)[:,0]
                
                tri = np.vstack((tri, t))
                
            # ls = LightSource(azdeg = 250, altdeg = 45) (need lightsource=ls below)
                                
            ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                            triangles = tri.astype(int),
                            color='pink', edgecolor='purple', alpha=0.5, linewidth=3,
                            shade=True)
                                        
        except:
            pass
        
        # # Plot volumes
        
        # chosen_volumes = [5, 24, 23]
        
        # for volume in chosen_volumes:
        
        #     trios = np.sort(np.array([i for i in combinations(volume, 3)])) # These represent possible faces in the selected volume
            
        #     true_trios = find_equal_rows(faces, trios).astype(int) # these are the triplets above that do correspond to faces
            
        #     faces_in_volume = true_trios[:,0] # Gives indices of edges that constitute the face
            
        #     volumes_as_faces = np.vstack((volumes_as_faces, faces_in_volume)).astype(int)


    if azimuth != 8:
        
        fig_name = fig_name + str(azimuth)
    
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_xlim([0, np.max(cells_0D[0][:,0])])
    ax.set_ylim([0, np.max(cells_0D[0][:,1])])
    ax.set_zlim([0, np.max(cells_0D[0][:,2])])
    
    if text == True:
        
        for i in range(0, np.shape(cells_0D[0])[0]):
            
            ax.text(cells_0D[0][i,0] + 0.05, cells_0D[0][i,1] + 0.05, cells_0D[0][i,2] + 0.05,
                    str(i), color='black', weight = 'bold', size = 15,
                    zorder = 100)
            
    plt.axis('off')
    
    fig_name = fig_name + '.png'
    plt.savefig(fig_name, dpi=400)
        
    plt.show()
    
    
    
# ----- # ----- # CODE # ----- #----- #



""" CHAPTER 1 : USER INPUTS & BASICS """


# # Ask for user input about the dimensions of the complex.

# DIM = input("\nPlease insert the spatial dimension of the complex (2 or 3). \n\n")

# if DIM == '':  # set default value for dimension
    
#     DIM = 3
    
# else:
    
#     DIM = int(DIM)

# SIZE = input("\nIn the format X Y Z, input the number of unit cells you want in the " +
#                       "x-, y- and z-directions, respectively.\n\n").split()

# if SIZE == []: # set default values for lattice size
    
#     SIZE = [5,5,5]

# SIZE = [int(i) for i in SIZE]


# # Ask for user input about the type of lattice.

# STRUC = input("\nChoose one lattice structure from: simple cubic, bcc, fcc or hcp. \n\n")

# if STRUC == '': # set default value for STRUC
    
#     STRUC = 'bcc'

# if STRUC not in VALID_STRUCTURES:
    
#     raise ValueError("\nfind_neighbours: structure must be one of %r." % VALID_STRUCTURES)
    
# # Inspired by Zero Piraeus's answer on https://stackoverflow.com/questions/25854380/enforce-arguments-to-a-specific-list-of-values (Accessed 16/03/2022)


# # Ask for user input about the lattice basis vectors.

# lattice_vector_1 = input("\nIn the format X Y Z, input the x-, y- and z-components of the first lattice basis " +
#                           "vector. If no values are specified, the default is (1,0,0).\n\n").split()

# lattice_vector_2 = input("\nIn the format X Y Z, input the x-, y- and z-components of the second lattice basis " +
#                           "vector. If no values are specified, the default is (0,1,0).\n\n").split()

# lattice_vector_3 = input("\nIn the format X Y Z, input the x-, y- and z-components of the third lattice basis " +
#                           "vector. If no values are specified, the default is (0,0,1).\n\n").split()

# LATTICE = check_input_lattice_vectors(lattice_vector_1, lattice_vector_2, lattice_vector_3, dim=DIM)

# del lattice_vector_1, lattice_vector_2, lattice_vector_3


t0 = time.time()


# Build the complex by starting with one unit cell and copying it along the three directions using the
# parameters defined above.

first_u_cell = np.array([[0,0,0]]) + np.sum(LATTICE, axis=0) / 2

u_cells = transl_copy(transl_copy(transl_copy(first_u_cell,
                                              LATTICE[0],  SIZE[0],  axis=0),
                                  LATTICE[1],  SIZE[1],  axis=1),
                      LATTICE[2],  SIZE[2],  axis=2)


# ------------------------------------------------------------------------------------------------------------------------------------

""" CHAPTER 2 : BUILDING A SIMPLE CUBIC CRYSTAL STRUCTURE """

""" CHAPTER 2.1: NODES, EDGES, FACES, VOLUMES AND ORIENTATIONS"""


if STRUC == 'simple cubic':
    
    if DIM == 2:
        
        nr_nodes = int((SIZE[0] + 1) * (SIZE[1] + 1))
        
        nr_edges = int(SIZE[0] * (SIZE[1] + 1) +
                       SIZE[1] * (SIZE[0] + 1))

        nr_faces = int(SIZE[0] * SIZE[1])
        
    elif DIM == 3:
        
        nr_nodes = int((SIZE[0] + 1) * (SIZE[1] + 1) * (SIZE[2] + 1))
        
        nr_edges = int(SIZE[0] * (SIZE[1] + 1) * (SIZE[2] + 1) +
                       SIZE[1] * (SIZE[2] + 1) * (SIZE[0] + 1) +
                       SIZE[2] * (SIZE[0] + 1) * (SIZE[1] + 1))
        
        nr_faces = int((SIZE[0] + 1) * SIZE[1] * SIZE[2] +
                       (SIZE[1] + 1) * SIZE[2] * SIZE[0] +
                       (SIZE[2] + 1) * SIZE[0] * SIZE[1])
        
        nr_volumes = int(SIZE[0] * SIZE[1] * SIZE[2])
    
    
    # NODES in SC
    
    first_node = first_u_cell - np.sum(LATTICE, axis=0) / 2
    
    nodes = create_nodes(structure = STRUC, origin = first_node, lattice = LATTICE, size = SIZE, dim = DIM)
        
    t1 = time.time() - t0
    
    print("The time taken to create the point lattice was: {0:1.4e} s.".format(t1))
    
    
    # EDGES in SC
    
    edges = find_neighbours(nodes, LATTICE, structure = STRUC, dim=DIM)
    
    t2 = time.time() - t1 - t0
    
    print("The time taken to create the edges was: {0:1.4e} s.".format(t2))

    edge_node_ratio = np.shape(edges)[0] / np.shape(nodes)[0]
    
    print("The edge-to-node ratio is: ", edge_node_ratio)
    
    node_degrees = degree_distribution(edges, nodes)
    
    
    # FACES in SC
    
    faces = create_faces(edges, structure = STRUC)
    
    faces_as_edges = faces_to_edges(faces, edges, edges_per_face = 4)
    
    t3 = time.time() - t2 - t1 - t0
    
    print("The time taken to create the faces was {0:1.4e} s.".format(t3))
    
    face_node_ratio = np.shape(faces)[0] / np.shape(nodes)[0]
    
    print("The face-to-node ratio is: ", face_node_ratio)
    
    
    # VOLUMES in SC
    
    volumes = create_volumes(LATTICE, STRUC, cells_0D = nodes)
    
    volumes_as_faces = volumes_to_faces(volumes, faces, faces_per_volume = 6)
    
    t4 = time.time() - t3 - t2 - t1 - t0
    
    print("The time taken to create the volumes was {0:1.4e} s.".format(t4))
    
    volume_node_ratio = np.shape(volumes)[0] / np.shape(nodes)[0]
    
    print("The volume-to-node ratio is: ", volume_node_ratio)
    
    volumes_as_faces, faces_as_edges = find_relative_orientations(volumes, faces, edges, nodes,
                                                                  volumes_as_faces, faces_as_edges)



    """ CHAPTER 2.2 : STRUCTURE MATRICES FOR SIMPLE CUBIC COMPLEXES """

    t5 = time.time()
    
    adjacency_matrices = combinatorial_form(0, volumes, faces, edges, nodes, volumes_as_faces, faces_as_edges)
    
    incidence_matrices = combinatorial_form(1, volumes, faces, edges, nodes, volumes_as_faces, faces_as_edges)

    t_matrices = time.time() - t5
    print("The time taken to create the ajacency and incidence matrices was {0:1.4e} s.".format(t_matrices))


# ------------------------------------------------------------------------------------------------------------------------------------

    """ CHAPTER 3 : BUILDING A BODY-CENTRED CUBIC CRYSTAL STRUCTURE """
    
    """ CHAPTER 3.1: NODES, EDGES, FACES, VOLUMES AND ORIENTATIONS"""


elif STRUC == 'bcc':
    
    if DIM == 2:
        
        nr_nodes = int((SIZE[0] + 1) * (SIZE[1] + 1) + np.shape(u_cells)[0])
        
        nr_edges = int(SIZE[0] * (SIZE[1] + 1) +
                       SIZE[1] * (SIZE[0] + 1) +
                       4 * SIZE[0] * SIZE[1])

        nr_faces = int(4 * SIZE[0] * SIZE[1])
        
    elif DIM == 3:
        
        nr_nodes = int((SIZE[0] + 1) * (SIZE[1] + 1) * (SIZE[2] + 1) + np.shape(u_cells)[0])
        
        nr_edges = int(SIZE[0] * (SIZE[1] + 1) * (SIZE[2] + 1) +
                       SIZE[1] * (SIZE[2] + 1) * (SIZE[0] + 1) +
                       SIZE[2] * (SIZE[0] + 1) * (SIZE[1] + 1) +
                       8 * SIZE[0] * SIZE[1] * SIZE[2])
        
        nr_faces = int((SIZE[0] + 1) * SIZE[1] * SIZE[2] +
                       (SIZE[1] + 1) * SIZE[2] * SIZE[0] +
                       (SIZE[2] + 1) * SIZE[0] * SIZE[1] +
                       12 * np.prod(SIZE))
                        
        nr_volumes = int(6 * 4 * SIZE[0] * SIZE[1] * SIZE[2])

    
    # NODES in BCC
        
    first_node = first_u_cell - np.sum(LATTICE, axis=0) / 2
    
    nodes, (nodes_sc, nodes_bcc, nodes_virtual) = create_nodes(structure = STRUC,
                                                               origin = first_node,
                                                               lattice = LATTICE,
                                                               size = SIZE,
                                                               dim = DIM,
                                                               axis = 2)
    
    t1 = time.time() - t0
    
    print("The time taken to create the point lattice was: {0:1.4e} s.".format(t1))
    
    
    # EDGES in BCC
    
    edges, (edges_sc, edges_bcc, edges_virtual) = find_neighbours(nodes, LATTICE,
                                                                  structure = STRUC,
                                                                  dim=DIM,
                                                                  special_0D = (nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_virtual]))
    
    t2 = time.time() - t1 - t0
    
    print("The time taken to create the edges was: {0:1.4e} s.".format(t2))

    edge_node_ratio = np.shape(edges)[0] / np.shape(nodes)[0]
    
    print("The edge-to-node ratio is: ", edge_node_ratio)
    
    node_degrees = degree_distribution(edges, nodes)
    
    
    # FACES in BCC
    
    faces, faces_sc = create_faces(edges, structure = STRUC, cells_0D = nodes)
    
    faces_as_edges = faces_to_edges(faces, edges)

    t3 = time.time() - t2 - t1 - t0
    
    print("The time taken to create the faces was {0:1.4e} s.".format(t3))
    
    face_node_ratio = np.shape(faces)[0] / np.shape(nodes)[0]
    
    print("The face-to-node ratio is: ", face_node_ratio)
    
    
    # VOLUMES in BCC
        
    volumes = create_volumes(LATTICE, STRUC, cells_2D = faces)
    
    volumes_as_faces = volumes_to_faces(volumes, faces)
    
    t4 = time.time() - t3 - t2 - t1 - t0
    
    print("The time taken to create the volumes was {0:1.4e} s.".format(t4))
    
    volume_node_ratio = np.shape(volumes)[0] / np.shape(nodes)[0]
    
    print("The volume-to-node ratio is: ", volume_node_ratio)

    volumes_as_faces, faces_as_edges = find_relative_orientations(volumes, faces, edges, nodes,
                                                                  volumes_as_faces, faces_as_edges)



    """ CHAPTER 3.2 : STRUCTURE MATRICES FOR BODY-CENTRED CUBIC COMPLEXES """

    t5 = time.time()
    
    adjacency_matrices = combinatorial_form(0, volumes, faces, edges, nodes, volumes_as_faces, faces_as_edges)
    
    incidence_matrices = combinatorial_form(1, volumes, faces, edges, nodes, volumes_as_faces, faces_as_edges)

    t_matrices = time.time() - t5
    print("The time taken to create the ajacency and incidence matrices was {0:1.4e} s.".format(t_matrices))


# ------------------------------------------------------------------------------------------------------------------------------------

    """ CHAPTER 4 : BUILDING A FACE-CENTRED CUBIC CRYSTAL STRUCTURE """

    """ CHAPTER 4.1: NODES, EDGES, FACES, VOLUMES AND ORIENTATIONS"""


elif STRUC == 'fcc':
    
    if DIM == 2:
        
        nr_nodes = int((SIZE[0] * 2 + 1) * (SIZE[1] * 2 + 1) - np.shape(u_cells)[0])
        
        nr_edges =  2 * int(SIZE[0] * (SIZE[1] + 1) +
                            SIZE[1] * (SIZE[0] + 1) +
                            2 * np.shape(u_cells)[0])

        nr_faces = int(5 * np.shape(u_cells)[0])
        
        start_node = first_u_cell - (LATTICE[0] + LATTICE[1]) / 2
        
        for i in range(0, SIZE[1] + 1):
            
            new_pckd_sheet = transl_copy(start_node, LATTICE[0]/2, 2 * SIZE[0] + 1)
            
            new_intmd_sheet = transl_copy(start_node + LATTICE[1]/2, LATTICE[0], SIZE[0] + 1)
            
            np.vstack((nodes, new_pckd_sheet, new_intmd_sheet))
        
    elif DIM == 3:
        
        nr_nodes = int((SIZE[0] + 1) * (SIZE[1] + 1) * (SIZE[2] + 1) +
                       SIZE[0] * SIZE[1] * (SIZE[2] + 1) +
                       SIZE[2] * SIZE[0] * (SIZE[1] + 1) +
                       SIZE[1] * SIZE[2] * (SIZE[0] + 1))
        
        nr_edges = int(SIZE[0] * (SIZE[1] + 1) * (SIZE[2] + 1) +
                       SIZE[1] * (SIZE[2] + 1) * (SIZE[0] + 1) +
                       SIZE[2] * (SIZE[0] + 1) * (SIZE[1] + 1) +
                       12 * np.shape(u_cells)[0] +
                       4 * (SIZE[0] * SIZE[1] * (SIZE[2] + 1) +
                            SIZE[2] * SIZE[0] * (SIZE[1] + 1) +
                            SIZE[1] * SIZE[2] * (SIZE[0] + 1)))
        
        nr_faces = int(4 * ((SIZE[0] + 1) * SIZE[1] * SIZE[2] +
                            (SIZE[1] + 1) * SIZE[2] * SIZE[0] +
                            (SIZE[2] + 1) * SIZE[0] * SIZE[1]) +
                       9 * np.shape(u_cells)[0] + 
                       8 * 3 * np.shape(u_cells)[0])
                        
        nr_volumes = int( 8 * # 8 tetrahedra per single unit cell corner
                          8 * # 8 tetrahedra per inner octahedron
                         12 * # 12 tetrahedra per single unit cell edge
                         np.shape(u_cells)[0])


    # NODES in FCC
            
    first_node = first_u_cell - (LATTICE[0] + LATTICE[1] + LATTICE[2]) / 2
    
    nodes, (nodes_sc, nodes_bcc, nodes_fcc) = create_nodes(structure = 'bcc',
                                                           origin = first_node,
                                                           lattice = LATTICE,
                                                           size = SIZE,
                                                           dim = DIM)
    
    t1 = time.time() - t0
    
    print("The time taken to create the point lattice was: {0:1.4e} s.".format(t1))
    
    # EDGES in FCC
    
    edges, edges_sc, edges_bcc_fcc, edges_fcc2, edges_fcc_sc = find_neighbours(nodes,
                                                                               lattice = LATTICE,
                                                                               structure = STRUC,
                                                                               dim = DIM,
                                                                               special_0D = (nodes_sc, nodes_bcc, nodes_fcc))
    
    t2 = time.time() - t1 - t0
    
    print("The time taken to create the edges was: {0:1.4e} s.".format(t2))

    edge_node_ratio = np.shape(edges)[0] / np.shape(nodes[0])[0]
    
    print("The edge-to-node ratio is: ", edge_node_ratio)
    
    node_degrees = degree_distribution(edges, nodes)

    
    # FACES in FCC
    
    faces, faces_slip = create_faces(edges,
                                     structure = STRUC,
                                     cells_0D = (nodes, nodes_sc, nodes_bcc, nodes_fcc))

    faces_as_edges = faces_to_edges(faces, edges)

    t3 = time.time() - t2 - t1 - t0
    
    print("The time taken to create the faces was {0:1.4e} s.".format(t3))
    
    face_node_ratio = np.shape(faces)[0] / np.shape(nodes)[0]
    
    print("The face-to-node ratio is: ", face_node_ratio)
    
    
    # VOLUMES in FCC
    
    volumes = create_volumes(LATTICE, STRUC, cells_2D = faces)
    
    volumes_as_faces = volumes_to_faces(volumes, faces)
    
    t4 = time.time() - t3 - t2 - t1 - t0
    
    print("The time taken to create the volumes was {0:1.4e} s.".format(t4))
    
    volume_node_ratio = np.shape(volumes)[0] / np.shape(nodes[0])[0]
    
    print("The volume-to-node ratio is: ", volume_node_ratio)

    volumes_as_faces, faces_as_edges = find_relative_orientations(volumes, faces, edges, nodes,
                                                                  volumes_as_faces, faces_as_edges)



    """ CHAPTER 4.2 : STRUCTURE MATRICES FOR FACE-CENTRED CUBIC COMPLEXES """

    t5 = time.time()

    adjacency_matrices = combinatorial_form(0, volumes, faces, edges, nodes, volumes_as_faces, faces_as_edges)
    
    incidence_matrices = combinatorial_form(1, volumes, faces, edges, nodes, volumes_as_faces, faces_as_edges)

    t_matrices = time.time() - t5
    print("The time taken to create the ajacency and incidence matrices was {0:1.4e} s.".format(t_matrices))

# ----- # I/O # ----- #


if STRUC == 'simple cubic':
    
    write_to_file('test.txt', nodes, 'nodes', edges, 'edges', faces, 'faces', volumes, 'volumes')
    
elif STRUC == 'bcc':
    
    write_to_file('test.txt', nodes, 'nodes', nodes_bcc, 'nodes_bcc', nodes_virtual, 'nodes_virtual',
                  edges, 'edges', edges_sc, 'edges_sc', edges_bcc, 'edges_bcc', edges_virtual, 'edges_virtual',
                  faces, 'faces', faces_sc, 'faces_sc', faces_as_edges, 'faces_as_edges',
                  volumes, 'volumes', volumes_as_faces, 'volumes_as_faces')
    
elif STRUC == 'fcc':
    
    pass


# ----- # PLOTS # ----- #


# if STRUC == 'bcc':
    
#     cells_0D = (nodes, nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_virtual])
#     cells_1D = (edges, edges[edges_sc], edges[edges_bcc], edges[edges_virtual])
    
# elif STRUC == 'fcc':
    
#     cells_0D = (nodes, nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_fcc])
#     cells_1D = (edges, edges[edges_sc], edges[edges_bcc_fcc], edges[edges_fcc2], edges[edges_fcc_sc])

# graph(cells_0D, STRUC, cells_1D = cells_1D)

# del cells_0D, cells_1D
    

# ----- # CLEAN UP # ----- #


t5 = time.time() - t0
print("\nThe total elapsed time was {0:1.4e} s.".format(t5))

del first_u_cell, first_node
del t0, t1, t2, t3, t4, t5, t_matrices
del volume_node_ratio, face_node_ratio, edge_node_ratio
del nr_nodes, nr_edges, nr_faces, nr_volumes

