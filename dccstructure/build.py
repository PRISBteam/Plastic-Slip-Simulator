# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2022

Last edited on: Oct 20 20:08 2024

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure_mp package. In here you will find functions pertinent to the construction of the cell complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations # to avoid nested for loops
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm


# ----- # ----- # FUNCTIONS # ------ # ----- #



def transl_copy(array, vector, nr_copies = 1, multiprocess = False):
    """
    Builds a larger array consisting of the elements of the input 'array' translated by the input 'vector'. In the case of 'nr_copies' > 1
    repeated translations are done by increasing the factor of 'vector' by one.
    
    Parameters
    ----------
    array : np array (N x 3)
        An array of points in 3D space. If N = 1 array still needs to be 2D, e.g. test = np.array([[1,0,0]]) would be a 2D 1x3 array.
    vector : np array OR list (1 x 3)
        Defines the direction in which we want to translate-copy the input array.
    nr_copies: int
        Sets the number of times we want to translate-copy the input array.
    multiprocess: bool
        Whether or not to use the version of transl_copy() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.
    
    Notes
    -----
    This function works with row vectors only. It is assumed that each row in the input 'array' is a row vector in a 3D space.
    
    Examples
    --------
        >> a = np.array([1,1,1])
        >> v = np.array([0.5, -1, 1])
        
        >> new1 = transl_copy(a, v, 1)
        >> new2 = transl_copy(a, v, 3)
        
        new1 = np.array([[1, 1, 1], [1.5, 0, 2]])
        new2 = np.array([[1, 1, 1], [1.5, 0, 2], [2, -1, 3], [2.5, -2, 4]])
            
    
    Returns
    -------
    An array of new (translated) points vstacked onto the input array.
    """
    
    # Need to make sure that 'array' is at least a 2D array.
    
    try:
        
        test = array[:,0]
        
        del test
        
    except:
        
        array = np.array([array])
        
    # Convert input 'vector' into array to avoid complications
    
    if type(vector) == list:
        
        vector = np.array(vector)
        
    
    if multiprocess == False:
        
        # We start with the input 'array' and vstack translated copies of it to it.
        
        new_array = np.copy(array)
                
        for i in range(1, nr_copies + 1): # need the +1 because range() is exclusive at the top value
            
            add_points = array + i * vector
            
            new_array = np.vstack((new_array, add_points))
            
        return new_array
    
    
    elif multiprocess == True:
        
        partition = int(nr_copies / mp.cpu_count())
        
        remainder = nr_copies % mp.cpu_count()
                
        starts = []
        
        for i in range(mp.cpu_count()):
            
            starts.append(array + vector * i * partition)
                            
        if remainder > 0:
            
            part_transl_copy_1 = partial(transl_copy,
                                         vector = vector,
                                         nr_copies = partition - 1, # the -1 is needed to avoid repetition
                                         multiprocess = False)
            
            part_transl_copy_2 = partial(transl_copy,
                                         vector = vector,
                                         nr_copies = remainder,
                                         multiprocess = False)
                    
            new_array = np.empty((0, np.shape(array)[1]))
    
            with mp.Pool() as pool:
                                
                for result in pool.map(part_transl_copy_1, starts):
                                        
                    new_array = np.vstack((new_array, result))
                        
            new_array = np.vstack((new_array, part_transl_copy_2(array + vector * mp.cpu_count() * partition)))
            
        elif remainder == 0:
            
            part_transl_copy_1 = partial(transl_copy,
                                         vector = vector,
                                         nr_copies = partition - 1,
                                         multiprocess = False)
            
            new_array = np.empty((0, np.shape(array)[1]))
    
            with mp.Pool() as pool:
                                
                for result in pool.map(part_transl_copy_1, starts):
                                        
                    new_array = np.vstack((new_array, result))
                
            new_array = np.vstack((new_array, array + vector * mp.cpu_count() * partition))
                                        
        return new_array




def find_equal_rows(points, array, multiprocess = False):
    """
    Parameters
    ----------
    array : np array (M x N)
        An array of points in 3D space. If M = 1 array still needs to be 2D, e.g. test = np.array([[1,0,0]]) would be a 2D 1x3 array.
    points : np array (L x N)
        An array of points in 3D space. Needs to be at least 2D.
    multiprocess: bool
        Whether or not to use the version of find_equal_rows() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.
    Returns
    -------
    row_indices : np array
        Returns a (K x 2) array where the first column gives indices of the 'points' array and the second column gives the
        indices of the input 'array'. Paired indices give the rows in each argument that are equal.
        
    Notes
    -----
    This function is more efficient when the smallest input is passed to "points".
    
    DISCLAIMER: this function returns inaccurate results for repeated rows in either 'array' or 'points'.
    """
    # Inspired by Daniel's reply on https://stackoverflow.com/questions/18927475/numpy-array-get-row-index-searching-by-a-row (Accessed 09/02/2022)
        
    row_indices = np.empty((0,2))
    
    if len(points.shape) < 1:
        points = np.array([points])
        
    if len(array.shape) < 1:
        array = np.array(array)
    
    if multiprocess == False:
        
        # We want to find the row indices in 'array' of rows that correspond to rows in 'points'.
        
        for row in points:
            
            # For each row in 'points', if that row is equal to a row in 'array'...
        
            if list(row) in array:
                
                try:
                    
                    # Then, find the row index in 'array' of the row in 'points'.
            
                    row_index = np.where(np.all(array == row, axis=1))[0][0] # Returns a tuple of arrays, so need to include [0][0]
                                                                             # to get the value we need
                    
                    # And put together the row index in 'points' of the row we are considering presently with the index found above.
                    
                    matching_rows = np.array([[np.where(np.all(points == row, axis=1))[0][0] ,
                                               row_index]])
                    
                    # Attach to the result: [row in 'points', row in 'array'].
                    
                    row_indices = np.vstack((row_indices, matching_rows))
                
                except:
                    
                    pass
                
            else:
                # print('\nThe array does not contain the point ' + list(row) + '.\n\n')
                pass
            
    elif multiprocess == True:
            
            partition = int(points.shape[0] / mp.cpu_count())
            
            remainder = points.shape[0] % mp.cpu_count()
            
            batches = []
            
            for i in range(mp.cpu_count()):
                
                batches.append(points[i*partition : (i+1)*partition])
            
            if remainder > 0:
                
                batches.append(points[mp.cpu_count()*partition :])
                
            row_indices = np.empty((0,2))
            
            # Remember that the function tries to find a row in 'array' that matches each row in 'points'.
            
            part_find_equal_rows = partial(find_equal_rows,
                                           array = array,
                                           multiprocess = False)
                
            with mp.Pool() as pool:
                
                for result in pool.map(part_find_equal_rows, batches):
                    
                    row_indices = np.vstack((row_indices, result))
                    
            # Now because the input 'points' was broken up into batches, the indices in the first column of row_indices start counting
            # from zero again at the start of every batch. So, we need to shift the numbers by 'partition' every 'partition' rows.
            
            for n in range(len(batches)):
                
                row_indices[n*partition : (n+1)*partition , 0] += n*partition
    
    return row_indices.astype(int)




def worker_sort(array):
    """
    Parameters
    ----------
    array : numpy array
        A numpy array to be sorted along the second, first and zeroth axes, respectively in order of priority.

    Returns
    -------
    array : numpy array
        The input array sorted along the second, first and zeroth axes, respectively in order of priority.
    
    Notes
    -----
    Based on J.J's reply on https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/2828121#2828121
    """
        
    # The first sort does not need to be stable so no 'kind' argument is passed, which makes it use a quicksort algorithm. Subsequent
    # sorts need to be stable to preserve the results from the preceeding sorts. The mergesort algorithm is fast and stable. An algorithm
    # being stable means that equal values are not reordered.
    
    array = array[array[:,0].argsort()]                 # sort along the x axis
    array = array[array[:,1].argsort(kind='mergesort')] # sort along the y axis
    array = array[array[:,2].argsort(kind='mergesort')] # sort along the z axis
    
    return array




def nr_nodes(size, structure, dim = 3):
    
    if dim == 2:
        
        size = [size[0], size[1], 0]
    
    if structure == "simple cubic":
        
        n = (size[0]+1)*(size[1]+1)*(size[2]+1)
        
    elif structure == 'fcc':
        
        """ For a complex with size L, M, N in the x, y and z directions, respectively (which is to say, e.g. L
        computational unit cells in the x direction), there is a node at the centre of each computational unit
        cell (CUC), at the centre of each CUC face, at the centre of each CUC edge and each CUC vertex. In essence,
        each CUC is divided into eight smaller Simple Cubic CUCs. """
        
        n = nr_nodes([size[0]*2, size[1]*2, size[2]*2], 'simple cubic', dim = 3)
        
    elif structure == "bcc":
        
        n = ((size[0]+1)*(size[1]+1)*(size[2]+1) +
              size[0]*size[1]*size[2] +
              size[0]*size[1]*(size[2]+1) + size[0]*(size[1]+1)*size[2] + (size[0]+1)*size[1]*size[2])
        
    return n


def nr_edges(size, structure, dim = 3):
    
    if dim == 2:
        
        size = [size[0], size[1], 0]
        
    if structure == "simple cubic":
        
        n = ((size[0]+1)*(size[1]+1)*size[2] + (size[0]+1)*size[1]*(size[2]+1) + size[0]*(size[1]+1)*(size[2]+1))
        
    elif structure == "bcc":
        
        n = ((8+6)*size[0]*size[1]*size[2] +
             4*((size[0]+1)*size[1]*size[2] + size[0]*(size[1]+1)*size[2] + size[0]*size[1]*(size[2]+1)) +
             ((size[0]+1)*(size[1]+1)*size[2] + (size[0]+1)*size[1]*(size[2]+1) + size[0]*(size[1]+1)*(size[2]+1)))
        
    elif structure == "fcc":
        
        """ Each FCC computational unit cell is divided into eight smaller Simple Cubic CUCs. In addition to the
        edges established in this SC complex, the nodes at the centres of the faces of the original CUC are connected
        to the nodes on the vertices of that face and to the nodes at the centres of adjacent faces. """
        
        n = (nr_edges([size[0]*2, size[1]*2, size[2]*2], 'simple cubic', dim = 3) + 
             4 * ((size[0]+1)*size[1]*size[2] + size[0]*(size[1]+1)*size[2] + size[0]*size[1]*(size[2]+1)) +
             12 * np.prod(size))
        
    return n


def nr_faces(size, structure, dim = 3):
    
    if dim == 2:
        
        size = [size[0], size[1], 0]

    if structure == "simple cubic":
        
        n = size[0]*size[1]*(size[2]+1) + size[2]*size[0]*(size[1]+1) + size[1]*size[2]*(size[0]+1)
        
    elif structure == "bcc":
        
        n = (4 * nr_faces(size, 'simple cubic') +
             6 * 6 * size[0]*size[1]*size[2])
        
    elif structure == "fcc":
        
        """ Each face of the original computational unit cell is divided into 8 triangles. Additionally, each Cartesian
        plane crossing the centre of the original CUC is divided into 8 triangles. Finally, each vertex of the original
        CUC has one tetrahedron radiating from it, giving 4 faces per vertex of the original CUC. """
        
        n = (8 * nr_faces(size, 'simple cubic', dim = 3) +
             (8*3 + 8*4) * np.prod(size))
        
    return n


def nr_faces_slip(size, structure, dim=3):
    
    if dim == 2:
        
        size = [size[0], size[1], 0]

    if structure == "simple cubic":
        
        n = nr_faces(size, structure)
        
    elif structure == "bcc":
        
        n = 6 * 6 * np.prod(size)
        
    elif structure == "fcc":
        
        n = 4 * 8 * np.prod(size)
        
    return n


def nr_volumes(size, structure, dim = 3):
    
    if dim == 2:
        
        size = [size[0], size[1], 0]

    if structure == "simple cubic":
        
        n = np.prod(size)
        
    elif structure == "bcc":
        
        n = np.prod(size) * 4 * 6
        
    elif structure == "fcc":
        
        """ Each edge of the original CUC has two tetrahedra attached to it. Additionally, each vertex of the original
        CUC has one tetrahedron associated with it. Finally, the inner octagon is divided into eight tetrahedra. """
        
        n = np.prod(size) * (2*12 + 8 + 8)
        
    return n


def check_uniqueness(array):
    
    _ , test = np.unique(array, return_counts = True, axis = 0)
    
    if len(np.argwhere(test != 1)) == 0:
        
        return True
        
    else:
        
        return False





def create_nodes(origin, lattice, size, structure, dim = 3, axis = None, multiprocess = False):
    """
    This function creates an array of nodes determined by their spatial Cartesian coordinates which constitute the nodes of a
    cell complex representing a particular crystal structure. The expected number of nodes is given by the funcion nr_nodes().
    
    Parameters
    ----------
    origin : numpy array (1 x 3)
        An array with the spatial coordinates of the origin point for the space. These are the coordinates of the centre of a unit cell at a corner
        of the complex.
    lattice : np array OR list (3x3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    size : list
        Lists the number of unit cells in each spatial coordinate.
    structure : str
        A descriptor of the basic structure of the lattice.
    dim: int
        The dimension of the space. If 2D, the space is still considered to be embedded in a 3D space, so that all inputs should be given with 3
        columns. The default is 3.
    axis: int
        Relevant for 2D lattices. The created lattice lies on the plane perpendicular to the axis, where 0 = x, 1 = y and 2 = z. The default is None.
    multiprocess: bool
        Whether or not to use the version of create_nodes() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.

    Returns
    -------
    Numpy arrays of positions in 3D space. In all cases the arrays returned include a 'nodes' array which contains all the nodes in the complex. For SC
    structure, that is the only return. For BCC structure, a 3-tuple (nodes_sc, nodes_bcc, nodes_virtual) is also returned. For FCC structure, a 2-tuple
    (nodes_sc, nodes_fcc) is also returned.
    """
        
    origin = origin - np.sum(lattice, axis=0) / 2  # the first node is not actually placed at the origin.
    
    # The 'simple cubic' case is easy to adapt to multiprocessing because it is solely based on the transl_copy() function.      
        
    ##----- SC 2D & 3D - with & w/o MP -----##

    if structure == 'simple cubic' and dim == 3:
        
        """ inputs needed: structure, origin, lattice, size """
        
        # The predicted number of nodes is (size[0] + 1) * (size[1] + 1) * (size[2] + 1)
        
        nodes = transl_copy(origin, lattice[0],  size[0], multiprocess = multiprocess)
        
        nodes = transl_copy(nodes, lattice[1], size[1], multiprocess = multiprocess)
        
        nodes = transl_copy(nodes, lattice[2], size[2], multiprocess = multiprocess)
                    
    elif structure == 'simple cubic' and dim == 2:
        
        # axis = ax0
        
        ax1 = (axis + 1) % 3  ;  ax2 = (axis + 2) % 3
        
        displ = lattice[axis] / 2
        
        # This means that if the 2D lattice is meant to lie on the xy-plane, then axis = 2 = z makes ax1 = 0 = x and ax2 = 1 = y
        
        nodes = transl_copy(origin + displ, lattice[ax1],  size[ax1], multiprocess = multiprocess)
        
        nodes = transl_copy(nodes, lattice[ax2], size[ax2], multiprocess = multiprocess)
            
    
    
    ##----- BCC 3D - with & w/o MP -----##
    
    elif structure == 'bcc' and dim == 3:
                
        """ In 3D, the BCC complex needs a node structure that joins the FCC and BCC lattice positions, i.e. in a cubic unit cell, we need nodes at every
        corner, at every centre of a face and at the centre of the cubic cell. If you imagine such a lattice in a Cartesian coordinate system, there are
        two types of 'layers'.
        For example, consider such a lattice with primitive vectors [1,0,0], [0,1,0] and [0,0,1]. We have the following structure:
    
                              *   *   *   *   *   *  ---> y
                                *   *   *   *   *
        At z = half-integer:  *   *   *   *   *   *
                                *   *   *   *   *
                              *   *   *   *   *   *
                              ¦
                              v
                              x
        
                            *   *   *   *   *   ---> y    (i=4)
                          * * * * * * * * * * *           (i=3)
        At z = integer:     *   *   *   *   *             (i=2)
                          * * * * * * * * * * *           (i=1)
                            *   *   *   *   *             (i=0)
                          ¦
                          v
                          x
        
        For a size = [L, M, N], the expected number of nodes is (L+1)(M+1)(N+1) + LMN + LM(N+1) + L(M+1)N + (L+1)MN.
        
        To build the first type of layer (half-integer z), we can superpose two 2D simple cubic lattices which are displaced by [0.5, 0.5, 0],
        and the second has size -= 1.
        
        When calling create_nodes() again, one must be careful to compensate the 'origin' parameter in regards to the displacement computed at the start of the function
        and at the start of the 'simple cubic' case!
        
        Creating multiprocessing pools is expensive and sometimes not worth the trouble. So, we use multiprocessing only at the end, to assemble the whole complex.
        
        Another step we must consider is keeping a record of the which nodes are of which type. We want to create lists of indices of cells0D that group together nodes
        of different character in what regards their relative arrangement in the 3-complex. We shall separate them into three categories: a 'SC' category of the nodes
        on the corners of unit cells, a 'FCC' category for the nodes on the centres of the faces of unit cells, and a 'BCC' category for the nodes on the
        centres of unit cells. """
        
        sc1 = create_nodes(structure = 'simple cubic',
                           lattice = lattice,
                           size = size,
                           origin = origin + np.sum(lattice, axis=0) / 2 - lattice[2] / 2,
                           dim = 2,
                           axis = 2,
                           multiprocess = False)
        
        displ = np.sum(lattice, axis=0) / 2  ; displ[2] = 0
        
        # nodes_sc = np.vstack((nodes_sc, sc1))
        
        sc2 = create_nodes(structure = 'simple cubic',
                           lattice = lattice,
                           size = [size[0] - 1, size[1] - 1, size[2] - 1],
                           origin = origin + np.sum(lattice, axis=0) / 2  - lattice[2] / 2 + displ,
                           dim = 2,
                           axis = 2,
                           multiprocess = False)
        
        # nodes_fcc = np.vstack((nodes_fcc, sc2))
    
        layer_1 = np.vstack((sc1, sc2)) ; del sc1, sc2, displ
            
    # The second type of layer (integer z) requires a more hands-on approach. We will need to define each row of lattice points
    # separately. In the picture above, we see that the first row on the top starts with the origin displaced by [0, 0.5, 0.5]
    # relative to the origin of the whole 3D complex and has size as passed to the function (i.e. the number of cells). The second
    # row from the top has the origin displaced by [0.5, 0, 0.5] and size = 2 * size + 1.
            
        layer_2 = np.empty((0,3))
        
        displ_1 = np.sum(lattice, axis=0) / 2  ; displ_1[0] = 0
        
        displ_2 = np.sum(lattice, axis=0) / 2  ; displ_2[1] = 0
        
        for x in range(2 * size[1] + 1):  # 2 * size[1] + 1 is the total number of rows in layer of type 2
            
            if x % 2 == 0: # these are the rows that are like the top one
                
                new_nodes = transl_copy(array = origin + displ_1,
                                        vector = lattice[1],
                                        nr_copies = size[0] - 1, # need -1 because size[0] already gives one too many nodes: note that nr_copies is literally the number of copies, not the total number of resulting points
                                        multiprocess = False)
                
                layer_2 = np.vstack((layer_2, new_nodes))
                
                # nodes_fcc = np.vstack((nodes_fcc, new_nodes))
                
            elif x % 2 == 1: # these are the rows that are like the second one from the top
                
                new_nodes = transl_copy(array = origin + displ_2,
                                        vector = lattice[1] / 2,
                                        nr_copies = 2 * size[0], # here we don't add +1 for the same reason as above, size[0] already gives one too many nodes
                                        multiprocess = False)

                layer_2 = np.vstack((layer_2, new_nodes))
                
                # nodes_fcc = np.vstack((nodes_fcc))
                
                # For every two rows we iterate over, we need to shift the origin by one unit length along the x-axis
                
                origin = origin + lattice[0]
                
        del new_nodes, displ_1, displ_2
                
        # Lastly, we just need to copy each of these layers in the z-direction a number of times equal to size[2] + 1 for the layers of
        # the first type and size[2] for the layers of the second type. Without multiprocessing, this is the most computationally
        # expensive stage of the function.
                
        layer_1 = transl_copy(array = layer_1,
                              vector = lattice[2],
                              nr_copies = size[2],
                              multiprocess = multiprocess)
        
        layer_2 = transl_copy(array = layer_2,
                              vector = lattice[2],
                              nr_copies = size[2] - 1,
                              multiprocess = multiprocess)
        
        nodes = np.vstack((layer_1, layer_2)) ; del layer_1, layer_2
                
        # Now we just sort the rows of 'nodes' by increasing order of the x-coordinate value, followed by the y-coordinate value, followed
        # by the z-coordinate value. Multiprocessing cannot be applied reliably here, because the sorting algorithm needs access to the whole
        # input array at once.
        
        nodes = worker_sort(nodes)


    ##----- FCC 3D - with & w/o MP -----##
    
    elif structure == 'fcc' and dim == 3:
                
        """ In 3D, the FCC complex needs a node structure that joins the FCC and BCC lattice positions, i.e. in a cubic unit cell, we need nodes at every
        corner and at every centre of a face. To achieve spatial homogeneity for the support volumes of 2-cells and the number of neighbours of 3-cells,
        we also require nodes at the centre of every edge of the original computation unit cell. Therefore, the FCC node structure is actually a double SC
        node structure.
        
        When calling create_nodes() again, one must be careful to compensate the 'origin' parameter in regards to the displacement computed at the start of the function
        and at the start of the 'simple cubic' case!
        
        Creating multiprocessing pools is expensive and sometimes not worth the trouble. So, we use multiprocessing only at the end, to assemble the whole complex.
        
        Another step we must consider is keeping a record of which nodes are of which type. We want to create lists of indices of cells0D that group together nodes
        of different character in what regards their relative arrangement in the 3-complex. We shall separate them into three categories: a 'SC' category of the nodes
        on the corners of unit cells, a 'FCC' category for the nodes on the centres of the faces of unit cells, and a 'BCC' category for the nodes on the
        centres of unit cells. """
        
        return create_nodes(origin = origin + np.sum(lattice, axis=0) / 4,
                            lattice = lattice/2,
                            size = [size[0]*2, size[1]*2, size[2]*2],
                            structure = 'simple cubic',
                            dim = 3,
                            axis = None,
                            multiprocess = multiprocess)

    
    ##----- BCC & FCC 2D - w/o MP -----##
    
    elif structure == 'bcc' and dim == 2:
        
    # Coincidentally, this is the same structure as the layers of type 1 the 3D case.
    
        sc1 = create_nodes(structure = 'simple cubic',
                           lattice = lattice,
                           size = size,
                           origin = origin + np.sum(lattice, axis=0) / 2, # have to compensate for the shift already computed at the start
                           dim = 2,
                           axis = axis)
        
        displ = np.sum(lattice, axis=0) / 2  ; displ[2] = 0
        
        sc2 = create_nodes(structure = 'simple cubic',
                           lattice = lattice,
                           size = [size[0] - 1, size[1] - 1, size[2] - 1],
                           origin = origin + np.sum(lattice, axis=0) / 2 + displ,
                           dim = 2,
                           axis = axis)
    
        layer_1 = np.vstack((sc1, sc2)) ; del sc1, sc2, displ
        
        return layer_1
    
    
    elif structure == 'fcc' and dim == 2:
        
    # This case is similar to the layers of type 2 in the 3D case.

        print("\nSorry, the case FCC 2D in create_nodes() isn't supported yet.\n")
    
    
    return nodes




def label_nodes(cells0D, origin, lattice, size, multiprocess = False):
    """
    This function takes an array of nodes of a cell complex and creates lists of its row indices that correspond to nodes in a simple
    cubic, BCC or FCC position in the complex structure.

    Parameters
    ----------
    cells0D : np array OR list (N x 3)
        The set of points for which we want to create a regular complex.
    origin : numpy array (1 x 3)
        An array with the spatial coordinates of the origin point for the space. These are the coordinates of the centre of a unit cell at a corner
        of the complex.
    lattice : np array OR list (3x3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    size : list
        Lists the number of unit cells in each spatial coordinate.
    multiprocess: bool
        Whether or not to use the version of label_nodes() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.

    Returns
    -------
    nodes_sc : list
        List of row indices of the input 'cells0D' that correspond to simple cubic-like positions.
    nodes_bcc : list
        List of row indices of the input 'cells0D' that correspond to BCC-like positions.
    nodes_fcc : list
        List of row indices of the input 'cells0D' that correspond to FCC-like positions.
    """
        
    # This function applies to FCC and BCC structures only.
    
    # We want to create lists of indices of cells0D that group together nodes of different character in what regards their relative
    # arrangement in the 3-complex. We shall separate them into four categories: a 'SC' category of the nodes on the corners of original
    # computational unit cells, a 'FCC' category for the nodes on the centres of the faces of original CUCs, a 'BCC' category for
    # the nodes on the centres of original CUCs, and a 'EC' category for the nodes on the centres of the edges of the original CUCs.
    
    SC = create_nodes(origin = origin,
                      lattice = lattice,
                      size = size,
                      structure = 'simple cubic',
                      dim = 3,
                      multiprocess = multiprocess)
    nodes_sc = find_equal_rows(points = SC,
                               array = cells0D,
                               multiprocess = multiprocess)
    nodes_sc = nodes_sc[:,1].tolist()
    
    ECx = create_nodes(origin = origin + lattice[0]/2,
                       lattice = lattice,
                       size = [size[0]-1, size[1], size[2]],
                       structure = 'simple cubic',
                       dim = 3,
                       multiprocess = multiprocess)
    ECy = create_nodes(origin = origin + lattice[1]/2,
                       lattice = lattice,
                       size = [size[0], size[1]-1, size[2]],
                       structure = 'simple cubic',
                       dim = 3,
                       multiprocess = multiprocess)
    ECz = create_nodes(origin = origin + lattice[2]/2,
                       lattice = lattice,
                       size = [size[0], size[1], size[2]-1],
                       structure = 'simple cubic',
                       dim = 3,
                       multiprocess = multiprocess)
    nodes_ec = find_equal_rows(points = np.vstack((ECx, ECy, ECz)),
                               array = cells0D,
                               multiprocess = multiprocess)
    nodes_ec = sorted(nodes_ec[:,1].tolist())
    
    BCC = create_nodes(origin = origin + np.sum(lattice, axis=0) / 2,
                       lattice = lattice,
                       size = [size[0] - 1, size[1] - 1, size[2] - 1],
                       structure = 'simple cubic',
                       dim = 3,
                       multiprocess = multiprocess)
    nodes_bcc = find_equal_rows(points = BCC,
                                array = cells0D,
                                multiprocess = multiprocess)
    nodes_bcc = nodes_bcc[:,1].tolist()
    
    nodes_fcc =  np.delete(np.arange(cells0D.shape[0]), nodes_sc + nodes_ec + nodes_bcc).tolist()
    
    return nodes_sc, nodes_bcc, nodes_fcc, nodes_ec




def partition_nodes(cells0D, size, lattice, special0D):
    """
    This function takes the nodes of a cell complex whose number of unit cells in each canonical direction is specified by 'size' and
    breaks it up (partitions it) into blocks that are only one unit cell high (layer). This works because the edge network of a unit
    cell in a multi-cell complex is self-contained (shared edges are only at the boundaries of unit cells).
    
    Example:
    
    Input complex:          Output partitioned complex(es):
        
    *   *   *   *   *       2. *   *   *   *   *
      *   *   *   *              *   *   *   *
    *   *   *   *   *          *   *   *   *   * --- 1. *   *   *   *   *
      *   *   *   *                                       *   *   *   *
    *   *   *   *   *       0. *   *   *   *   * ------ *   *   *   *   *
      *   *   *   *              *   *   *   *
    *   *   *   *   *          *   *   *   *   *
    
    Parameters
    ----------
    cells0D : np array (N x 3)
        An array listing the coordinates of points in 3D space that make up the nodes of a cell complex.
    size: list
        Lists the number of unit cells in each spatial coordinate.
    lattice : np array OR list (3x3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    special0D: tuple of numpy arrays
        Contains numpy arrays defining points in 3D space which are nodes of different types.
        ( Required: special0D = (nodes_sc, nodes_bcc, nodes_fcc) )
    
    Returns
    -------
    blocks : list of np arrays
        Each element is a Nx3 array that gives the coordinates of the nodes that constitute one of the partitioning blocks.
    blocks_sc : list
        Lists the indices of the nodes of simple cubic type inside each block (equal for all blocks).
    blocks_bcc : list
        Lists the indices of the nodes of BCC type inside each block (equal for all blocks).
    blocks_fcc : list
        Lists the indices of the nodes of FCC type inside each block (equal for all blocks).
        
    Limitations
    -----------
    As is written, this function only allows for a diagonal 'lattice' parameter (10/01/2023),
    also apparently the number of unit cells in the z-direction must be the greatest (30/01/2023).
    """
    
    ax = np.argmax(size)
    
    if size[2] == size[ax]:
        
        ax = 2
    
    blocks = []
    
    # The total number of planes of atoms as one walks along the 'ax' direction is size[ax] * 2 - 1;
    # Each layer of unit cells comprises 3 planes of atoms, with the detail that the first and third planes are shared with the previous and next layers, respectively;
    # The very first plane of atoms is defined by the coordinates ax = min(cells0D[:,ax])
    # With this in mind, we iterate over every other plane of atoms and build blocks with that plane, the previous one and the next one.
    # The previous and next plane of atoms are defined by a decrease and an increase of lattice[ax,ax] / 2 in the 'ax' coordinate.
    
    for i in tqdm(range(0, size[ax] * 2 - 1, 2), desc = 'Partitioning nodes'):
        
        # We want to single out the nodes in the i-th layer of unit cells
    
        b0 = np.argwhere(cells0D[:,ax] == (i - 1) * lattice[ax,ax] / 2)[:,0]
        b1 = np.argwhere(cells0D[:,ax] == i * lattice[ax,ax] / 2)[:,0]
        b2 = np.argwhere(cells0D[:,ax] == (i + 1) * lattice[ax,ax] / 2)[:,0]
        
        b = np.vstack((cells0D[list(b0)], cells0D[list(b1)], cells0D[list(b2)]))
        b = worker_sort(b)
        blocks.append(b)
        
    # Now, each block will contain only some nodes_sc, nodes_bcc and nodes_fcc, which we must find and convert their global indices
    # into block-specific indices. However, assuming the complex has a regular parallelipiped shape, notice that each layer i of unit
    # cells has the exact same structure, so we only need to determine these indices for one block, namely the last as we exit the for-loop.
        
    blocks_sc = find_equal_rows(b, cells0D[special0D[0]])[:,0]
    
    blocks_bcc = find_equal_rows(b, cells0D[special0D[1]])[:,0]
    
    blocks_fcc = find_equal_rows(b, cells0D[special0D[2]])[:,0]
        
    return blocks, blocks_sc.tolist(), blocks_bcc.tolist(), blocks_fcc.tolist()




""" The following 4 functions are to be used specifically in the function create_edges(). """


def sc2ec_edges(nodes, nbrs, cells0D, lattice):

    """ This function will be used in create_edges() for any 3D structure. 
        It establishes edges between any two nodes that form a regular hexahedral grid.
    """
        
    counter = 1 # This will keep track of the index of the next point as listed in 'nodes_sc_and_ec'
        
    for point in tqdm(nodes, desc = 'Creating edges', miniters = 1000): # for an SC or EC point
        
        # Calculate only distances between the current point and all other points which have not been considered
        # yet and are not the very point we're considering (so, all points with indices >= counter).
        
        dist = cdist(np.array([point]), # Need 'point' to be a 2D array in cdist()
                     nodes[counter : ])
        
        # Consider as neighbours only the points which are one half lattice constant away. Need to add 'counter' to each
        # element, because the i-th entry in 'dist' is actually the distance between 'point' and the (i + counter)-th node
        # in 'nodes'.
        
        close_points = np.argwhere( dist <= np.max(np.linalg.norm(lattice, axis=1))/2 )[:,1] + counter
        
        # The array close_points is a list of indices of 'nodes' corresponding to points which are up to one half lattice
        # constant away from 'point', the point we are considering now. The next step is to pair these indices with the
        # index of 'point', so as to have an organised list of neighbours. The index of the current 'point' will be equal to
        # 'counter' - 1.
        
        # There's a catch, though, because nothing in life is simple. The indices we are working with are those of the array 'nodes',
        # but the indices we really want are relative to the array 'cells0D'. So, we need to convert them over.
        
        point = np.array([point]) # the inputs of find_equal_rows() must be 2D arrays
        
        index = list(find_equal_rows(point, cells0D)[:,1])
        
        # In order to pair the index of the current point (i.e. 'index') with the indices in 'close_points', we need them both to be Nx1 numpy
        # arrays so we can use np.hstack().
        
        index = index * np.size(close_points)
        
        index = np.array(index).reshape(np.size(close_points), 1)
        
        # The array close_points is a list of indices of 'nodes' corresponding to points which are up to one lattice
        # constant away from 'point', the point we are considering now. We need to scale these indices back as indices of 'cells0D'.
        
        close_points = find_equal_rows(points = nodes[close_points], array = cells0D)[:,1]
        
        # Now we join it all together.
        
        current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
        
        nbrs = np.vstack((nbrs, current_neighbours))
        
        counter += 1
    
    return nbrs


def BCC_onbcc_edges(nodes_bcc, nbrs, cells0D, lattice):
    
    """ This function will be used in create_edges() for the 3D BCC structure.
        It establishes the edges connecting BCC nodes to any other kind of node.
    """
    
    for point in tqdm(nodes_bcc, desc = 'Creating more edges', miniters = 1000): # for a BCC point
        
        # We will make use of the geometry of the unit cell in this context and of the index ordering of the nodes as given by the function
        # create_nodes(). When calculating the distance between the current BCC node and other points, there is no need to cycle through EVERY
        # other node when we know, in terms of indices relative to the array 'cells0D', the lowest and highest nodes we may consider for neighbours.
        
        min_node_index = find_equal_rows(cells0D, np.array([point - np.sum(lattice, axis=1) / 2]))[0,0]
        
        max_node_index = find_equal_rows(cells0D, np.array([point + np.sum(lattice, axis=1) / 2]))[0,0]
        
        points_considered = cells0D[min_node_index : max_node_index + 1] # Need a +1 because it is exclusive; note that this will include 'point'
        
        dist = cdist(np.array([point]),    # Need 'point' to be a 2D array in cdist()
                     points_considered)
    
        # We consider as neighbours only the points which are at most half a cubic diagonal away. But we need to be careful that the column indices
        # in 'dist' will correspond to indices in points_considered, not indices in 'cells0D', which is what we want. But we know that the nodes in
        # points_considered have indices ranging in order from min_node_index to max_node_index, so to the indices below we just need to add the former.
    
        close_points = np.argwhere( dist <= np.linalg.norm(np.sum(lattice, axis=1)) / 2 )[:,1] + min_node_index
        
        # Note that close_points actually includes the very 'point' we are considering, since we did not exclude it from 'dist' above. we just need to
        # remove it. The index of 'point' in 'cells0D' is given by:
        
        index = list(find_equal_rows(cells0D, np.array([point]))[:,0]) # the inputs of find_equal_rows() must be 2D arrays
        
        # And so we just need to:
        
        close_points = np.delete(close_points, np.argwhere(close_points == index)[0], axis=0)

        # In order to pair the index of the current node with the indices in 'close_points', we need them both to be 3x1 numpy
        # arrays so we can use np.hstack().
                                                                                       
        index = index * np.size(close_points)
        
        index = np.array(index).reshape(np.size(close_points), 1)
                        
        current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
        
        nbrs = np.vstack((nbrs, current_neighbours))
        
    return nbrs


def BCC_onfcc_edges(nodes_fcc, nodes_sc, nbrs, cells0D, lattice):
    
    """This function will be used in create_edges() for the 3D BCC structure.
        It establishes the edges connecting FCC nodes to SC and FCC nodes (FCC-BCC edges are covered in BCC_onbcc_edges().
    """
    
    for point in tqdm(nodes_fcc, desc = 'Creating more edges', miniters = 1000): # for an FCC point
            
        # Any edges connecting FCC nodes to BCC nodes have already been established in the previous step, with BCC_onbcc_edges().
        
        # We consider as neighbours only the SC points which are at most half a square diagonal away. To allow for the lattice basis vectors
        # to have asymmetrical dimensions, that "half a square diagonal" is actually better understood as half a cubic diagonal multiplied by
        # the cosine of the angle between (the vector from one of the SC nodes on the same face as 'point' to the BCC node in the
        # same computational unit cell) and (the vector from one of the SC nodes on the same face as 'point' to 'point').
        
        # So, the first step is to find an SC node on the same plane as the virtual node we are considering. Note that, if the plane is
        # of the type xi = const (where xi is a coordinate), then 'point' and the SC nodes will have the same xi coordinate. There will, of course,
        # be several SC nodes that fulfil this condition, so we simply take the first one.
        
        same_plane_SC_node = nodes_sc[np.argwhere(nodes_sc == point)[0,0]]
        
        xi = np.argwhere(nodes_sc == point)[0,1] # This is the coordinate that stays constant along the plane
        
        same_u_cell_BCC_node = point + lattice[xi] / 2
        
        vector1 = same_plane_SC_node - same_u_cell_BCC_node ; vector1_norm = np.linalg.norm(vector1)
        
        vector2 = same_plane_SC_node - point ; vector2_norm = np.linalg.norm(vector2)
        
        max_dist = np.linalg.norm(np.sum(lattice, axis=1)) / 2 * (np.inner(vector1, vector2) / (vector1_norm * vector2_norm))
        
        # Now, there is a problem here with precision. The variable max_dist ends up having more precision than the function cdist() below, and so
        # no points will actually satisfy the condition for close_points. To fix that, we add a little uncertainty.
        
        max_dist = max_dist + 0.000000000000001
        
        # When calculating the distance function next, we consider only SC nodes.
        
        dist = cdist(np.array([point]), # Need 'point' to be a 2D array in cdist()
                     nodes_sc)

        close_points = np.argwhere( dist <= max_dist )[:,1]
        
        # The indices in close_points will be relative to the column indices of 'dist', which are in turn relative to the indice of the SC nodes.
        # So, we just need to convert them to indices relative to 'cells0D'.
        
        close_points = find_equal_rows(cells0D, nodes_sc[close_points])[:,0]
        
        index = list(find_equal_rows(cells0D, np.array([point]))[:,0]) # the inputs of find_equal_rows() must be 2D arrays
                                                                                       
        index = index * np.size(close_points)
        
        index = np.array(index).reshape(np.size(close_points), 1)
                            
        current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
        
        nbrs = np.vstack((nbrs, current_neighbours))
        
    return nbrs


def FCC_other_edges(nodes_fcc: list, size: list, cells0D, nbrs, lattice):
    
    """ This function will be used in create_edges() for the 3D FCC structure.
        It establishes edges connecting FCC nodes to any other kind of node.
    """
        
    # We consider as neighbours only the points which are at most half a square diagonal away. To allow for the lattice basis vectors
    # to have asymmetrical dimensions, we will consider three distinct square diagonals (which are actually rectangular diagonals, but that's
    # just semantics) and, for simplicity, take the largest one as the neighbour distance cut-off (this assumes that there isn't a huuuge
    # difference between the three diagonals).
    
    diag1 = np.sqrt(np.linalg.norm(lattice[0]) ** 2 + np.linalg.norm(lattice[1]) ** 2)
    diag2 = np.sqrt(np.linalg.norm(lattice[1]) ** 2 + np.linalg.norm(lattice[2]) ** 2)
    diag3 = np.sqrt(np.linalg.norm(lattice[2]) ** 2 + np.linalg.norm(lattice[0]) ** 2)
    
    max_dist = max([diag1, diag2, diag3]) / 2 + 0.000001 # introduce some tolerance

    for index in tqdm(nodes_fcc, desc = 'Creating even more edges', miniters = 1000):
        
        point = cells0D[index]
        
        # To save on computation, we consider only the distance between the current point and all FCC nodes that have not been considered yet.
        # Additionally, because we are targeting FCC-type nodes, we need to allow for 'point' to be connected to the FCC node that comes
        # previous on the list. We do this by setting a 'search_range' using our foreknowledge that the 'cells0D' are listed first along the
        # x direction, then the y direction and only then along the z direction. So, to allow both upper and lower FCC nodes to be screened,
        # we set the 'search_range' to be equivalent to two 'layers' of atoms, that is, on the x-y plane.
        
        search_range = nr_nodes([size[0]*2, size[1]*2, 2], 'simple cubic')
        search_range = slice(max(0, index - search_range - 1), min(len(cells0D), index + search_range + 1), None)
        search_range = cells0D[search_range]
        
        dist = cdist(np.array([point]), # Need 'point' to be a 2D array in cdist()
                     search_range)
        
        close_points = np.intersect1d(np.argwhere(dist > 0)[:,1], np.argwhere(dist <= max_dist)[:,1])
        
        # The indices listed in 'close_points' refer to the array 'coords_fcc', so we need them as indices of 'cells0D'.
        
        close_points = find_equal_rows(search_range[close_points], cells0D)[:,1]
        
        # In order to pair the index of the current point with the indices in 'close_points', we need them both to be 3x1 numpy
        # arrays so we can use np.hstack().
        
        index = [index] * np.size(close_points)
        index = np.array(index).reshape(np.size(close_points), 1)
        current_neighbours = np.hstack((index, close_points.reshape(np.size(close_points), 1)))
        
        for pair in current_neighbours.tolist():
            if pair not in nbrs.tolist() and pair[::-1] not in nbrs.tolist():
                nbrs = np.append(nbrs, [pair], axis=0)
    
    nbrs = nbrs[nbrs[:,1].argsort()]                 # sort along the second indicex
    nbrs = nbrs[nbrs[:,0].argsort(kind='mergesort')] # sort along the first indices
    
    return nbrs




def create_edges(cells0D, special0D, lattice, structure, dim = 3, size = None, multiprocess = False):
    """
    This function takes an array of nodes as given by the create_nodes() function and returns an array of paired nodes which
    determines the edges of the cell complex representing a particular crystal structure. The expected number of edges can be
    obtained by the function nr_edges().
    
    Parameters
    ----------
    cells0D : np array (N x 3)
        An array listing the coordinates of points in 3D space that make up the nodes of a cell complex.
    special0D: tuple of numpy arrays
        Contains numpy arrays defining points in 3D space which are nodes of different types.
    lattice : np array OR list (3x3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    structure: str, optional
        A descriptor of the basic structure of the lattice. The default is 'simple cubic'.
    dim: int
        The spatial dimension of the lattice.
    size: list
        Lists the number of unit cells in each spatial coordinate. The default is None because it is only necessary if multiprocess = True.
    multiprocess: bool
        Whether or not to use the version of create_edges() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.


    Returns
    -------
    An array of pairs of indices of the input 'cells0D' which relate neighbouring points according to their slip plane topology.
    """    
        
    ##----- SC w/o multiprocessing -----##
    
    if structure == 'simple cubic' and multiprocess == False:
        
        neighbours = np.empty((0,2))
        
        counter = 1 # This will keep track of the index of the next point as listed in 'cells0D'
        
        for row in tqdm(cells0D, desc = 'Creating edges'):
            
            # Calculate only distances between the current point (row) and all other points which have not been considered
            # yet and are not the very point we're considering (so, all points with indices >= counter).
            
            dist = cdist(
                         np.array([row]),                        # Need 'row' to be a 2D array in cdist()
                         cells0D[counter : np.shape(cells0D)[0], :])
            
            # Consider as neighbours only the points which are one lattice constant away. Need to add 'counter' to each
            # element, because the i-th entry in 'dist' is actually the distance between 'row' and the (i + counter)-th node.
            
            close_points = np.argwhere( dist <= np.max(lattice) )[:,1] + counter
            
            # The array close_points is a list of indices of 'cells0D' corresponding to points which are up to one lattice
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
    
    
    ##----- BCC w/o multiprocessing -----##
    
    elif structure == 'bcc' and multiprocess == False:
        
        """ Required: special0D = (nodes_sc, nodes_bcc, nodes_fcc). """
        
        if dim == 2:
            
            neighbours = np.empty((0,2))
        
        elif dim == 3:
            
            # We will go through each type of node and connect it to the appropriate nodes.
            
            # First off, we have SC nodes. All the simple cubic nodes will connect to other simple cubic nodes a maximum distance of one lattice constant away.
            # For that, we follow a similar algorithm to the one above.      
            
            neighbours = np.empty((0,2))
            
            neighbours = sc2ec_edges(nbrs = neighbours,
                                     cells0D = cells0D,
                                     nodes = ...,
                                     lattice = lattice)
                        
            # Now we move on to the BCC nodes. These need to be connected to every other node in the unit cell, i.e. every node a maximum distance away equal
            # to half a cubic diagonal.
                        
            neighbours = BCC_onbcc_edges(nbrs = neighbours,
                                         cells0D = cells0D,
                                         nodes_bcc = cells0D[special0D[1]],
                                         lattice = lattice)
                        
            # Lastly, we move on to the virtual FCC nodes. Since the connections between these and the BCC nodes are already included above, we only need to
            # establish new connections between an FCC node and the SC nodes on the same face as itself.
                        
            neighbours = BCC_onfcc_edges(nbrs = neighbours,
                                         cells0D = cells0D,
                                         nodes_fcc = cells0D[special0D[2]],
                                         nodes_sc = cells0D[special0D[0]],
                                         lattice = lattice)
                        
            # Now we sort the array by values on the second column and then by values on the first column, so that the index order goes: connections from node 0
            # in order of increasing index of the other endpoint, connections from node 1 in order of increasing index of the other endpoint, etc.
                        
            neighbours = np.sort(neighbours.astype(int)) # Sorts columns in individual rows
            neighbours = neighbours[neighbours[:,1].argsort()] # sort by value on column 1
            neighbours = neighbours[neighbours[:,0].argsort(kind='mergesort')] # sort by value on column 0
            
            # Sorting technique adapted from https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted%20by%20a%20single,the%20values%20in%20an%20array%20in%20ascending%20value. (Accessed 08/04/22)
            
            neighbours = neighbours.astype(int)
                                
    
    ##----- FCC w/o multiprocessing -----##
    
    elif structure == 'fcc' and multiprocess == False:
        
        """ Required: special0D = (nodes_sc, nodes_bcc, nodes_fcc, nodes_ec). """
        
        neighbours = np.empty((0,2))
        
        if dim == 2:
            
            pass
        
        elif dim == 3:
                
            # In the FCC structure, we again need SC, BCC and FCC nodes. The SC nodes will connect to each other and to the nearest
            # FCC nodes; adittionally, the FCC nodes will connect to each other and to the BCC node.
                        
            # Let's deal first with the connections forming a regular hexahedral grid. In the case of FCC, every node is a vertex
            # of some hexahedron.
                        
            neighbours = sc2ec_edges(nbrs = neighbours,
                                     cells0D = cells0D,
                                     nodes = cells0D,
                                     lattice = lattice)
                        
            # Now we deal with all other connections. In the case of FCC, these are the edges connecting FCC-type nodes to other
            # FCC-type nodes.
            
            neighbours = FCC_other_edges(nbrs = neighbours,
                                         nodes_fcc = special0D[2],
                                         size = size,
                                         cells0D = cells0D,
                                         lattice = lattice)
                        
            # Now we sort the array.
                        
            neighbours = np.sort(neighbours.astype(int)) # Sorts columns in individual rows
            neighbours = neighbours[neighbours[:,1].argsort()] # sort by value on column 1
            neighbours = neighbours[neighbours[:,0].argsort(kind='mergesort')] # sort by value on column 0
            
            # Sorting technique adapted from https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted%20by%20a%20single,the%20values%20in%20an%20array%20in%20ascending%20value. (Accessed 08/04/22)
                
            neighbours = neighbours.astype(int)
        
    
    ##----- Multiprocessing -----##
    
    elif multiprocess == True:
        
        """ Required: special0D = (nodes_sc, nodes_bcc, nodes_fcc). """
        
        if dim == 2:
            
            neighbours = np.empty((0,2))
        
        elif dim == 3:
            
            # We will divide the total workload into batches by making use of the partition_nodes() function.
            
            neighbours = np.empty((0,2))
            
            blocks, blocks_sc, blocks_bcc, blocks_fcc = partition_nodes(cells0D, size, lattice, special0D)
            
            # We create a partial function that can be passed into a multiprocessing pool.
            
            part = partial(create_edges,
                           special0D = (blocks_sc, blocks_bcc, blocks_fcc),
                           lattice = lattice,
                           structure = structure,
                           multiprocess = False)
            
            # Instantiate multiprocessing pool.
            
            count = 0
            
            with mp.Pool() as pool:
                
                for result in pool.imap(part, blocks): # use .imap so tasks are issued in the same order as the blocks are issued
                
                # Note that these edges will be defined by the node indices specific to each block, which will result in redundancies, since
                # each block has the same size. We need to account for this, while keeping in mind that each block shares one sheet of nodes
                # with the previous and following ones.
                    
                    if 0 < count < len(blocks):
                        
                        # We have to adjust the node indices in the edge pairs by the number of nodes in one layer of unit cells and subtract the number of nodes shared by each block.
                        
                        result = result + count * (len(blocks[0]) - len(np.array([x for x in set(tuple(y) for y in blocks[count-1]) & set(tuple(z) for z in blocks[count])])))
                
                    neighbours = np.vstack((neighbours, result))
                    
                    count += 1
                    
            neighbours = np.unique(neighbours, axis=0)
        
    
    return neighbours.astype(int)




""" The following 3 functions are to be used specifically in the function create_faces(). """


def bcc_faces_worker(batch, cells1D, structure, size):
    
    faces = np.empty((0,3))
        
    for edge in batch:
        
        # First we find the neighbours of the two endpoints. If we have to search the entire cells1D array, this will be a costly
        # operation. So, we limit the search to a set of likely neighbours of the endpoints of 'edge'. Any node with which the endpoints
        # of 'edge' will form a face must have an index in-between those of the endpoints of 'edge'. If the endpoints of 'edge' would
        # form a face with a node whose index does not lie between the indices of the endpoints, then that same face will be formed when
        # the algorithm gets to the node of highest index.
        
        likely_nb = list(range(edge[0], edge[1])) # 'nb' short for neighbour
        
        for nb in likely_nb:
            
            trial = np.array([[edge[0], nb],
                              [nb, edge[1]]])
            
            # We need to compare this 'trial' array (which defines two edges that might make up a face together with 'edge') with the existing
            # edges in cells1D. However, running through the whole cells1D array would be very expensive, so let's try to find a way to restrict
            # the search to a smaller set of edges. Namely, of course, we can isolate the indices of all edges which start or end at one of the
            # endpoints of 'edge' according to what we specified above.
            
            to_compare = np.where((cells1D[:,0] == edge[0]) | (cells1D[:,1] == edge[1]))
            
            # The array 'to_compare' gives the indices in cells1D of any edges as described above. Now we need to convert those indices back into
            # edges and only then can we use find_equal_rows().
            
            to_compare = cells1D[to_compare]
            
            if len(find_equal_rows(trial, to_compare)[:,1]) == 2: # i.e. if both trial edges exist
                
                new_face = np.array([[edge[0], nb, edge[1]]])
                
                faces = np.vstack((faces, new_face)).astype(int)
                                
            else:
                
                pass
    
    return faces


def fcc_faces_worker(batch, cells1D, structure, size):
               
   faces = np.empty((0,3))
                  
   for edge in batch:
       
       # First we find the neighbours of the two endpoints. If we have to search the entire cells1D array, this will be a costly
       # operation. So, we limit the search to a set of likely neighbours of the endpoints of 'edge'. Any node with which the endpoints
       # of 'edge' will form a face must have an index in-between those of the endpoints of 'edge'. If the endpoints of 'edge' would
       # form a face with a node whose index does not lie between the indices of the endpoints, then that same face will be formed when
       # the algorithm gets to the node of highest index.
       
       likely_nb = list(range(edge[0], edge[1])) # 'nb' short for neighbour
       
       for nb in likely_nb:
           
           # Any face must be composed of existing edges, so using the likely neighbours we form trial edges and test their existence.
           
           trial = np.array([[edge[0], nb],
                             [nb, edge[1]]])
           
           # We need to compare this 'trial' array (which defines two edges that might make up a face together with 'edge') with the existing
           # edges in cells1D. However, running through the whole cells1D array would be very expensive, so let's try to find a way to restrict
           # the search to a smaller set of edges. Namely, of course, we can isolate the indices of all edges which start or end at one of the
           # endpoints of 'edge' according to what we specified above.
           
           to_compare = np.where((cells1D[:,0] == edge[0]) | (cells1D[:,1] == edge[1]))
           
           # The array 'to_compare' gives the indices in cells1D of any edges as described above. Now we need to convert those indices back into
           # edges and only then can we use find_equal_rows().
           
           to_compare = cells1D[to_compare]
           
           if len(find_equal_rows(trial, to_compare)[:,1]) == 2: # i.e. if both trial edges exist
                              
               new_face = np.array([[edge[0], nb, edge[1]]])
               
               faces = np.vstack((faces, new_face)).astype(int)
                              
           else:
               
               pass
   
   return faces


def find_faces_slip(face, structure, cells0D, special0D):
    """
    Determines if the input 'face' corresponds to a slip plane 2-cell.
    
    Parameters
    ----------
    structure: str, optional
        A descriptor of the basic structure of the lattice.
    special0D: tuple of numpy arrays
        Contains numpy arrays defining points in 3D space which are nodes of different types.
        Requires: special0D = (nodes_sc, nodes_bcc, nodes_fcc)
        
    Returns
    -------
    Returns True if the input 'face' corresponds to a slip plane 2-cell.
    """
    
    if structure == 'fcc':
        
        # A face is a slip face if all of its nodes are of type FCC or if two of its nodes are of type FCC and the last is of
        # type SC.
        
        is_slip_face = ((face[0] in special0D[2] and face[1] in special0D[2] and face[2] in special0D[2])
                     or (face[0] in special0D[0] and face[1] in special0D[2] and face[2] in special0D[2])
                     or (face[0] in special0D[2] and face[1] in special0D[0] and face[2] in special0D[2])
                     or (face[0] in special0D[2] and face[1] in special0D[2] and face[2] in special0D[0]))
        
        # # But now we need to exclude those faces that are inside the inner octagon, since they are composed of all-FCC nodes,
        # # but are not slip planes. We can do this by investigating the distances between the nodes of the face.
        
        # if is_slip_face:
            
        #     d = [np.linalg.norm(cells0D[face[2]] - cells0D[face[1]]),
        #          np.linalg.norm(cells0D[face[1]] - cells0D[face[0]]),
        #          np.linalg.norm(cells0D[face[0]] - cells0D[face[2]])]
            
        #     if max(d) / min(d) > 1:
        #         is_slip_face = False
        
    elif structure == 'bcc':
        
        # A face is a slip face if one and only one of its nodes is of type BCC.
        
        is_slip_face = ((face[0] in special0D[1] and face[1] not in special0D[1] and face[2] not in special0D[1])
                     or (face[0] not in special0D[1] and face[1] in special0D[1] and face[2] not in special0D[1])
                     or (face[0] not in special0D[1] and face[1] not in special0D[1] and face[2] in special0D[1]))
    
    return is_slip_face
        
        


def create_faces(cells1D, structure, size, cells0D, special0D = None, multiprocess = False):
    """
    Parameters
    ----------
    cells1D : np array (M x 2)
        A numpy array of pairs of indices as given by the create_edges function.
    structure: str, optional
        A descriptor of the basic structure of the lattice.
    size: list
        Lists the number of unit cells in each spatial coordinate. The default is None because it is only necessary if multiprocess = True.
    special0D: tuple of numpy arrays
        Contains numpy arrays defining points in 3D space which are nodes of different types. The default is None because it
        is only needed for 'bcc' and 'fcc'.
    multiprocess: bool
        Whether or not to use the version of create_faces() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.
    
    Returns
    -------
    A numpy array where each row contains the indices of poits in 'array' that delineate a face.
    """
    
    ##----- SC -----##
    
    if structure == 'simple cubic' and multiprocess == False:
    
    # Proceed as follows: take a point in an edge in the input 'cells1D'. Find out if any two neighbours of that point have another
    # neighbour in common (which is not the original point). Then, the face will be described by the original point, those
    # two points and finally the second common neighbour.
    
        faces = np.empty((0,4))
        
    # Because the first column of 'neighbours' will have several repeated entries in sequence, instead of cycling through
    # each row, we can just cycle through each node index and find all its neighbours in one go.
        
        # For each node:

        for i in tqdm(range(0, np.max(cells1D[:,1]) + 1), desc = 'Creating faces', miniters = 1000):  # Need to add +1 because range() is exclusive at the top value
        
            # We select the rows in 'cells1D', i.e. the edges, that contain that node in the first column.
        
            rows = np.where(cells1D[:,0] == i)
            neighbour_points = cells1D[rows][:,1]
            
            # This last array will be a list of the neighbours of point i. Now we need to find the neighbours of those
            # neighbours.
            
            # For each pair of neighbours of point i, we check which ones have a second neighbour in common (which is not i itself).
            
            for [x,y] in combinations(neighbour_points, 2):
                
                # To deconstruct the neighbours of x and y, we have to work with both columns of 'cells1D', which requires
                # more tact than the simple procedure above.
                
                # First, we identify the rows in 'cells1D' where x and y are mentioned.
                
                rows_x = cells1D[np.where(cells1D == x)[0]]
                rows_y = cells1D[np.where(cells1D == y)[0]]
                
                # These arrays contain pairs of x or y and their neighbours, including i. So, secondly, we need to remove any 
                # mention of i, and then any mention of x or y.
                
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
    
    elif structure == 'bcc' and multiprocess == False:
        
        """ Requires input: special0D = (nodes_sc, nodes_bcc, nodes_fcc) """
        
        # For a BCC structure, the network of faces is composed solely of triangular faces. The procedure here will be: take an edge, take its
        # endpoints, find which other node(s) have those endpoints as neighbours, make faces.
        
        faces = np.empty((0,3))
        
        faces_slip = []
        
        for edge in tqdm(cells1D, desc = 'Creating faces'):
            
            # First we find the neighbours of the two endpoints. If we have to search the entire cells1D array, this will be a costly
            # operation. So, we limit the search to a set of likely neighbours of the endpoints of 'edge'. Any node with which the endpoints
            # of 'edge' will form a face must have an index in-between those of the endpoints of 'edge'. If the endpoints of 'edge' would
            # form a face with a node whose index does not lie between the indices of the endpoints, then that same face will be formed when
            # the algorithm gets to the node of highest index.
            
            likely_nb = list(range(edge[0], edge[1])) # 'nb' short for neighbour
            
            for nb in likely_nb:
                
                trial = np.array([[edge[0], nb],
                                  [nb, edge[1]]])
                
                # We need to compare this 'trial' array (which defines two edges that might make up a face together with 'edge') with the existing
                # edges in cells1D. However, running through the whole cells1D array would be very expensive, so let's try to find a way to restrict
                # the search to a smaller set of edges. Namely, of course, we can isolate the indices of all edges which start or end at one of the
                # endpoints of 'edge' according to what we specified above.
                
                to_compare = np.where((cells1D[:,0] == edge[0]) | (cells1D[:,1] == edge[1]))
                
                # The array 'to_compare' gives the indices in cells1D of any edges as described above. Now we need to convert those indices back into
                # edges and only then can we use find_equal_rows().
                
                to_compare = cells1D[to_compare]
                
                if len(find_equal_rows(trial, to_compare)[:,1]) == 2: # i.e. if both trial edges exist
                    
                    new_face = np.array([[edge[0], nb, edge[1]]])
                    faces = np.vstack((faces, new_face)).astype(int)
                    new_face = new_face[0] # Need the slice [0] because new_face was left as a 2D array above
                    
                    # The last step is to identify those faces which lie on slip planes. The way we can identify them is by the fact
                    # that they must all include 1 BCC node, whereas the remaining non-slip faces are all composed solely of SC nodes.
                    
                    """ Old version
                    condition = ((new_face[0] in special0D[1] and new_face[1] not in special0D[1] and new_face[2] not in special0D[1])
                              or (new_face[0] not in special0D[1] and new_face[1] in special0D[1] and new_face[2] not in special0D[1])
                              or (new_face[0] not in special0D[1] and new_face[1] not in special0D[1] and new_face[2] in special0D[1]))
                    
                    if condition:
                        faces_slip.append(len(faces) - 1) # because 'new_face' was the last one to be added
                    """
                else:
                    pass

        # Now we sort the array 'faces' by order of the indices of the constituent nodes, in order of priority: along the
        # first column, then the second and then the third.
        
        faces = faces[faces[:,2].argsort()]                 # sort along the last column
        faces = faces[faces[:,1].argsort(kind='mergesort')] # sort along the second column
        faces = faces[faces[:,0].argsort(kind='mergesort')] # sort along the first column
        
        for face in faces:
            faces_slip.append(find_faces_slip(face, structure, cells0D, special0D))
                
        faces_slip = np.array(list(range(0, len(faces))))[faces_slip].tolist()
        
        return faces, faces_slip
    
    
    ##----- FCC -----##
    
    elif structure == 'fcc' and multiprocess == False:
        
        """ Requires input: special0D = (nodes_sc, nodes_bcc, nodes_fcc, nodes_ec) """
        
        # Since all faces are triangles, like in BCC, we can take the algorithm for the BCC case and adapt it.
        
        # For an FCC structure, the network of faces is composed solely of triangular faces. The procedure here will be: take an edge, take its
        # endpoints, find which other node(s) have those endpoints as neighbours, make faces.
        
        faces = np.empty((0,3))
        faces_slip = []
                    
        for edge in tqdm(cells1D, desc = 'Creating faces'):
            
            # First we find the neighbours of the two endpoints. If we have to search the entire cells1D array, this will be a costly
            # operation. So, we limit the search to a set of likely neighbours of the endpoints of 'edge'. Any node with which the endpoints
            # of 'edge' will form a face must have an index in-between those of the endpoints of 'edge'. If the endpoints of 'edge' would
            # form a face with a node whose index does not lie between the indices of the endpoints, then that same face will be formed when
            # the algorithm gets to the node of highest index.
            
            likely_nb = list(range(edge[0], edge[1])) # 'nb' short for neighbour
            
            for nb in likely_nb:
                
                # Any face must be composed of existing edges, so using the likely neighbours we form trial edges and test their existence.
                
                trial = np.array([[edge[0], nb],
                                  [nb, edge[1]]])
                
                # We need to compare this 'trial' array (which defines two edges that might make up a face together with 'edge') with the existing
                # edges in cells1D. However, running through the whole cells1D array would be very expensive, so let's try to find a way to restrict
                # the search to a smaller set of edges. Namely, of course, we can isolate the indices of all edges which start or end at one of the
                # endpoints of 'edge' according to what we specified above.
                
                to_compare = np.where((cells1D[:,0] == edge[0]) | (cells1D[:,1] == edge[1]))
                
                # The array 'to_compare' gives the indices in cells1D of any edges as described above. Now we need to convert those indices back into
                # edges and only then can we use find_equal_rows().
                
                to_compare = cells1D[to_compare]
                
                if len(find_equal_rows(trial, to_compare)[:,1]) == 2: # i.e. if both trial edges exist
                    
                    new_face = np.array([[edge[0], nb, edge[1]]])
                    faces = np.vstack((faces, new_face)).astype(int)
                    
                    # Here we change the algorithm slightly. We want to keep a log of the faces which will constitute slip planes.
                    # This does not include faces on the sides of the cubic unit cell nor faces on the inside of the inner octahedron.
                    
                    new_face = new_face[0] # Need the slice [0] because new_face was left as a 2D array
                    
                else:
                    pass
        
        # Now we sort the array 'faces' by order of the indices of the constituent nodes, in order of priority: along the
        # first column, then the second and then the third.
        
        faces = faces[faces[:,2].argsort()]                 # sort along the last column
        faces = faces[faces[:,1].argsort(kind='mergesort')] # sort along the second column
        faces = faces[faces[:,0].argsort(kind='mergesort')] # sort along the first column
        
        for face in faces:
            faces_slip.append(find_faces_slip(face, structure, cells0D, special0D))
                
        faces_slip = np.array(list(range(0, len(faces))))[faces_slip].tolist()
        
        return faces, faces_slip
    
    
    ##----- MULTIPROCESSING -----##

    elif multiprocess == True:
        
        # The main issue with the non-multiprocessing algorithms above seems to be simply the sheer number of edges that the
        # function needs to go through. The most computationally costly operation in the 'BCC' and 'FCC' cases defined above
        # that can be straightforwardly parallelised is the for-loop "for edge in cells1D". To accomplish this parallelisation,
        # we partition the set of indices of all 1-cells according to the number of available CPUs.
        
        partition = int(len(cells1D) / mp.cpu_count())
        remainder = len(cells1D) % mp.cpu_count()
        
        batches = [list(range(x, x + partition)) for x in range(0, len(cells1D) - partition, partition)] + [list(range(partition * mp.cpu_count(), partition * mp.cpu_count() + remainder))]
        
        # We create the workers for the multiprocessing pool. They will simply do the work of the non-multiprocessing cases
        # above, but with limited 1-cell intakes.
        
        if structure == 'bcc':
            
            part = partial(bcc_faces_worker,
                           cells1D = cells1D,
                           structure = structure,
                           size = size)
            
        elif structure == 'fcc':
            
            part = partial(fcc_faces_worker,
                           cells1D = cells1D,
                           structure = structure,
                           size = size)
                    
        global_faces = np.empty((0,3))  ;  global_faces_slip = []
        
        # Need to turn the 'batches' from edge indices to edges, i.e. (node,node) pairs.
        
        for i in range(len(batches)):
            batches[i] = cells1D[batches[i]]
            
        # Instantiate multiprocessing pool:
        
        with mp.Pool() as pool:            
            for result in pool.imap(part, batches):
                global_faces = np.vstack((global_faces, result))
        
        # Now we sort the array 'global_faces' and list 'global_faces_slip'.
        
        global_faces = global_faces[global_faces[:,2].argsort()]                 # sort along the last column
        global_faces = global_faces[global_faces[:,1].argsort(kind='mergesort')] # sort along the second column
        global_faces = global_faces[global_faces[:,0].argsort(kind='mergesort')] # sort along the first column

        # Now we need to sort out which faces correspond to slip planes.
        
        for face in global_faces:
            global_faces_slip.append(find_faces_slip(face, structure, cells0D, special0D))
        
        global_faces_slip = np.array(list(range(0, len(global_faces))))[global_faces_slip].tolist()
                
        return global_faces.astype(int), global_faces_slip



""" The following function is to be used specifically in the function create_volumes(). """


def bccfcc_volumes_worker(batch, cells2D):
    
    # In the BCC and FCC structures, each unit cell contains several 3-cells which are not "shared" with any other unit cell. They are
    # non-regular tetrahedra, so will contain 4 nodes and 4 faces.
    
    # So, here's how we're gonna do this. We take a face and consider every other face which shares its nodes. Then, we take the nodes of
    # those faces and see which one pops up the most times. recall that faces in BCC and FCC structures are triangles.
    
    volumes = np.empty((0,4))
    
    counter = 0 # will keep track of which face we are currently on
    
    for face in batch:
        
        nb_faces_0 = np.argwhere(cells2D == face[0])[:,0]           # first find the faces that have the same node face[0], 
        nb_faces_0 = np.delete(nb_faces_0, nb_faces_0 == counter)   # then remove the current face to avoid issues
        
        nb_faces_1 = np.argwhere(cells2D == face[1])[:,0]           # first find the faces that have the same node face[1],
        nb_faces_1 = np.delete(nb_faces_1, nb_faces_1 == counter)   # then remove the current face to avoid issues
        
        nb_faces_2 = np.argwhere(cells2D == face[2])[:,0]           # first find the faces that have the same node face[2],
        nb_faces_2 = np.delete(nb_faces_2, nb_faces_2 == counter)   # then remove the current face to avoid issues
        
        # Now we consider intersections between these 3 arrays. Somewhere in there we will find the 4th node to complete the volume.
        
        int_1 = np.intersect1d(nb_faces_0, nb_faces_1)
        int_2 = np.intersect1d(nb_faces_0, nb_faces_2)
        int_3 = np.intersect1d(nb_faces_1, nb_faces_2)
        
        last_node = np.hstack((int_1, int_2, int_3)) # gather the 'cells2D' indices that were left from the intersections
        
        last_node = cells2D[last_node].reshape(1, np.size(cells2D[last_node]))[0] # transform into 'cells0D' indices and reshape into one row
        
        last_node = np.delete(last_node, np.where(last_node == face[0])) # remove node face[0] to avoid double counting
        last_node = np.delete(last_node, np.where(last_node == face[1])) # remove node face[1] to avoid double counting
        last_node = np.delete(last_node, np.where(last_node == face[2])) # remove node face[2] to avoid double counting
        
        # What we have left in the array 'last_node' is a collection of points which might be the final 4th node needed to complete
        # the volume. We can identify it by the fact that it will be the most common one in last_node. So,
        
        unique, counts = np.unique(last_node, return_counts=True)
        
        last_node = unique[counts == max(counts)]
        
        for i in range(np.size(last_node)):
        
            new_volume = np.hstack((face, last_node[i]))
            new_volume = np.sort(new_volume)
            
            if list(find_equal_rows(np.array([new_volume]), volumes)) == []:  # If the volume does not exist already
                
                volumes = np.vstack((volumes, new_volume))
            
            else: # Otherwise it already exists and so we skip it to avoid repetition
            
                pass
        
        counter += 1
        
    volumes = np.sort(volumes)
        
    return volumes.astype(int)




def create_volumes(lattice, structure, cells0D = None, cells2D = None, multiprocess = False):
    """
    Parameters
    ----------
    lattice : np array OR list (3 x 3)
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions.
    structure : str, optional
        A descriptor of the basic structure of the lattice.
    cells0D : np array
        A numpy array whose rows list the spatial coordinates of points. The default is None. Only required for 'simple cubic'.
    cells2D : np array
        An array whose rows list the indices of nodes which make up one face. The default is None.
    multiprocess: bool
        Whether or not to use the version of create_volumes() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.

    
    Returns
    -------
    A numpy array whose rows list the indices of points in 3D space that define a 3-cell of a 3D complex.
    """
    
    
    ##----- SC w/o Multiprocessing -----##
    
    if structure == 'simple cubic' and multiprocess == False:
        
        """ Need input 'cells0D = nodes' """
        
        # The most direct way is to take each node and build the volume for which it is a corner (corresponding to the origin
        # of the unit cell). This does not work for the nodes on surfaces corresponding to xi = L, where xi is a spatial coordinate
        # (such as x1, x2, x3) and L**3 is the size of the (simple) cubic complex, so we will need to remove these.
        
        volumes = np.empty((0, 8))
        
        L1 = np.max(cells0D[:,0])
        L2 = np.max(cells0D[:,1])
        L3 = np.max(cells0D[:,2])
        
        nodes_to_remove = list(np.argwhere(cells0D[:,0] == L1)[:,0]) # All the nodes on the surface x1 = L1
        
        for i in np.argwhere(cells0D[:,1] == L2)[:,0]:
            
            nodes_to_remove.append(i) # All the nodes on the surface x2 = L2
            
        for i in np.argwhere(cells0D[:,2] == L3)[:,0]:
        
            nodes_to_remove.append(i) # All the nodes on the surface x3 = L3
        
        # The list nodes_to_remove will contain some repeated elements corresponding to the corners of the complex. We can
        # remove these easily by making use of sets, which are unordered collections of unique elements.
        # Inspired by poke's reply on https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists (Accessed 16/02/2022)
        
        nodes_to_remove = list(set(nodes_to_remove)); nodes_to_remove.sort()
        
        relevant_nodes = np.delete(cells0D, nodes_to_remove, axis=0)
        
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
                
                nodes_in_volume_indices = np.sort(find_equal_rows(cells0D, nodes_in_volume)[:,0]) # Use np.sort() just to organise
                
                volumes = np.vstack((volumes, nodes_in_volume_indices))
                
        elif np.shape(lattice)[0] < 3:
            
            pass # There are no volumes in a 2D complex
            
        return volumes.astype(int)
    
    
    ##----- BCC & FCC w/o Multiprocessing-----##
    
    elif structure in ['bcc', 'fcc'] and multiprocess == False:
        
        """ Need input 'cells2D = faces' """
        
        # In the BCC and FCC structures, each unit cell contains several 3-cells which are not "shared" with any other unit cell. They are
        # non-regular tetrahedra, so will contain 4 nodes and 4 faces.
        
        # So, here's how we're gonna do this. We take a face and consider every other face which shares its nodes. Then, we take the nodes
        # of those faces and see which one pops up the most times. Recall that faces in BCC and FCC structures are triangles.
        
        volumes = np.empty((0,4))
        
        counter = 0 # will keep track of which face we are currently on
        
        for face in tqdm(cells2D, desc = 'Creating volumes', miniters = 1000):
            
            nb_faces_0 = np.argwhere(cells2D == face[0])[:,0]           # first find the faces that have the same node face[0], 
            nb_faces_0 = np.delete(nb_faces_0, nb_faces_0 == counter)   # then remove the current face to avoid issues
            
            nb_faces_1 = np.argwhere(cells2D == face[1])[:,0]           # first find the faces that have the same node face[1],
            nb_faces_1 = np.delete(nb_faces_1, nb_faces_1 == counter)   # then remove the current face to avoid issues
            
            nb_faces_2 = np.argwhere(cells2D == face[2])[:,0]           # first find the faces that have the same node face[2],
            nb_faces_2 = np.delete(nb_faces_2, nb_faces_2 == counter)   # then remove the current face to avoid issues
            
            # Now we consider intersections between these 3 arrays. Somewhere in there we will find the 4th node to complete the volume.
            
            int_1 = np.intersect1d(nb_faces_0, nb_faces_1)
            int_2 = np.intersect1d(nb_faces_0, nb_faces_2)
            int_3 = np.intersect1d(nb_faces_1, nb_faces_2)
            
            last_node = np.hstack((int_1, int_2, int_3)) # gather the 'cells2D' indices that were left from the intersections
            
            last_node = cells2D[last_node].reshape(1, np.size(cells2D[last_node]))[0] # transform into 'cells0D' indices and reshape into one row
            
            # Remove nodes face[0], face[1] and face[2] to avoid double counting
            
            last_node = np.delete(last_node, np.where(last_node == face[0]))
            last_node = np.delete(last_node, np.where(last_node == face[1]))
            last_node = np.delete(last_node, np.where(last_node == face[2]))
            
            # What we have left in the array 'last_node' is a collection of points which might be the final 4th node needed to complete
            # the volume. We can identify it by the fact that it will be the most common one in last_node.
            
            unique, counts = np.unique(last_node, return_counts = True)
            
            last_node = unique[counts == max(counts)]
            
            for i in range(np.size(last_node)):
                
                new_volume = np.hstack((face, last_node[i]))
                new_volume = np.sort(new_volume)
                
                if list(find_equal_rows(np.array([new_volume]), volumes)) == []:  # If the volume does not exist already
                    
                    volumes = np.vstack((volumes, new_volume))
                
                else: # Otherwise it already exists and so we skip it to avoid repetition
                
                    pass
            
            counter += 1
            
        volumes = np.sort(volumes)
                
        return volumes.astype(int)
    
    
    ##----- BCC & FCC with Multiprocessing-----##
    
    elif structure in ['bcc', 'fcc'] and multiprocess == True:
        
        """ Need input 'cells2D = faces' """
        
        # The main issue with the non-multiprocessing algorithm above seems to be simply the sheer number of faces that the
        # function needs to go through. The most computationally costly operation in the BCC & FCC case defined above
        # that can be straightforwardly parallelised is the for-loop "for face in cells2D". To accomplish this parallelisation,
        # we partition the set of indices of all 2-cells according to the number of available CPUs.
        
        partition = int(len(cells2D) / mp.cpu_count())
        remainder = len(cells2D) % mp.cpu_count()
        
        if remainder == 0:
            
            batches = [list(range(x, x + partition)) for x in range(0, len(cells2D), partition)]
        
        else:
            
            batches = [list(range(x, x + partition)) for x in range(0, len(cells2D) - partition, partition)] + [list(range(partition * mp.cpu_count(), partition * mp.cpu_count() + remainder))]
                
        # We create the workers for the multiprocessing pool. They will simply do the work of the non-multiprocessing case
        # above, but with limited 2-cell intakes.
                    
        part = partial(bccfcc_volumes_worker,
                       cells2D = cells2D)
            
        # Instantiate multiprocessing pool
        
        global_volumes = np.empty((0,4))
        
        # Need to turn the 'batches' from 'cells2D' indices to faces, i.e. (node,node,node) triplets.
        
        for i in range(len(batches)):
            
            batches[i] = cells2D[batches[i]]
        
        with mp.Pool() as pool:
            
            for result in pool.imap(part, batches):
                
                global_volumes = np.vstack((global_volumes, result))
                        
        # Now we sort the array 'global_volumes'.
        
        global_volumes = global_volumes[global_volumes[:,2].argsort()]                 # sort along the last column
        global_volumes = global_volumes[global_volumes[:,1].argsort(kind='mergesort')] # sort along the second column
        global_volumes = global_volumes[global_volumes[:,0].argsort(kind='mergesort')] # sort along the first column
        
        global_volumes= np.unique(global_volumes.astype(int), axis = 0)

        return global_volumes.astype(int)




def build_complex(struc,
                  size,
                  lattice = [[1,0,0],[0,1,0],[0,0,1]],
                  origin = np.array([0,0,0]),
                  dim = 3,
                  multiprocess = False):
    """
    Parameters
    ----------
    struc : str
        A descriptor of the basic structure of the lattice.
    size : list
        Lists the number of unit cells in each spatial coordinate.
    lattice : np array OR list (3x3 or 2x3), optional
        An array of vectors describing the periodicity of the lattice in the 3 canonical directions. The default is [[1,0,0],[0,1,0],[0,0,1]].
    origin : numpy array (1 x 3)
        An array with the spatial coordinates of the origin point for the space. These are the coordinates of the centre of a unit cell at a corner
        of the complex.
    dim : int, optional
        The dimension of the space. The default is 3.
        multiprocess: bool
        Whether or not to use the version of label_nodes() that makes use of multiprocessing (ideal for large input arrays). The
        default is False.

    Returns
    -------
    The nodes, edges, faces and volumes (and other optional topological information) of a discrete cell complex which reproduces the slip planes of
    the "struc" crystal structure (simple cubic, FCC, BCC).
    
    Notes
    -----
    standard call:
nodes, edges, faces, faces_slip, volumes = build.build_complex(struc, size, multiprocess)
    
    """
    
    if type(lattice) == list:
        lattice = np.array(lattice)
        
    first_u_cell = np.array([[0,0,0]]) + np.sum(lattice, axis=0) / 2
    
    
    if struc == 'simple cubic':
        
        multiprocess = False # Multiprocessing not supported for simple cubic yet
        
        #------- NODES in SC

        first_node = first_u_cell - np.sum(lattice, axis=0) / 2
        nodes = create_nodes(structure = struc,
                             origin = first_node,
                             lattice = lattice,
                             size = size,
                             dim = dim,
                             multiprocess = multiprocess)
        
        #------- EDGES in SC
        
        edges = create_edges(nodes,
                             lattice,
                             structure = struc,
                             dim = dim,
                             multiprocess = multiprocess)
                            
        #------- FACES in SC
        
        faces = create_faces(edges,
                             structure = struc,
                             multiprocess = multiprocess)
        faces_slip = np.copy(faces)
                
        #------- VOLUMES in SC
        
        volumes = create_volumes(lattice,
                                 struc,
                                 cells_0D = nodes,
                                 multiprocess = multiprocess)
        
        
        
    elif struc in ['fcc', 'bcc']:
        
        status = []
        
        print("\n\\\\---------- 1. CREATING NODES ----------//\n")
        
        t0 = time.time()
        
        nodes = create_nodes(structure = struc,
                              lattice = lattice,
                              size = size,
                              origin = origin,
                              multiprocess = multiprocess)
        t1 = time.time() - t0
        
        if len(nodes) == nr_nodes(size, struc):
            print('\nThe function create_nodes() produced the expected number of nodes.')
            status.append(True)
        elif len(nodes) != nr_nodes(size, struc):
            print('\nThe function create_nodes() did NOT produce the expected number of nodes.')
            status.append(False)
            
        if check_uniqueness(nodes):            
            print('The function create_nodes() produced unique nodes.')
            status.append(True)
        else:
            print('The function create_nodes() did NOT produce unique nodes.')
            status.append(False)
            
        print(f"Time elapsed: {t1} s.\n")
        
        
        
        print("\\\\---------- 2. LABELING NODES ----------//\n")
        
        t0 = time.time()
        
        special0D = label_nodes(cells0D = nodes,
                                origin = origin,
                                lattice = lattice,
                                size = size,
                                multiprocess = multiprocess)
        t2 = time.time() - t0
                
        print(f"Time elapsed: {t2} s.\n")
        
        
        
        print("\\\\---------- 3. CREATING EDGES ----------//\n")
        
        t0 = time.time()
    
        edges = create_edges(cells0D = nodes,
                             special0D = special0D,
                             lattice = lattice,
                             structure = struc,
                             dim = dim,
                             size = size,
                             multiprocess = multiprocess)
        t3 = time.time() - t0
        
        if len(edges) == nr_edges(size, struc):
            print("\nThe function create_edges() produced the expected number of edges.")
            status.append(True)
            
        elif len(edges) != nr_edges(size, struc):
            print("\nThe function create_edges() did NOT produce the expected number of edges.")
            status.append(False)
        if check_uniqueness(edges):
            print('The function create_edges() produced unique edges.')
            status.append(True)
        else:
            print('The function create_edges() did NOT produce unique edges.')
            status.append(False)
            
        print(f"Time elapsed: {t3} s.\n")
        
        
        
        print("\\\\---------- 4. CREATING FACES ----------//\n")
        
        t0 = time.time()
        
        faces, faces_slip = create_faces(cells1D = edges,
                                          structure = struc,
                                          size = size,
                                          cells0D = nodes,
                                          special0D = special0D,
                                          multiprocess = multiprocess)
        t4 = time.time() - t0
        
        if len(faces) == nr_faces(size, struc):
            print("\nThe function create_faces() produced the expected number of faces.")
            status.append(True)
        elif len(faces) != nr_faces(size, struc):
            print("\nThe function create_faces() did NOT produce the expected number of faces.")
            status.append(False)
            
        if check_uniqueness(faces):
            print('The function create_faces() produced unique faces.')
            status.append(True)
        else:
            print('The function create_faces() did NOT produce unique faces.')
            status.append(False)
            
        if len(faces_slip) == nr_faces_slip(size, struc):
            print("The function create_faces() produced the expected number of slip faces.")
            status.append(True)
        elif len(faces_slip) != nr_faces_slip(size, struc):
            print("The function create_faces() did NOT produce the expected number of slip faces.")
            status.append(False)
            
        if check_uniqueness(faces_slip):
            print('The function create_faces() produced unique slip faces.')
            status.append(True)
        else:
            print('The function create_faces() did NOT produce unique slip faces.')
            status.append(False)
            
        print(f"Time elapsed: {t4} s.\n")
        
        
        
        print("\\\\---------- 4. CREATING VOLUMES ----------//\n")
        
        t0 = time.time()
        
        volumes = create_volumes(lattice = lattice,
                                  structure = struc,
                                  cells0D = nodes,
                                  cells2D = faces,
                                  multiprocess = multiprocess)
        
        t5 = time.time() - t0
        
        if len(volumes) == nr_volumes(size, struc):
            print("\nThe function create_volumes() produced the expected number of volumes.")
            status.append(True)
        elif len(volumes) != nr_volumes(size, struc):
            print("\nThe function create_volumes() did NOT produce the expected number of volumes.")
            status.append(False)
            
        if check_uniqueness(volumes):
            print('The function create_volumes() produced unique volumes.')
            status.append(True)
        else:
            print('The function create_volumes() did NOT produce unique volumes.')
            status.append(False)
            
        print(f"Time elapsed: {t5} s.\n")
        
        if np.all(status):
            print("-> SUCCESS!\n")
        else:
            print("-> FAILURE!\n")
        
        del t0

    return nodes, edges, faces, faces_slip, volumes
    




"""
----------------------------------------------------------------------------------------------------------------------------
"""



if __name__ == '__main__':
    
    struc = 'fcc'
    size = [2,2,2]
    lattice = np.array([[1,0,0], [0,1,0], [0,0,1]])
    origin = np.array([0.5,0.5,0.5])
    multi = False
        
    # nodes, edges, faces, faces_slip, volumes = build_complex(struc, size, lattice, origin, multiprocess = multi)
    
    # nr_cells = [nr_nodes(size, struc), nr_edges(size, struc), nr_faces(size, struc), nr_volumes(size, struc)]
    
    # from iofiles import write_to_file
    
    # write_to_file(nodes, 'nodes',
    #               edges, 'edges',
    #               faces, 'faces',
    #               faces_slip, 'faces_slip',
    #               volumes, 'volumes',
    #               nr_cells, 'nr_cells',
    #               new_folder = True)
