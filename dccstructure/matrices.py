# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 12:43 2022

Last edited on: 21/06/2022 11:15

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions that find the degree node distribution, incidence
matrices, adjacency matrices, and other combinatorial forms, given a cell complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
from itertools import combinations # to avoid nested for loops
from scipy import sparse
from build import find_equal_rows
import orientations as dcco


# ----- # ----- # FUNCTIONS # ------ # ----- #


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




def combinatorial_form(structure, degree, cells_3D, cells_2D, cells_1D, cells_0D, v2f=None, f2e=None):
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
        An array whose rows list the indices of faces which make up one volume, with considerations for relative orientation.
        The default value (=None) should be kept if you want the function to calculate both v2f and f2e, otherwise you should
        pass your own v2f and f2e arrays.
    f2e : np array
        An array whose rows list the indices of edges which make up one face, with considerations for relative orientation.
        The default value (=None) should be kept if you want the function to calculate both v2f and f2e, otherwise you should
        pass your own v2f and f2e arrays.

    Returns
    -------
    For degrees 0 and 1 (adjacency and incidence matrices), returns a numpy array (N x 3) which gives the relevant matrix in
    explicit sparse format, that is, the first column refers to the row index, the second column to the column index and the
    third column to the element value.
    For degrees 2 and 3, returns the same as above but in a sparse.csc_matrix() format.
    """
    
    try:
        
        # If we ever want to return to sparse matrix format: comb_form = sparse.csc_matrix((values, (rows, cols)), shape = (np.size(cells[:,0]), np.size(cells[:,0])))
    
        
        e2n = cells_1D * [1, -1] # this defines the relative orientation between edges and nodes, where an edge-node pair is considered
                                 # to be positively relatively oriented if the edge points away from the node.
                                 
        if v2f is None and f2e is None:
            
            v2f = dcco.volumes_to_faces(cells_3D, cells_2D)
            
            f2e = dcco.faces_to_edges(cells_2D, cells_1D)
            
            v2f, f2e = dcco.find_relative_orientations(cells_3D, cells_2D, cells_1D, cells_0D, v2f, f2e)
    
    
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
                
            comb_form_0 = np.hstack(( np.transpose(np.array([rows_0])),
                                      np.transpose(np.array([cols_0])),
                                      np.transpose(np.array([values_0])) ))
            
            comb_form_0 = comb_form_0[comb_form_0[:,2].argsort()]                 # sort by values (col 2)
            comb_form_0 = comb_form_0[comb_form_0[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_0 = comb_form_0[comb_form_0[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
    
            
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
                        
            comb_form_1 = np.hstack(( np.transpose(np.array([rows_1])),
                                      np.transpose(np.array([cols_1])),
                                      np.transpose(np.array([values_1])) ))
            
            comb_form_1 = comb_form_1[comb_form_1[:,2].argsort()]                 # sort by values (col 2)
            comb_form_1 = comb_form_1[comb_form_1[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_1 = comb_form_1[comb_form_1[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
            
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
            
            comb_form_2 = np.hstack(( np.transpose(np.array([rows_2])),
                                      np.transpose(np.array([cols_2])),
                                      np.transpose(np.array([values_2])) ))
            
            comb_form_2 = comb_form_2[comb_form_2[:,2].argsort()]                 # sort by values (col 3)
            comb_form_2 = comb_form_2[comb_form_2[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_2 = comb_form_2[comb_form_2[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
            
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
            
            comb_form_3 = np.hstack(( np.transpose(np.array([rows_3])),
                                      np.transpose(np.array([cols_3])),
                                      np.transpose(np.array([values_3])) ))
            
            comb_form_3 = comb_form_3[comb_form_3[:,2].argsort()]                 # sort by values (col 2)
            comb_form_3 = comb_form_3[comb_form_3[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_3 = comb_form_3[comb_form_3[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
            
            
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
            
            comb_form_1 = np.hstack(( np.transpose(np.array([rows_1])),
                                      np.transpose(np.array([cols_1])),
                                      np.transpose(np.array([values_1])) ))
            
            comb_form_1 = comb_form_1[comb_form_1[:,2].argsort()]                 # sort by alues (col 2)
            comb_form_1 = comb_form_1[comb_form_1[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_1 = comb_form_1[comb_form_1[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
            
            ### --- FACES --------------------
            
            # This matrix transforms faces into their constituent edges, taking into account relative orientations. The rows correspond
            # to edges and the columns correspond to faces.
                    
            cols_2 = list(range(0, np.shape(cells_2D)[0])) * np.shape(f2e)[1]
            cols_2.sort()
            
            rows_2 = list(abs(f2e).flatten())
            
            values_2 = np.sign(f2e.flatten())
            
            values_2[np.argwhere(values_2 == 0)] = 1
            
            comb_form_2 = np.hstack(( np.transpose(np.array([rows_2])),
                                      np.transpose(np.array([cols_2])),
                                      np.transpose(np.array([values_2])) ))
            
            comb_form_2 = comb_form_2[comb_form_2[:,2].argsort()]                 # sort by values (col 2)
            comb_form_2 = comb_form_2[comb_form_2[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_2 = comb_form_2[comb_form_2[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
            
            ### --- VOLUMES --------------------
            
            # This matrix transforms volumes into their constituent faces, taking into account relative orientations. The rows correspond
            # to faces and the columns correspond to volumes.
                    
            cols_3 = list(range(0, np.shape(cells_3D)[0])) * np.shape(v2f)[1]
            cols_3.sort()
            
            rows_3 = list(abs(v2f).flatten())
            
            values_3 = np.sign(v2f.flatten())
            
            values_3[np.argwhere(values_3 == 0)] = 1
            
            comb_form_3 = np.hstack(( np.transpose(np.array([rows_3])),
                                      np.transpose(np.array([cols_3])),
                                      np.transpose(np.array([values_3])) ))
            
            comb_form_3 = comb_form_3[comb_form_3[:,2].argsort()]                 # sort by values (col 2)
            comb_form_3 = comb_form_3[comb_form_3[:,1].argsort(kind='mergesort')] # sort by cols (col 1)
            comb_form_3 = comb_form_3[comb_form_3[:,0].argsort(kind='mergesort')] # sort by rows (col 0)
            
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
            
            if (structure == 'bcc' or structure == 'fcc'):
                
                v2e = np.empty((0,6)) # Need to find which edges constitute which volumes.
                
            elif structure == 'simple cubic':
                
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
    
    except: print("Something went wrong with the function combinatorial_form().")



    
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




def convert_from_cscmatrix(matrix):
    """
    Parameters
    ----------
    matrix : sparse.csc_matrix
        A sparse matrix.

    Returns
    -------
    The same sparse matrix but with row, column and value entries laid out in a np array (N x 3).
    """


