# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:00 2022

Last edited on: 12/07/2022 12:10

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions pertinent to the construction of the cell complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import argparse
import numpy as np

import os

directory = os.getcwd().split('/')

if 'dccstructure' in directory:
    
    directory.remove('dccstructure')
    
    directory = '/'.join(str(i) for i in directory)
    
    os.chdir(directory)
    
    from dccstructure import build
    from dccstructure.iofiles import write_to_file
    
    directory = os.path.join(directory, r'dccstructure')
       
    os.chdir(directory)
    
else:
    
    from dccstructure import build
    from dccstructure.iofiles import write_to_file
    
del directory



# ----- # ----- # FUNCTIONS # ------ # ----- #


def build_complex(struc, size, lattice=[[1,0,0],[0,1,0],[0,0,1]], dim=3):
    """
    Parameters
    ----------
    struc : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.
    lattice : TYPE, optional
        DESCRIPTION. The default is [[1,0,0],[0,1,0],[0,0,1]].
    dim : TYPE, optional
        DESCRIPTION. The default is 3.
    d : TYPE, optional
        DESCRIPTION. The default is False.
    n : TYPE, optional
        DESCRIPTION. The default is True.
    s : TYPE, optional
        DESCRIPTION. The default is True.
    r : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    
    
    Notes
    -------
nodes, edges, faces, volumes = build_complex(struc, size, lattice=[[1,0,0],[0,1,0],[0,0,1]], dim=3)
    
    """
    
    if type(lattice) == list:
        
        lattice = np.array(lattice)

    first_u_cell = np.array([[0,0,0]]) + np.sum(lattice, axis=0) / 2


    if struc == 'simple cubic':
        
        #------- NODES in SC

        first_node = first_u_cell - np.sum(lattice, axis=0) / 2
        
        nodes = build.create_nodes(structure = struc,
                                   origin = first_node,
                                   lattice = lattice,
                                   size = size,
                                   dim = dim)
        
        #------- EDGES in SC
        
        edges = build.find_neighbours(nodes, lattice, structure = struc, dim = dim)
                            
        #------- FACES in SC
        
        faces = build.create_faces(edges, structure = struc)
                
        #------- VOLUMES in SC
        
        volumes = build.create_volumes(lattice, struc, cells_0D = nodes)
        
        
        
    elif struc == 'bcc':
        
        #------- NODES in BCC

        first_node = first_u_cell - np.sum(lattice, axis=0) / 2
        
        nodes, (nodes_sc, nodes_bcc, nodes_virtual) = build.create_nodes(structure = struc,
                                                                         origin = first_node,
                                                                         lattice = lattice,
                                                                         size = size,
                                                                         dim = dim,
                                                                         axis = 2)
        
        #------- EDGES in BCC
        
        edges, (edges_sc, edges_bcc, edges_virtual) = build.find_neighbours(nodes,
                                                                            lattice,
                                                                            structure = struc,
                                                                            dim = dim,
                                                                            special_0D = (nodes[nodes_sc],
                                                                                          nodes[nodes_bcc],
                                                                                          nodes[nodes_virtual]))
        
        del edges_sc, edges_bcc, edges_virtual
        
                                
        #------- FACES in BCC
        
        faces, faces_sc = build.create_faces(edges, structure = struc, cells_0D = nodes)        
                    
        #------- VOLUMES in BCC
            
        volumes = build.create_volumes(lattice, struc, cells_2D = faces)
        

        
    elif struc == 'fcc':

        #------- NODES in FCC
                
        first_node = first_u_cell - (lattice[0] + lattice[1] + lattice[2]) / 2

        nodes, (nodes_sc, nodes_bcc, nodes_fcc) = build.create_nodes(structure = 'bcc',
                                                                     origin = first_node,
                                                                     lattice = lattice,
                                                                     size = size,
                                                                     dim = dim)
        #------- EDGES in FCC

        edges, edges_sc, edges_bcc_fcc, edges_fcc2, edges_fcc_sc = build.find_neighbours(nodes,
                                                                                         lattice = lattice,
                                                                                         structure = struc,
                                                                                         dim = dim,
                                                                                         special_0D = (nodes_sc, nodes_bcc, nodes_fcc))
        
        #------- FACES in FCC
        
        faces, faces_slip = build.create_faces(edges,
                                               structure = struc,
                                               cells_0D = (nodes, nodes_sc, nodes_bcc, nodes_fcc))
                        
        #------- VOLUMES in FCC
        
        volumes = build.create_volumes(lattice, struc, cells_2D = faces)
        

    return nodes, edges, faces, volumes





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Some description')

    parser.add_argument(
        '--dim',
        action ='store',
        default = 3,
        type = int,
        choices = [2, 3],
        required = True,
        help = 'The spatial dimension of the complex (2 or 3).'
    )

    parser.add_argument(
        '--size',
        action = 'store',
        nargs = 3,
        default = [1,1,1],
        type = int,
        required = True,
        help = 'The number of unit cells in each spatial direction (each unit cell is bounded by 8 nodes in simple cubic-like positions). ' +
             'For an FCC structure in the complex, each unit cell has 28 3-cells; for BCC, each unit cell has 24 3-cells.'
    )

    parser.add_argument(
        '--struc',
        action = 'store',
        default = 'bcc',
        choices = ['simple cubic', 'bcc', 'fcc', 'hcp'],
        required = True,
        help = "The complex's lattice structure. Choose from: simple cubic, bcc, fcc or hcp."
    )
    
    parser.add_argument(
        '--basisv',
        action = 'store',
        nargs = 9,
        default = [1, 0, 0, 0, 1, 0, 0, 0, 1],
        type = int,
        required = False,
        help = "The basis vectors of the complex's lattice (vectors between corners of unit cells in each spatial direction)."
    )
    
    parser.add_argument(
        '-d',
        action = 'store_true',
        required = False,
        help = "Whether to also output the node degree matrix."
    )
    
    parser.add_argument(
        '-n',
        action = 'store_true',
        required = False,
        help = "Whether to also output the unit normal vectors to the 2-cells."
    )

    parser.add_argument(
        '-a',
        action = 'store_true',
        required = False,
        help = "Whether to also output the areas of 2-cells."
    )

    parser.add_argument(
        '-s',
        action = 'store_true',
        required = False,
        help = "Whether to also output the indices of the 2-cells corresponding to slip planes."
    )
    
    parser.add_argument(
        '-r',
        action = 'store_true',
        required = False,
        help = "Whether to also output the 'results.txt' file from iofiles.write_to_file()."
    )
    
    # Sort out variables from the arguments
        
    args = parser.parse_args()

    DIM = args.dim
    STRUC = args.struc
    SIZE = args.size
    degrees_yes = args.d
    normals_yes = args.n
    results_yes = args.r
    areas_yes = args.a
    slip_yes = args.s
    
    try:
        
        LATTICE = np.zeros((3,3))
                        
        LATTICE[0,:] = args.basisv[[0,1,2]]
        LATTICE[1,:] = args.basisv[[3,4,5]]
        LATTICE[2,:] = args.basisv[[6,7,8]]
    
    except:
        LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]]) ################


    # Build the complex

    nodes, edges, faces, volumes = build_complex(struc = STRUC,
                                                 size = SIZE,
                                                 lattice = LATTICE,
                                                 dim = DIM)
        
    write_to_file(nodes, 'nodes',
                  edges, 'edges',
                  faces, 'faces',
                  volumes, 'volumes',
                  new_folder = True)


