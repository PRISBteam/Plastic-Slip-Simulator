# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:00 2022

Last edited on: 17/10/2022 16:55

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. It is meant to be run from a command line/terminal in the directory of the
dccstructure package. It executes the whole package from scratch as intended, building a discrete (simplicial) cell complex
with the parameters specified. It returns the nodes, edges, faces and volumes as well as other optional topological information.

This file can be run with the command

    python build_complex.py (+ arguments)

Arguments that can be passed:
    
    Mandatory:
        
        --size int int int : specifies the number of unit cells in the directions x, y, z (default is 1 1 1);
        --struc str : specifies the crystallographic structure of the complex;
        
    Optional:
        
        --dim int : specifies the dimension of the complex. The default is 3.
        --basisv flt flt flt flt flt flt flt flt flt : specifies the 9 components of the 3 lattice basis vectors. The default is a unit vector in each canonical direction.
        -e bool : if True, the code will output additional topological information. There is no extra topological information for the simple cubic structure. The default is False.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import argparse
import numpy as np

import os


# ----- # ----- # CODE # ------ # ----- #



if __name__ == "__main__":
    
    
    
    directory = os.getcwd().split('/')

    if 'dccstructure' in directory:
        
        directory.remove('dccstructure')
        
        directory = '/'.join(str(i) for i in directory)
        
        os.chdir(directory)
        
        import dccstructure.build as build
        from dccstructure.iofiles import write_to_file
        
        directory = os.path.join(directory, r'dccstructure')
           
        os.chdir(directory)
        
    else:
        
        from dccstructure import build
        from dccstructure.iofiles import write_to_file
        
    del directory

    

    parser = argparse.ArgumentParser(description='Some description')

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
        '--dim',
        action ='store',
        default = 3,
        type = int,
        choices = [2, 3],
        required = False,
        help = 'The spatial dimension of the complex (2 or 3).'
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
        '-e',
        action = 'store_true',
        default = False,
        required = False,
        help = "Whether to output additional topological information about the cell complex."
    )

    
    
    # Sort out variables from the arguments
        
    args = parser.parse_args()

    DIM = args.dim
    STRUC = args.struc
    SIZE = args.size
    extras_yes = args.e
    
    try:
        
        LATTICE = np.zeros((3,3))
                        
        LATTICE[0,:] = args.basisv[[0,1,2]]
        LATTICE[1,:] = args.basisv[[3,4,5]]
        LATTICE[2,:] = args.basisv[[6,7,8]]
    
    except:
        
        LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]]) ################


    # Build the complex

    results = build.build_complex(struc = STRUC,
                                  size = SIZE,
                                  lattice = LATTICE,
                                  dim = DIM,
                                  extras = extras_yes)
    
    # Print the results into .txt files in a new folder
        
    if extras_yes == False:
        
        nodes = results[0] ; edges = results[1] ; faces = results[2] ; faces_slip = results[3] ; volumes = results[4]
        
        write_to_file(nodes, 'nodes',
                      edges, 'edges',
                      faces, 'faces',
                      faces_slip, 'faces_slip',
                      volumes, 'volumes',
                      new_folder = True)
    
    elif extras_yes == True and STRUC == 'bcc':
        
        write_to_file(results[0], 'nodes',
                      results[1], 'nodes_sc',
                      results[2], 'nodes_bcc',
                      results[3], 'nodes_virtual',
                      results[4], 'edges',
                      results[5], 'edges_sc',
                      results[6], 'edges_bcc',
                      results[7], 'edges_virtual',
                      results[8], 'faces',
                      results[9], 'faces_slip',
                      results[10], 'volumes',
                      new_folder = True)
        
    elif extras_yes == True and STRUC == 'fcc':
        
        write_to_file(results[0], 'nodes',
                      results[1], 'nodes_sc',
                      results[2], 'nodes_bcc',
                      results[3], 'nodes_fcc',
                      results[4], 'edges',
                      results[5], 'edges_sc',
                      results[6], 'edges_bcc_fcc',
                      results[7], 'edges_fcc2',
                      results[8], 'edges_fcc_sc',
                      results[9], 'faces',
                      results[10], 'faces_slip',
                      results[12], 'volumes',
                      new_folder = True)




