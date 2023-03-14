# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2022

Last edited on: Mar 14 11:20 2023

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. It is meant to be run from a command line/terminal in the directory of the
dccstructure package. It executes the whole package from scratch as intended, building a discrete (simplicial) cell complex
with the parameters specified. It returns the nodes, edges, faces and volumes as well as other optional topological information.

This file can be run with the command

    python build_complex_mp.py (+ arguments)

Arguments that can be passed:
    
    Mandatory:
        
        --size int int int : specifies the number of unit cells in the directions x, y, z (default is 1 1 1);
        --struc str : specifies the crystallographic structure of the complex;
        
    Optional:
        
        --dim int : specifies the dimension of the complex. The default is 3.
        --basisv flt flt flt flt flt flt flt flt flt : specifies the 9 components of the 3 lattice basis vectors. The default is a unit vector in each canonical direction.
        --mp bool : if True, the code will run with Python's multiprocessing package, i.e. paralellisation of operations.
        -e bool : if True, the code will output additional geometric information. There is no extra information for the simple cubic structure. The default is False.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import argparse
import numpy as np
import multiprocessing as mp

import sys
sys.path.append('../')

import dccstructure_mp.build as build
from dccstructure_mp.orientations import get_orientations
from dccstructure_mp.geometry import unit_normal, geo_measure
from dccstructure_mp.iofiles import write_to_file


# ----- # ----- # CODE # ------ # ----- #



if __name__ == "__main__":
    

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
        '--mp',
        action = 'store_true',
        required = False,
        help = "If True, employs Python's multiprocessing package to parallelise operations and speed up processing."
    )
    
    parser.add_argument(
        '-e',
        action = 'store_true',
        default = False,
        required = False,
        help = "Whether to output additional geometrical information about the cell complex."
    )

    
    
    # Sort out variables from the arguments
        
    args = parser.parse_args()

    DIM = args.dim
    STRUC = args.struc
    SIZE = args.size
    MULT = args.mp
    extras_yes = args.e
    
    try:
        
        LATTICE = np.zeros((3,3))
                        
        LATTICE[0,:] = args.basisv[[0,1,2]]
        LATTICE[1,:] = args.basisv[[3,4,5]]
        LATTICE[2,:] = args.basisv[[6,7,8]]
    
    except:
        
        LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]])
        print('-> Due to error in parsing argument, the lattice basis vectors have been changed to [1,0,0], [0,1,0] and [0,0,1].')

    # Build the complex
    
    with mp.Pool() as pool:

        results = build.build_complex(struc = STRUC,
                                      size = SIZE,
                                      lattice = LATTICE,
                                      dim = DIM,
                                      multiprocess = MULT)
    
    nodes = results[0] ; edges = results[1] ; faces = results[2] ; faces_slip = results[3] ; volumes = results[4]
    
    # Define relative orientations
    
    v2f, f2e = get_orientations(cells0D = nodes,
                                cells1D = edges,
                                cells2D = faces,
                                cells3D = volumes,
                                faces_per_volume = 4,
                                edges_per_face = 3)    
    
    # Print the results into .txt files in a new folder
    
    if extras_yes == False:
        write_to_file(nodes, 'nodes',
                      edges, 'edges',
                      faces, 'faces',
                      faces_slip, 'faces_slip',
                      volumes, 'volumes',
                      f2e, 'faces_to_edges',
                      v2f, 'volumes_to_faces',
                      new_folder = True)
    
    elif extras_yes == True:
        
        normals = []
        
        with mp.Pool() as pool:
            
            for result in pool.imap(unit_normal, nodes[faces]):
                
                normals.append(result)
                
        write_to_file(normals, 'normals', new_folder = False)
        
        del normals
        
        areas = []
        
        with mp.Pool() as pool:
            
            for result in pool.imap(geo_measure, nodes[faces]):
                
                areas.append(result)
                
        write_to_file(areas, 'faces_areas', new_folder = False)
        
        del areas



