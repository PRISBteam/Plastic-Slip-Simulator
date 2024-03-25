"""
Created on Tue Oct 25 2022

Last edited on: Mar 13 12:00 2024

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. It is meant to be run from a command line/terminal in a directory containing
the dccstructure package. It executes the whole package from scratch as intended, building a discrete (simplicial) cell complex
with the parameters specified. It does not return the nodes, edges, faces, volumes, but rather the topological matrices (incidence
and adjacency) and other topological and geometric information pertinent for the discrete Naimark microshear model.

This file can be run with the command

    python -m dccstructure (+ arguments)

Arguments that can be passed:
    
    Mandatory:
        
        --size int int int : specifies the number of unit cells in the directions x, y, z (default is 1 1 1);
        --struc str : specifies the crystallographic structure of the complex;
        
    Optional:
        
        --dim int : specifies the dimension of the complex (default is 3);
        --basisv flt flt flt flt flt flt flt flt flt : specifies the 9 components of the 3 lattice basis vectors;
        --mp : if passed, the code will run with Python's multiprocessing package, i.e. paralellisation of operations. This will not work if all of the complex dimensions (as passed to --size) are smaller than the number of available CPUs;
        -d : if passed, the code will output the node degrees;
        -n : if passed, the code will output the unit normals to the 2-cells;
        -a : if passed, the code will output the areas of the 2-cells;
        -v : if passed, the code will output the volumes of the 3-cells;
        -s : if passed, the code will output the indices of 2-cells corresponding to slip planes;

"""


import argparse
import numpy as np
import multiprocessing as mp
import os

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')

from dccstructure import build as build
from dccstructure import matrices as mat
from dccstructure import orientations as ori
from dccstructure.geometry import unit_normal, geo_measure, tetrahedron_volume
from dccstructure.iofiles import write_to_file


# Set the arguments for the command line / terminal
parser = argparse.ArgumentParser(description='Some description')

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
    '--mp',
    action = 'store_true',
    required = False,
    help = "If True, employs Python's multiprocessing package to parallelise operations and speed up processing. This will not work if all of the complex dimensions (as passed to --size) are smaller than the number of available CPUs."
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
    '-v',
    action = 'store_true',
    required = False,
    help = "Whether to also output the volumes of 3-cells."
)

parser.add_argument(
    '-s',
    action = 'store_true',
    required = False,
    help = "Whether to also output the indices of the 2-cells corresponding to slip planes."
)



# Sort out variables from the arguments
    
args = parser.parse_args()

DIM = args.dim
STRUC = args.struc
SIZE = args.size
MULT = args.mp
degrees_yes = args.d
normals_yes = args.n
areas_yes = args.a
vols_yes = args.v
slips_yes = args.s

try:
    
    LATTICE = np.zeros((3,3))
    
    for i in range(3):
            
            LATTICE[i,0] = args.basisv[[0,1,2]]
            LATTICE[i,1] = args.basisv[[3,4,5]]
            LATTICE[i,2] = args.basisv[[6,7,8]]

except:
    
    LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]])

if np.all(MULT == True and [x < os.cpu_count() + 1 for x in SIZE]):

    print("\nWarning! Due to constraints in division of labour, the multiprocessing feature is only viable when the number " +
          "of unit cells of the complex is greater than the number of available CPUs in at least one of the complex's dimensions.\n")    

if __name__ == '__main__':
    
    
    # Execute the complex
    
    results = build.build_complex(struc = STRUC,
                                  size = SIZE,
                                  lattice = LATTICE,
                                  dim = DIM,
                                  multiprocess = MULT)

    nodes = results[0] ; edges = results[1] ; faces = results[2] ; faces_slip = results[3] ; volumes = results[4]
    
    
    # Define relative orientations
    
    v2f, f2e = ori.get_orientations(cells0D = nodes,
                                    cells1D = edges,
                                    cells2D = faces,
                                    cells3D = volumes,
                                    faces_per_volume = 4,
                                    edges_per_face = 3)    
    
    
    # Compute MATRICES in all structures
    
    A0, A1, A2, A3, B1, B2, B3, node_degrees = mat.get_matrices(nodes, edges, faces, volumes, f2e, v2f)
    
    write_to_file(A0, 'A0',
                  A1, 'A1',
                  A2, 'A2',
                  A3, 'A3',
                  B1, 'B1',
                  B2, 'B2',
                  B3, 'B3',
                  new_folder = True)
    
    write_to_file(nodes, 'nodes',
                  edges, 'edges',
                  faces, 'faces',
                  faces_slip, 'faces_slip',
                  volumes, 'volumes',
                  f2e, 'faces_to_edges',
                  v2f, 'volumes_to_faces',
                  new_folder = False)

    
    del A0, A1, A2, A3, B1, B2, B3
    
    nrs_cells = np.array([[len(nodes)], [len(edges)], [len(faces)], [len(volumes)]])
    write_to_file(nrs_cells, 'nr_cells', new_folder = False)
    
    del nrs_cells
    
        
    if degrees_yes:
        write_to_file(node_degrees, 'node_degrees', new_folder = False)
        del node_degrees
        
    if normals_yes:
        normals = []
        with mp.Pool() as pool:
            for result in pool.imap(unit_normal, nodes[faces]):
                normals.append(result.astype(float))
        write_to_file(np.array(normals), 'faces_normals', new_folder = False)
        del result
        
    if areas_yes:
        areas = []
        with mp.Pool() as pool: 
            for result in pool.imap(geo_measure, nodes[faces]):
                areas.append(float(result))
        write_to_file(areas, 'faces_areas', new_folder = False)
        del result
        
    if vols_yes:
        volumes_vols = []
        with mp.Pool() as pool:
            for result in pool.imap(tetrahedron_volume, nodes[volumes]):
                volumes_vols.append(float(result))
        write_to_file(volumes_vols, 'volumes_vols', new_folder = False)
        del result, volumes_vols

    if slips_yes: 
        write_to_file(faces_slip, 'faces_slip', new_folder = False)
    
    
