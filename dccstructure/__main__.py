"""
Created on Sat Jul 2 2022

Last edited on: 17/10/2022 14:30

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
        -d bool : if True, the code will output the node degrees;
        -n bool : if True, the code will output the unit normals to the 2-cells;
        -a bool : if True, the code will output the areas of the 2-cells;
        -s bool : if True, the code will output the indices of 2-cells corresponding to slip planes;

"""


import argparse
import numpy as np

from dccstructure import build
from dccstructure import matrices
from dccstructure import orientations
from dccstructure.geometry import unit_normal, geo_measure
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


# Sort out variables from the arguments
    
args = parser.parse_args()

DIM = args.dim
STRUC = args.struc
SIZE = args.size
degrees_yes = args.d
normals_yes = args.n
areas_yes = args.a
slips_yes = args.s

try:
    
    LATTICE = np.zeros((3,3))
    
    for i in range(3):
            
            LATTICE[i,0] = args.basisv[[0,1,2]]
            LATTICE[i,1] = args.basisv[[3,4,5]]
            LATTICE[i,2] = args.basisv[[6,7,8]]

except:
    
    LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]])
    print('/n/nDue to error in parsing argument, the lattice basis vectors have been changed to [1,0,0], [0,1,0] and [0,0,1]./n/n') ################
    

# Execute the complex

nodes, edges, faces, faces_slip, volumes = build.build_complex(struc = STRUC,
                                                               size = SIZE,
                                                               lattice = LATTICE,
                                                               dim = DIM,
                                                               extras = False)

if STRUC == 'simple cubic':
    
    faces_as_edges = orientations.faces_to_edges(faces, edges, edges_per_face = 4)

    volumes_as_faces = orientations.volumes_to_faces(volumes, faces, faces_per_volume = 6)



elif STRUC in ['bcc', 'fcc']:

    faces_as_edges = orientations.faces_to_edges(faces, edges)

    volumes_as_faces = orientations.volumes_to_faces(volumes, faces)            


# MATRICES in all structures

volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                           cells_2D = faces,
                                                                           cells_1D = edges,
                                                                           cells_0D = nodes,
                                                                           v2f = volumes_as_faces,
                                                                           f2e = faces_as_edges)

incidence_matrices = matrices.combinatorial_form(structure = STRUC,
                                                 degree = 1,
                                                 cells_3D = volumes,
                                                 cells_2D = faces,
                                                 cells_1D = edges,
                                                 cells_0D = nodes,
                                                 v2f = volumes_as_faces,
                                                 f2e = faces_as_edges)

write_to_file(incidence_matrices[1], 'B1',
              incidence_matrices[2], 'B2',
              incidence_matrices[3], 'B3',
              new_folder = True)

del incidence_matrices



adjacency_matrices = matrices.combinatorial_form(structure = STRUC,
                                                 degree = 0,
                                                 cells_3D = volumes,
                                                 cells_2D = faces,
                                                 cells_1D = edges,
                                                 cells_0D = nodes,
                                                 v2f = volumes_as_faces,
                                                 f2e = faces_as_edges)

write_to_file(adjacency_matrices[0], 'A0',
              adjacency_matrices[1], 'A1',
              adjacency_matrices[2], 'A2',
              adjacency_matrices[3], 'A3',
              new_folder = False)

del adjacency_matrices



nrs_cells = np.array([[len(nodes)], [len(edges)], [len(faces)], [len(faces_slip)], [len(volumes)]])

write_to_file(nrs_cells, 'number_of_cells',
              new_folder = False)

del nrs_cells


    
if degrees_yes:
    
    node_degrees = matrices.degree_distribution(edges, nodes)
    
    write_to_file(node_degrees, 'node_degrees', new_folder = False)
    
    del node_degrees
    
    
    
if normals_yes:
    
    normals = np.array([unit_normal(i) for i in nodes[faces]])
    
    write_to_file(normals, 'normals', new_folder = False)
    
    del normals
    
    
    
if areas_yes:
    
    areas = [geo_measure(i) for i in nodes[faces]]
    
    write_to_file(areas, 'faces_areas', new_folder = False)
    
    del areas
    
    
    
if slips_yes:
    
    write_to_file(faces_slip, 'faces_slip', new_folder = False)


