# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 14:40 2022

Last edited on: 30/06/2022 18:05

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the DCC_Structure package. In here you will find a useful function for executing the whole package from
setting the nodes to obtaining the adjacency matrices and oriented incidence matrices.
"""

import argparse
import numpy as np
import build
import matrices
import orientations
from geometry import unit_normal, geo_measure
from iofiles import write_to_file

# https://docs.python.org/3/library/argparse.html

def exe_main():
    """
    """
    
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
        LATTICE = np.array([[1,0,0],[0,1,0],[0,0,1]]) ################
        
    
    # Execute the complex

    first_u_cell = np.array([[0,0,0]]) + np.sum(LATTICE, axis=0) / 2


    if STRUC == 'simple cubic':

        #------- NODES in SC

        first_node = first_u_cell - np.sum(LATTICE, axis=0) / 2

        nodes = build.create_nodes(structure = STRUC,
                                   origin = first_node,
                                   lattice = LATTICE,
                                   size = SIZE,
                                   dim = DIM)

        #------- EDGES in SC

        edges = build.find_neighbours(nodes, LATTICE, structure = STRUC, dim=DIM)

        #------- FACES in SC

        faces = build.create_faces(edges, structure = STRUC)

        faces_as_edges = orientations.faces_to_edges(faces, edges, edges_per_face = 4)

        #------- VOLUMES in SC

        volumes = build.create_volumes(LATTICE, STRUC, cells_0D = nodes)

        volumes_as_faces = orientations.volumes_to_faces(volumes, faces, faces_per_volume = 6)



    elif STRUC == 'bcc':

        #------- NODES in BCC

        first_node = first_u_cell - np.sum(LATTICE, axis=0) / 2

        nodes, (nodes_sc, nodes_bcc, nodes_virtual) = build.create_nodes(structure = STRUC,
                                                                         origin = first_node,
                                                                         lattice = LATTICE,
                                                                         size = SIZE,
                                                                         dim = DIM,
                                                                         axis = 2)

        #------- EDGES in BCC

        edges, edges_sc, edges_bcc, edges_virtual = build.find_neighbours(nodes, LATTICE,
                                                                          structure = STRUC,
                                                                          dim=DIM,
                                                                          special_0D = (nodes[nodes_sc],
                                                                                        nodes[nodes_bcc],
                                                                                        nodes[nodes_virtual]))

        #------- FACES in BCC

        faces, faces_sc = build.create_faces(edges, structure = STRUC, cells_0D = nodes)
        
        faces_slip = np.array(list(range(len(faces))))
        
        faces_slip = list(np.delete(faces_slip, faces_sc))

        faces_as_edges = orientations.faces_to_edges(faces, edges)

        #------- VOLUMES in BCC

        volumes = build.create_volumes(LATTICE, STRUC, cells_2D = faces)

        volumes_as_faces = orientations.volumes_to_faces(volumes, faces)        
        


    elif STRUC == 'fcc':

        #------- NODES in FCC

        first_node = first_u_cell - (LATTICE[0] + LATTICE[1] + LATTICE[2]) / 2

        nodes, (nodes_sc, nodes_bcc, nodes_fcc) = build.create_nodes(structure = 'bcc',
                                                                     origin = first_node,
                                                                     lattice = LATTICE,
                                                                     size = SIZE,
                                                                     dim = DIM)
                
        #------- EDGES in FCC

        edges, edges_sc, edges_bcc_fcc, edges_fcc2, edges_fcc_sc = build.find_neighbours(nodes,
                                                                                         lattice = LATTICE,
                                                                                         structure = STRUC,
                                                                                         dim = DIM,
                                                                                         special_0D = (nodes_sc, nodes_bcc, nodes_fcc))
        
        del edges_sc, edges_bcc_fcc, edges_fcc2, edges_fcc_sc
        
        #------- FACES in FCC

        faces, faces_slip = build.create_faces(edges,
                                               structure = STRUC,
                                               cells_0D = (nodes, nodes_sc, nodes_bcc, nodes_fcc))

        faces_as_edges = orientations.faces_to_edges(faces, edges)

        #------- VOLUMES in FCC

        volumes = build.create_volumes(LATTICE, STRUC, cells_2D = faces)

        volumes_as_faces = orientations.volumes_to_faces(volumes, faces)
        
    
    
    #------- MATRICES in all structures

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



    nrs_cells = np.array([[len(nodes)], [len(edges)], [len(faces)], [len(volumes)]])

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
        
    del faces_slip



""" ----------------------------------------------------------------------------------------- """



if __name__ == "__main__":
    exe_main()
    
    