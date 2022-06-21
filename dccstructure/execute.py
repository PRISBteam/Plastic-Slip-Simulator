# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 14:40 2022

Last edited on: 08/06/2022 17:10

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the DCC_Structure package. In here you will find a useful function for executing the whole package from
setting the nodes to obtaining the adjacency matrices and oriented incidence matrices.

"""

import argparse
import numpy as np
import build
import matrices
import orientations
from iofiles import write_to_file

# https://docs.python.org/3/library/argparse.html

def main():
    """
    """
    
    # Set the arguments for the command line / terminal
    
    parser = argparse.ArgumentParser(description='Some description')

    parser.add_argument(
        '--dim',
        action='store',
        default=3,
        type=int,
        choices=[2, 3],
        required=False,
        help='The spatial dimension of the complex (2 or 3).'
    )

    parser.add_argument(
        '-size',
        action='store',
        nargs=3,
        default=[1,1,1],
        type=int,
        required=True,
        help='The number of unit cells in each spatial direction (each unit cell is bounded by 8 nodes in simple cubic-like positions). ' +
             'For an FCC structure in the complex, each unit cell has 28 3-cells; for BCC, each unit cell has 24 3-cells.'
    )

    parser.add_argument(
        '-struc',
        action='store',
        default='bcc',
        choices=['simple cubic', 'bcc', 'fcc', 'hcp'],
        required=True,
        help="The complex's lattice structure. Choose from: simple cubic, bcc, fcc or hcp."
    )
    
    parser.add_argument(
        '--basisv',
        action='store',
        nargs=3, # or 9?
        default=[[1,0,0],[0,1,0],[0,0,1]],
        type=list, # or int?
        required=False,
        help="The basis vectors of the complex's lattice (vectors between nodes in each spatial direction)."
    )
    
    parser.add_argument(
        '--degrees',
        action='store_true',
        #default=False,
        #type=bool,
        #choices=[True, False],
        required=False,
        help="Whether to also output the node degree matrix."
    )    
    
    # Execute the complex
    
    try:
        
        args = parser.parse_args()
        
        DIM = args.dim
        STRUC = args.struc
        SIZE = args.size
        LATTICE = np.array(args.basisv)
        degree_yes = args.degrees

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
                        
            if degree_yes:
                node_degrees = matrices.degree_distribution(edges, nodes)
            
            #------- FACES in SC
            
            faces = build.create_faces(edges, structure = STRUC)
            
            faces_as_edges = orientations.faces_to_edges(faces, edges, edges_per_face = 4)
            
            #------- VOLUMES in SC
            
            volumes = build.create_volumes(LATTICE, STRUC, cells_0D = nodes)
            
            volumes_as_faces = orientations.volumes_to_faces(volumes, faces, faces_per_volume = 6)
                        
            volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                                            cells_2D = faces,
                                                                                            cells_1D = edges,
                                                                                            cells_0D = nodes,
                                                                                            v2f = volumes_as_faces,
                                                                                            f2e = faces_as_edges)
        
            #------- MATRICES in SC
                    
            adjacency_matrices = matrices.combinatorial_form(structure = STRUC,
                                                                    degree = 0,
                                                                    cells_3D = volumes,
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
            
            edges, (edges_sc, edges_bcc, edges_virtual) = build.find_neighbours(nodes, LATTICE,
                                                                                    structure = STRUC,
                                                                                    dim=DIM,
                                                                                    special_0D = (nodes[nodes_sc],
                                                                                                    nodes[nodes_bcc],
                                                                                                    nodes[nodes_virtual]))
                            
            if degree_yes:
                node_degrees = matrices.degree_distribution(edges, nodes)
            
            #------- FACES in BCC
            
            faces, faces_sc = build.create_faces(edges, structure = STRUC, cells_0D = nodes)
            
            faces_as_edges = orientations.faces_to_edges(faces, edges)
                        
            #------- VOLUMES in BCC
                
            volumes = build.create_volumes(LATTICE, STRUC, cells_2D = faces)
            
            volumes_as_faces = orientations.volumes_to_faces(volumes, faces)
                        
            volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                                            cells_2D = faces,
                                                                                            cells_1D = edges,
                                                                                            cells_0D = nodes,
                                                                                            v2f = volumes_as_faces,
                                                                                            f2e = faces_as_edges)
        
            #------- MATRICES in BCC
                        
            adjacency_matrices = matrices.combinatorial_form(structure = STRUC,
                                                                    degree = 0,
                                                                    cells_3D = volumes,
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
                            
            if degree_yes:
                node_degrees = matrices.degree_distribution(edges, nodes)
        
            
            #------- FACES in FCC
            
            faces, faces_slip = build.create_faces(edges,
                                                        structure = STRUC,
                                                        cells_0D = (nodes, nodes_sc, nodes_bcc, nodes_fcc))
        
            faces_as_edges = orientations.faces_to_edges(faces, edges)
                        
            #------- VOLUMES in FCC
            
            volumes = build.create_volumes(LATTICE, STRUC, cells_2D = faces)
            
            volumes_as_faces = orientations.volumes_to_faces(volumes, faces)
                        
            volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                                            cells_2D = faces,
                                                                                            cells_1D = edges,
                                                                                            cells_0D = nodes,
                                                                                            v2f = volumes_as_faces,
                                                                                            f2e = faces_as_edges)
        
            #------- MATRICES in FCC
                    
            adjacency_matrices = matrices.combinatorial_form(structure = STRUC,
                                                                    degree = 0,
                                                                    cells_3D = volumes,
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

        if degree_yes:

            write_to_file('result.txt',
                                    node_degrees, 'node_degrees',
                                    adjacency_matrices[0], 'A0',
                                    adjacency_matrices[1], 'A1',
                                    adjacency_matrices[2], 'A2',
                                    adjacency_matrices[3], 'A3',
                                    incidence_matrices[1], 'B10',
                                    incidence_matrices[2], 'B21',
                                    incidence_matrices[3], 'B32')
        
        else:

            write_to_file('result.txt',
                        adjacency_matrices[0], 'A0',
                        adjacency_matrices[1], 'A1',
                        adjacency_matrices[2], 'A2',
                        adjacency_matrices[3], 'A3',
                        incidence_matrices[1], 'B10',
                        incidence_matrices[2], 'B21',
                        incidence_matrices[3], 'B32')

        
    except:
       print("\nSomething went wrong with the function DCC_Structure.execute.main().")


if __name__ == "__main__":
    main()

