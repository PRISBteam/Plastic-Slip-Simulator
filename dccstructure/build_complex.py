# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:00 2022

Last edited on: 28/06/2022 16:22

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the dccstructure package. In here you will find functions pertinent to the construction of the cell complex.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
import orientations
import matrices
import build
from geometry import unit_normal
from iofiles import write_to_file


# ----- # ----- # FUNCTIONS # ------ # ----- #



def build_complex(structure, size, dimension=3, basis_vectors=np.array([[1,0,0],[0,1,0],[0,0,1]]), degree_yes=False):
    """
    Parameters
    ----------
    structure : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.
    dimension : TYPE, optional
        DESCRIPTION. The default is 3.
    basis_vectors : TYPE, optional
        DESCRIPTION. The default is np.array([[1,0,0],[0,1,0],[0,0,1]]).

    Returns
    -------
    None.
    """
    
    first_u_cell = np.array([[0,0,0]]) + np.sum(basis_vectors, axis=0) / 2


    if structure == 'simple cubic':
        
        #------- NODES in SC

        first_node = first_u_cell - np.sum(basis_vectors, axis=0) / 2
        
        nodes = build.create_nodes(structure = structure,
                                        origin = first_node,
                                        lattice = basis_vectors,
                                        size = size,
                                        dim = dimension)
        
        #------- EDGES in SC
        
        edges = build.find_neighbours(nodes, basis_vectors, structure = structure, dim=dimension)
                    
        if degree_yes:
            node_degrees = matrices.degree_distribution(edges, nodes)
        
        #------- FACES in SC
        
        faces = build.create_faces(edges, structure = structure)
        
        faces_as_edges = orientations.faces_to_edges(faces, edges, edges_per_face = 4)
        
        #------- VOLUMES in SC
        
        volumes = build.create_volumes(basis_vectors, structure, cells_0D = nodes)
        
        volumes_as_faces = orientations.volumes_to_faces(volumes, faces, faces_per_volume = 6)
                    
        volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                                        cells_2D = faces,
                                                                                        cells_1D = edges,
                                                                                        cells_0D = nodes,
                                                                                        v2f = volumes_as_faces,
                                                                                        f2e = faces_as_edges)
    
        #------- MATRICES in SC
                
        adjacency_matrices = matrices.combinatorial_form(structure = structure,
                                                                degree = 0,
                                                                cells_3D = volumes,
                                                                cells_2D = faces,
                                                                cells_1D = edges,
                                                                cells_0D = nodes,
                                                                v2f = volumes_as_faces,
                                                                f2e = faces_as_edges)
        
        incidence_matrices = matrices.combinatorial_form(structure = structure,
                                                                degree = 1,
                                                                cells_3D = volumes,
                                                                cells_2D = faces,
                                                                cells_1D = edges,
                                                                cells_0D = nodes,
                                                                v2f = volumes_as_faces,
                                                                f2e = faces_as_edges)
        
        

    elif structure == 'bcc':
        
        #------- NODES in BCC

        first_node = first_u_cell - np.sum(basis_vectors, axis=0) / 2
        
        nodes, (nodes_sc, nodes_bcc, nodes_virtual) = build.create_nodes(structure = structure,
                                                                                origin = first_node,
                                                                                lattice = basis_vectors,
                                                                                size = size,
                                                                                dim = dimension,
                                                                                axis = 2)
        
        #------- EDGES in BCC
        
        edges, (edges_sc, edges_bcc, edges_virtual) = build.find_neighbours(nodes, basis_vectors,
                                                                                structure = structure,
                                                                                dim=dimension,
                                                                                special_0D = (nodes[nodes_sc],
                                                                                                nodes[nodes_bcc],
                                                                                                nodes[nodes_virtual]))
                        
        if degree_yes:
            node_degrees = matrices.degree_distribution(edges, nodes)
        
        #------- FACES in BCC
        
        faces, faces_sc = build.create_faces(edges, structure = structure, cells_0D = nodes)
        
        faces_as_edges = orientations.faces_to_edges(faces, edges)
                    
        #------- VOLUMES in BCC
            
        volumes = build.create_volumes(basis_vectors, structure, cells_2D = faces)
        
        volumes_as_faces = orientations.volumes_to_faces(volumes, faces)
                    
        volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                                        cells_2D = faces,
                                                                                        cells_1D = edges,
                                                                                        cells_0D = nodes,
                                                                                        v2f = volumes_as_faces,
                                                                                        f2e = faces_as_edges)
    
        #------- MATRICES in BCC
                    
        adjacency_matrices = matrices.combinatorial_form(structure = structure,
                                                                degree = 0,
                                                                cells_3D = volumes,
                                                                cells_2D = faces,
                                                                cells_1D = edges,
                                                                cells_0D = nodes,
                                                                v2f = volumes_as_faces,
                                                                f2e = faces_as_edges)
        
        incidence_matrices = matrices.combinatorial_form(structure = structure,
                                                                degree = 1,
                                                                cells_3D = volumes,
                                                                cells_2D = faces,
                                                                cells_1D = edges,
                                                                cells_0D = nodes,
                                                                v2f = volumes_as_faces,
                                                                f2e = faces_as_edges)

        
    elif structure == 'fcc':

        #------- NODES in FCC
                
        first_node = first_u_cell - (basis_vectors[0] + basis_vectors[1] + basis_vectors[2]) / 2

        nodes, (nodes_sc, nodes_bcc, nodes_fcc) = build.create_nodes(structure = 'bcc',
                                                                            origin = first_node,
                                                                            lattice = basis_vectors,
                                                                            size = size,
                                                                            dim = dimension)
        #------- EDGES in FCC

        edges, edges_sc, edges_bcc_fcc, edges_fcc2, edges_fcc_sc = build.find_neighbours(nodes,
                                                                                                lattice = basis_vectors,
                                                                                                structure = structure,
                                                                                                dim = dimension,
                                                                                                special_0D = (nodes_sc, nodes_bcc, nodes_fcc))
                        
        if degree_yes:
            node_degrees = matrices.degree_distribution(edges, nodes)
    
        
        #------- FACES in FCC
        
        faces, faces_slip = build.create_faces(edges,
                                                    structure = structure,
                                                    cells_0D = (nodes, nodes_sc, nodes_bcc, nodes_fcc))
    
        faces_as_edges = orientations.faces_to_edges(faces, edges)
                    
        #------- VOLUMES in FCC
        
        volumes = build.create_volumes(basis_vectors, structure, cells_2D = faces)
        
        volumes_as_faces = orientations.volumes_to_faces(volumes, faces)
                    
        volumes_as_faces, faces_as_edges = orientations.find_relative_orientations(cells_3D = volumes,
                                                                                        cells_2D = faces,
                                                                                        cells_1D = edges,
                                                                                        cells_0D = nodes,
                                                                                        v2f = volumes_as_faces,
                                                                                        f2e = faces_as_edges)
    
        #------- MATRICES in FCC
                
        adjacency_matrices = matrices.combinatorial_form(structure = structure,
                                                                degree = 0,
                                                                cells_3D = volumes,
                                                                cells_2D = faces,
                                                                cells_1D = edges,
                                                                cells_0D = nodes,
                                                                v2f = volumes_as_faces,
                                                                f2e = faces_as_edges)
        
        incidence_matrices = matrices.combinatorial_form(structure = structure,
                                                                degree = 1,
                                                                cells_3D = volumes,
                                                                cells_2D = faces,
                                                                cells_1D = edges,
                                                                cells_0D = nodes,
                                                                v2f = volumes_as_faces,
                                                                f2e = faces_as_edges)

        
    if degree_yes:
        return nodes, edges, faces, volumes, adjacency_matrices, incidence_matrices, node_degrees
    else:
        return nodes, edges, faces, volumes, adjacency_matrices, incidence_matrices
    
    
    
nodes, edges, faces, volumes, adjacency_matrices, incidence_matrices = build_complex('fcc', [1,1,1])

normals = np.array([unit_normal(i) for i in nodes[faces]])

write_to_file(normals, 'normals')





