#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:19:08 2024

Last edited on: Nov 12 13:05 2024

Author: Afonso Barroso, The University of Manchester
"""


# ----- # ----- #  IMPORTS # ----- # ----- #


import numpy as np
import sys ; sys.path.append('../Voronoi_PCC_Analyser/'); sys.path.append('../dccstructure')
from matgen.base import Vertex3D, Edge3D, Face3D, Poly, CellComplex
from geometry import tetrahedron_volume
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm


# ----- # ----- #  FUNCTIONS # ----- # ----- #


def createNodes(cells0D: np.array,
                A0: np.array,
                B1: np.array,
                measures0D: list[float] = None,
                progress_bar: bool = True) -> list[Vertex3D]:
    """
    Parameters
    ----------
    cells0D : numpy.array
        An array listing the coordinates of points in 3D space that make up the nodes of a cell complex.
    A0 : numpy.array
        A 3-column representation of the sparse matrix that is the 0D adjacency matrix.
    B1 : numpy.array
        A 3-column representation of the sparse matrix that is the edges-to-nodes incidence matrix.
    measures0D : list
        The measures of the nodes in the cell complex. Default is 1 for every node.

    Returns
    -------
    list
        A list of instances of Vertex3D.
    """
    
    if measures0D is None:
        measures0D = [1] * len(cells0D)
    
    # For the 0-cells, or nodes, we redefine each one as an instance of Vertex3D, which inherits from Vertex, LowerOrderCell and Cell
    
    Nodes = [Vertex3D(id = i+1, x = cells0D[i,0], y = cells0D[i,1], z = cells0D[i,2]) for i in range(len(cells0D))]
    
    if progress_bar == True:
        progress = tqdm(desc = 'Creating nodes in the cell complex', total = len(Nodes), miniters = 1000)
        
    for n in Nodes:
        
        # from Cell
        
        n.set_measure(measures0D[n.id-1])
        
        neighbors = A0[:,1][np.argwhere(A0[:,0] == n.id).tolist()]
        number = len(neighbors)
        neighbors = neighbors.reshape((1,number))[0] # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        neighbors = neighbors.tolist()
        n.add_neighbors(neighbors)
        
        # from LowerOrderCell
        
        incident = np.argwhere(B1[:,1] == n.id)
        number = len(incident)
        incident = incident.reshape((1,number))[0].tolist() # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        incident = B1[incident][:,0] * B1[incident][:,2]
        for e in range(number):
            n.add_incident_cell(incident[e])
        
        if progress_bar == True:
            progress.update()
            
    return Nodes


def createEdges(nodes: list[Vertex3D],
                cells1D: np.array,
                A1: np.array,
                B2: np.array,
                progress_bar: bool = True) -> list[Edge3D]:
    """
    Parameters
    ----------
    nodes : list
        A list of instances of Vertex3D such as is obtained from the function createNodes().
    cells1D : numpy.array
        An array whose rows list the indices of nodes which make up one edge.
    A1 : numpy.array
        A 3-column representation of the sparse matrix that is the 1D adjacency matrix.
    B2 : numpy.array
        A 3-column representation of the sparse matrix that is the faces-to-edges incidence matrix.

    Returns
    -------
    list
        A list of instances of Edge3D.
    """
    
    # For the 1-cells, or edges, we redefine each one as an instance of Edge3D, which inherits from Edge, TripleJunction, LowerOrderCell and Cell

    Edges = [Edge3D(id = i+1, v_ids = cells1D.astype(int)[i].tolist()) for i in range(len(cells1D.astype(int)))]
    
    if progress_bar == True:
        progress = tqdm(desc = 'Creating edges in the cell complex', total = len(Edges), miniters = 1000)
        
    for e in Edges:
        
        # from Cell
        
        node1 = np.array(nodes[e.v_ids[0] - 1].coord)
        node2 = np.array(nodes[e.v_ids[1] - 1].coord)
        e.set_measure(np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2 + (node2[2] - node1[2]) ** 2))
        e.unit_vector = tuple(((node2 - node1) / e.measure).tolist())
        
        neighbors = A1[:,1][np.argwhere(A1[:,0] == e.id).tolist()]
        number = len(neighbors)
        neighbors = neighbors.reshape((1,number))[0] # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        neighbors = neighbors.tolist()
        e.add_neighbors(neighbors)
        
        # from LowerOrderCell
        
        incident = np.argwhere(B2[:,1] == e.id)
        number = len(incident)
        incident = incident.reshape((1,number))[0].tolist() # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        incident = B2[incident][:,0] * B2[incident][:,2]
        for f in range(number):
            e.add_incident_cell(incident[f])
            
        if progress_bar == True:
            progress.update()
            
    return Edges


def createFaces(nodes: list[Vertex3D],
                cells2D: np.array,
                measures2D: list[float],
                normals: np.array,
                A2: np.array,
                B2: np.array,
                B3: np.array,
                progress_bar: bool = True) -> list[Face3D]:
    """
    Parameters
    ----------
    nodes : list
        A list of instances of Vertex3D such as is obtained from the function createNodes().
    cells2D : np.array
        An array whose rows list the indices of nodes which make up one face.
    measures2D : list
        A list of the areas of 2-cells, in order of their index in the cell complex.
    normals : np.array
        A 2D array of the normals of each 2-cell, in order of their index in the cell complex.
    A2 : np.array
        A 3-column representation of the sparse matrix that is the 2D adjacency matrix.
    B2 : np.array
        A 3-column representation of the sparse matrix that is the faces-to-edges incidence matrix.
    B3 : np.array
        A 3-column representation of the sparse matrix that is the volumes-to-faces incidence matrix.

    Returns
    -------
    list
        A list of instances of Face3D.
    """
    
    # For the 2-cells, or faces, we redefine each one as an instance of Face3D, which inherits from Face, GrainBoundary, LowerOrderCell and Cell
    
    Faces = [Face3D(id = i+1, v_ids = cells2D.astype(int)[i].tolist()) for i in range(len(cells2D.astype(int)))]
    
    if progress_bar == True:
        progress = tqdm(desc = 'Creating faces in the cell complex', total = len(Faces), miniters = 1000)
        
    for f in Faces:
        
        # from Cell
        
        f.set_measure(measures2D[f.id-1])
        
        neighbors = A2[:,1][np.argwhere(A2[:,0] == f.id).tolist()]
        number = len(neighbors)
        neighbors = neighbors.reshape((1,number))[0] # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        neighbors = neighbors.tolist()
        f.add_neighbors(neighbors)
        
        # from LowerOrderCell
        
        incident = np.argwhere(B3[:,1] == f.id)
        number = len(incident)
        incident = incident.reshape((1,number))[0].tolist() # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        incident = B3[incident][:,0] * B3[incident][:,2]
        for v in range(number):
            f.add_incident_cell(incident[v])
            
        # from Face
        
        composing_edges = B2[:,1][np.argwhere(B2[:,0] == f.id)]
        number = len(composing_edges)
        composing_edges = composing_edges.reshape((1,number))[0] # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        f.add_edges(composing_edges.tolist())
        
        a = normals[f.id-1][0]
        b = normals[f.id-1][1]
        c = normals[f.id-1][2]
        d = a * nodes[f.v_ids[0]].coord[0] + b * nodes[f.v_ids[0]].coord[1] + c * nodes[f.v_ids[0]].coord[2]
        f.add_equation(d, a, b, c)
    
        if progress_bar == True:
            progress.update()
            
    return Faces


def createPolyhedra(faces: list[Face3D],
                    cells3D: np.array,
                    measures3D: list[float],
                    A3: np.array,
                    B3: np.array,
                    progress_bar: bool = True) -> list[Poly]:
    """
    Parameters
    ----------
    faces : list
        A list of instances of Face3D such as is obtained from the function createFaces().
    cells3D : np.array
        An array whose rows list the indices of nodes which make up one polyhedron.
    measures3D : list
        The volume (measure) of each polyhedron, in order of their index in the cell complex.
    A3 : np.array
        A 3-column representation of the sparse matrix that is the 3D adjacency matrix.
    B3 : np.array
        A 3-column representation of the sparse matrix that is the volumes-to-faces incidence matrix.

    Returns
    -------
    list
        A list of instances of Poly.
    """
    # For the 3-cells, or polyhedra, we redefine each one as an instance of Poly, which inherits from Grain and Cell
    
    Polyhedra = []
    
    if progress_bar == True:
        progress = tqdm(desc = 'Creating polyhedra in the cell complex', total = len(cells3D.astype(int)), miniters = 1000)
    
    for i in range(len(cells3D.astype(int))):
        
        # from Poly
        
        composing_faces = B3[:,1][np.argwhere(B3[:,0] == i+1)]
        number = len(composing_faces)
        composing_faces = composing_faces.reshape((1,number))[0] # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        
        p = Poly(id = i+1, f_ids = composing_faces.tolist())
        
        p.add_vertices(cells3D[i].tolist())
        
        composing_edges = []
        for f in p.f_ids:
            composing_edges += faces[f-1].e_ids
        composing_edges = list(set(composing_edges)) # to remove repeated values
        composing_edges.sort()
        p.add_edges(composing_edges)
        
        # from Cell
        
        p.set_measure(measures3D[p.id-1])
        
        neighbors = A3[:,1][np.argwhere(A3[:,0] == p.id).tolist()]
        number = len(neighbors)
        neighbors = neighbors.reshape((1,number))[0] # need to add the slice [0] at the end because it returns a 2D array, but we want a 1D array
        neighbors = neighbors.tolist()
        p.add_neighbors(neighbors)
        
        # add to list
        
        Polyhedra.append(p)
    
        if progress_bar == True:
            progress.update()
            
    return Polyhedra


def createAllCells(cells0D: np.array,
                   cells1D: np.array,
                   cells2D: np.array,
                   cells3D: np.array,
                   normals: np.array,
                   A0: np.array,
                   A1: np.array,
                   A2: np.array,
                   A3: np.array,
                   B1: np.array,
                   B2: np.array,
                   B3: np.array,
                   measures2D: list[float],
                   measures3D: list[float],
                   measures0D: list[float] = None,
                   progress_bar: bool = True,
                   perturb_vertices: bool = False,
                   complex_size: list = None) -> tuple:
    """
    Parameters
    ----------
    cells0D : np.array
        An array listing the coordinates of points in 3D space that make up the nodes of a cell complex.
    cells1D : np.array
        An array whose rows list the indices of nodes which make up one edge.
    cells2D : np.array
        An array whose rows list the indices of nodes which make up one face.
    cells3D : np.array
        An array whose rows list the indices of nodes which make up one polyhedron.
    measures2D : list
        A list of the areas of 2-cells, in order of their index in the cell complex.
    measures3D : list
        A list of the volumes of 3-cells, in order of their index in the cell complex.
    normals : np.array
        A 2D array of the normals of each 2-cell, in order of their index in the cell complex.
    A0 : np.array
        A 3-column representation of the sparse matrix that is the 0D adjacency matrix.
    A1 : np.array
        A 3-column representation of the sparse matrix that is the 1D adjacency matrix.
    A2 : np.array
        A 3-column representation of the sparse matrix that is the 2D adjacency matrix.
    A3 : np.array
        A 3-column representation of the sparse matrix that is the 3D adjacency matrix.
    B1 : np.array
        A 3-column representation of the sparse matrix that is the edges-to-nodes incidence matrix.
    B2 : np.array
        A 3-column representation of the sparse matrix that is the faces-to-edges incidence matrix.
    B3 : np.array
        A 3-column representation of the sparse matrix that is the volumes-to-faces incidence matrix.
    measures0D : list
        The measures of the nodes in the cell complex. Default is 1 for every node.

    Returns
    -------
    tuple
        A tuple of the lists of instances of Vertex3D, Edge3D, Face3D and Poly.

    """
    
    
    # We proceed by redefining all p-cells of the complex as instances of the various classes from the module matgen.base
    
    Nodes = createNodes(cells0D = cells0D,
                        A0 = A0,
                        B1 = B1,
                        measures0D = measures0D,
                        progress_bar = progress_bar)
    
    if perturb_vertices == True:
        x_min: float = min(cells0D[:,0])
        x_max: float = max(cells0D[:,0])
        y_min: float = min(cells0D[:,1])
        y_max: float = max(cells0D[:,1])
        z_min: float = min(cells0D[:,2])
        z_max: float = max(cells0D[:,2])
        for v in Nodes:
            coord: np.ndarray = np.array(v.coord)  
            if (coord[0] in [x_min, x_max] or coord[1] in [y_min, y_max] or coord[2] in [z_min, z_max]):
                continue
            else:
                # max_coords = [x_max, y_max, z_max]
                # mean: list = [(max_coords[k] / complex_size[k]) / 3.9 for k in range(3)]
                mean: list = [0,0,0]
                # std: list = [k*0.5 for k in mean]
                std: list = [0.1, 0.1, 0.1]
                perturbation: np.ndarary = np.random.normal(mean, std)
                new_coord: np.ndarray = coord + perturbation
                v.x, v.y, v.z = tuple(new_coord)
                
            
    Edges = createEdges(nodes = Nodes,
                        cells1D = cells1D,
                        A1 = A1,
                        B2 = B2,
                        progress_bar = progress_bar)
                
    Faces = createFaces(nodes = Nodes,
                        cells2D = cells2D,
                        measures2D = measures2D,
                        normals = normals,
                        A2 = A2,
                        B2 = B2,
                        B3 = B3,
                        progress_bar = progress_bar)
        
    Polyhedra = createPolyhedra(faces = Faces,
                                cells3D = cells3D,
                                measures3D = measures3D,
                                A3 = A3,
                                B3 = B3,
                                progress_bar = progress_bar)
    
    return (Nodes, Edges, Faces, Polyhedra)


def createCellComplex(dim: int,
                      nodes: list[Vertex3D],
                      edges: list[Edge3D],
                      faces: list[Face3D],
                      faces_slip: list[int],
                      polyhedra: list[Poly],
                      ip: int) -> CellComplex:
    """
    Parameters
    ----------
    dim : int
        The dimension of the cell complex.
    nodes : list[Vertex3D]
        A list of instances of Vertex3D.
    edges : list[Edge3D]
        A list of instances of Edge3D.
    faces : list[Face3D]
        A list of instances of Face3D.
    polyhedra : list[Poly]
        A list of instances of Poly.

    Returns
    -------
    CellComplex
        An instance of the class CellComplex describing the cell complex.
    """
        
    dict0D = {nodes[i].id : nodes[i] for i in range(len(nodes))}
    dict1D = {edges[i].id : edges[i] for i in range(len(edges))}
    dict2D = {faces[i].id : faces[i] for i in range(len(faces))}
    dict3D = {polyhedra[i].id : polyhedra[i] for i in range(len(polyhedra))}
    
    C = CellComplex(dim, dict0D, dict1D, dict2D, dict3D)
    
    C.structure = 'FCC simplicial'
    
    for v in C.vertices:
        v.incident_polyhedra_ids = set()
    for p in C.polyhedra:
        for v in C.get_many('v', p.v_ids):
            if p.id not in v.incident_polyhedra_ids:
                v.incident_polyhedra_ids.add(p.id)
    
    for v in C.vertices:
        v.incident_polyhedra_ids = list(v.incident_polyhedra_ids)
        if ip == 2022:
            v.set_measure(round(node_curvature(C, v.id), 4))
            C.ip = 2022
        elif ip == 2023:
            v.set_measure(1)
            C.ip = 2023
        
    for e in C.edges:
        e.incident_polyhedra_ids = set()
    for p in C.polyhedra:
        for e in C.get_many('e', p.e_ids):
            if p.id not in e.incident_polyhedra_ids:
                e.incident_polyhedra_ids.add(p.id)
    
    """
    Slip in FCC is limited to 12 slip systems. To define the slip systems, we use Schmid's and Boas's notation:
       plane normals:   [-1,1,1] : A , [1, 1,1] : B , [-1,-1,1] : C , [ 1,-1,1] : D
       slip direcitons: [ 0,1,1] : 1 , [0,-1,1] : 2 , [ 1, 0,1] : 3 , [-1, 0,1] : 4 , [-1,1,0] : 5 , [1,1,0] : 6
    Then, the 12 slip systems are A2, A3, A6, B2, B4, B5, C1, C3, C5, D1, D4, D6.
    We can build a dictionary that encodes this information.
    """
    
    slip_planes = {'A' : [-1, 1,1],
                   'B' : [ 1, 1,1],
                   'C' : [-1,-1,1],
                   'D' : [ 1,-1,1]}
    
    slip_directions = {'1' : np.array([ 0, 1, 1]),
                       '2' : np.array([ 0,-1, 1]),
                       '3' : np.array([ 1, 0, 1]),
                       '4' : np.array([-1, 0, 1]),
                       '5' : np.array([-1, 1, 0]),
                       '6' : np.array([ 1, 1, 0])}
    
    edges_slip = []

    for f in C.faces:
        if f.degree == 1:
            f.set_external(True)
            for v_id in f.v_ids:
                C.get_one('v', v_id).set_external(True)
            for e_id in f.e_ids:
                C.get_one('e', e_id).set_external(True)
            for p_id in f.incident_ids:
                C.get_one('p', p_id).set_external(True)
        f.support_volume = sum([p.measure for p in C.get_many(3, f.incident_ids)])
        if f.id in faces_slip:
            if f.degree == 1:
                faces_slip.remove(f.id)
            else:
                plane_index, plane = get_label_from_vector(f.normal, slip_planes)
                f.slip_plane = {plane_index : tuple(plane)}
                edges_slip = edges_slip + f.e_ids
        else:
            f.slip_plane = None
    
    edges_slip.sort()
    edges_slip = np.unique(edges_slip)
    
    for e in C.edges:
        if e.id in edges_slip:
            direction_index, direction = get_label_from_vector(e.unit_vector, slip_directions)
            e.slip_direction = {direction_index : tuple(direction)}
        else:
            e.slip_direction = None
            
    C.faces_slip = faces_slip
    
    return C


def scaleAllCells(nodes: list[Vertex3D],
                  edges: list[Edge3D],
                  faces: list[Face3D],
                  polyhedra: list[Poly],
                  scaling: float | list[float]):
    """
    Scales the coordinates of vertices and the measures of all p-cells (lengths, areas and volumes).
    
    Parameters
    ----------
    nodes : list[Vertex3D]
        A list of instances of Vertex3D.
    edges : list[Edge3D]
        A list of instances of Edge3D.
    faces : list[Face3D]
        A list of instances of Face3D.
    polyhedra : list[Poly]
        A list of instances of Poly.
    scaling : float | list[float]
        Describes how the cell complex mesh is scaled in each direction.
    """
    
    nodes2: list[Vertex3D] = nodes.copy()
    edges2: list[Edge3D] = edges.copy()
    faces2: list[Face3D] = faces.copy()
    polyhedra2: list[Poly] = polyhedra.copy()
    
    if type(scaling) is float:
        for v in nodes2:
            v.x *= scaling
            v.y *= scaling
            v.z *= scaling
        for e in edges2:
            e.measure *= scaling
        for f in faces2:
            f.measure *= scaling ** 2
        for p in polyhedra2:
            p.measure *= scaling ** 3
            
    elif type(scaling) is list:
        for v in nodes2:
            v.x *= scaling[0]
            v.y *= scaling[1]
            v.z *= scaling[2]
        for e in edges2:
            transformed_vector = np.matmul(np.diag(scaling), np.array(e.unit_vector))
            e.measure = np.linalg.norm(transformed_vector)
            e.unit_vector = transformed_vector / e.measure if e.measure != 0 else 0
        for f in faces2:
            transformed_bivector = np.matmul(np.diag(scaling), np.matmul(np.array(f.normal) , np.linalg.inv(np.diag(scaling))))
            f.measure *= np.linalg.norm(transformed_bivector)
            f.normal = transformed_bivector / f.measure if f.measure != 0 else 0
        for p in polyhedra2:
            p.measure *= scaling[0] * scaling[1] * scaling[2]
            
    return nodes2, edges2, faces2, polyhedra2
            

def vector_angle(vector1: np.ndarray,
                 vector2: np.ndarray) -> float:
    """
    Returns
    -------
    The angle between two vetors in Cartesian coordinates.
    """
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    vector1 *= 1 / norm1 if norm1 != 0 else 1
    vector2 *= 1 / norm2 if norm2 != 0 else 1
    return np.arccos(np.dot(vector1,vector2))


def solid_angle(compl: CellComplex,
                vertex_id: int,
                polyhedron_id: int) -> float:
    """
    Computes the solid angle at a vertex into one of its incident polyhedra in a cell complex.
    
    Parameters
    ----------
    compl : CellComplex
        The cell complex where the solid angle is evaluated.
    vertex_id : int
        The index in the cell complex of the vertex.
    top_cell_id : int
        The index in the cell complex of the polyhedron.

    Returns
    -------
    float
        The solid angle subtended by the polyhedron at the vertex.
        
    Notes
    -----
    Based on https://en.wikipedia.org/wiki/Solid_angle#Solid_angles_for_common_objects (the one from L'Huilier's theorem) (Accessed 01 Apr 2024).
    """
        
    polyhedron = compl.get_one('p', polyhedron_id)
    
    v = compl.get_one('v', vertex_id)
    other_v = polyhedron.v_ids.copy()
    other_v.remove(vertex_id)
    other_v = compl.get_many('v', other_v)
    
    v1 = np.array(other_v[0].coord) - np.array(v.coord)
    v2 = np.array(other_v[1].coord) - np.array(v.coord)
    v3 = np.array(other_v[2].coord) - np.array(v.coord)

    angle1 = vector_angle(v2, v3)
    angle2 = vector_angle(v1, v3)
    angle3 = vector_angle(v1, v2)
    
    s = sum([angle1, angle2, angle3]) / 2
    s = 4 * np.arctan(np.sqrt(np.tan(s/2) * np.tan((s - angle1)/2) * np.tan((s - angle2)/2) * np.tan((s - angle3)/2)))
    
    return s


def node_curvature(compl: CellComplex,
                   vertex_id: int) -> float:
    """
    Parameters
    ----------
    compl : CellComplex
        The cell complex where the solid angle is evaluated.
    vertex_id : int
        The index in the cell complex of the vertex.

    Returns
    -------
    float
        The node weight as given in Definition 3.13 of Berbatov, K., et al. (2022). "Diffusion in multi-dimensional solids using Forman's combinatorial differential forms". App Math Mod 110, pp. 172-192.
    """
    
    denominator = 0
    
    v = compl.get_one('v', vertex_id)
    for p in v.incident_polyhedra_ids:
        denominator += solid_angle(compl, vertex_id, p)
        
    return (4 * np.pi) / denominator


def create_ax_afonso(dim: int = 3,
                     figsize: tuple = (10, 10),
                     xyz_lim: list[float] = [1., 1., 1.],
                     elevation: float = 15,
                     azimuthal: float = 11) -> Axes:        
    """
    Create Axes object for plotting.
    
    Parameters
    ----------
    dim : int
        The dimension of the plot.
    figsize : tuple
        The edge lengths of the figure where the plot will be shown.
    xyz_lim : list
        The upper limits of the x, y and z axes.
        
    Returns
    -------
        Axes objects.
        
    Notes
    -----
    Based on the function matgen.base._create_ax by Oleg Bushuev.
    """
    if dim == 2:
        projection = None
    elif dim == 3:
        projection = '3d'
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111,
                         projection = projection)
    ax.set_xlim([0, xyz_lim[0]])
    ax.set_ylim([0, xyz_lim[1]])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if dim == 3:
        ax.set_zlim([0, xyz_lim[2]])
        ax.set_zlabel('z')
    ax.set_axis_off()
    ax.view_init(elev = elevation, azim = azimuthal)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    return ax


def get_label_from_vector(normal: tuple[float], dictionary: dict) -> tuple:
        
    options = np.array(list(dictionary.values()))
    label = np.tensordot(options, np.array(normal), axes=1)
    label = np.argwhere(abs(label) == max(abs(label)))[:,0][0]
    answer = options[label]
    label = list(dictionary.keys())[label]
        
    return label, answer


def support_volume(face: Face3D, lst_poly: list[Poly], get) -> float:
        
    if len(face.v_ids) == 3:
        supp_vol = 0
        for p in lst_poly:
            barycentre = np.mean(np.array([v.coord for v in get(0, p.v_ids)]), axis = 0)
            points = np.vstack((np.array([v.coord for v in get(0, face.v_ids)]), barycentre))
            supp_vol += tetrahedron_volume(points)
    else:
        supp_vol = None
        
    return supp_vol
    


# ----- # ----- #  CODE # ----- # ----- #


if __name__ == '__main__':
    
    sys.path.append('../')
    from dccstructure.iofiles import import_complex_data
    from pathlib import Path
    import time
    
    t0 = time.time()
    
    data_folder = Path('../Built_Complexes/FCC_1x1x1')
    
    complex_size = [int(str(data_folder)[-5]), int(str(data_folder)[-3]), int(str(data_folder)[-1])] # the number of computational unit cells in each 3D dimension
    
    nodes, edges, faces, faces_slip, faces_areas, faces_normals, volumes, volumes_vols, nr_cells, A0, A1, A2, A3, B1, B2, B3 = import_complex_data(data_folder)
    
    edges += 1 ; faces += 1 ; volumes += 1  ;  edges = edges.astype(int)
    if faces_slip: faces_slip = [x + 1 for x in faces_slip]
    else: faces_slip = list(range(1, len(faces)+1))
    A0[:,0:2] += 1 ; A1[:,0:2] += 1 ; A2[:,0:2] += 1 ; A3[:,0:2] += 1 ; B1[:,0:2] += 1 ; B2[:,0:2] += 1 ; B3[:,0:2] += 1
    
    perturb_vertices: bool = False
    if perturb_vertices:
        faces_slip = list(range(1, len(faces)+1))
    
    Nodes, Edges, Faces, Polyhedra = createAllCells((nodes + 0.5),
                                                    edges,
                                                    faces,
                                                    volumes,
                                                    faces_normals,
                                                    A0, A1, A2, A3,
                                                    B1, B2, B3,
                                                    faces_areas,
                                                    volumes_vols,
                                                    perturb_vertices = perturb_vertices,
                                                    complex_size = complex_size)
    
    del A0, A1, A2, A3, B1, B2, B3
    # del nodes, edges, faces, faces_areas, faces_normals, volumes, volumes_vols
    
    print('\nScaling all cells...')
    Nodes, Edges, Faces, Polyhedra = scaleAllCells(Nodes, Edges, Faces, Polyhedra, scaling = 1e-6/complex_size[0])
    
    print('Creating the cell complex...')
    cc = createCellComplex(dim = 3,
                           nodes = Nodes,
                           edges = Edges,
                           faces = Faces,
                           faces_slip = faces_slip,
                           polyhedra = Polyhedra,
                           ip = 2022)
    
    xyz_lim = list(cc.vertices[-1].coord)
    ax = create_ax_afonso(xyz_lim = xyz_lim)
    
    cc.plot_polyhedra(ax = ax, labels=True, alpha=0.1, color='purple', edgecolor='black', linewidth=2)
    # cc.plot_faces(f_ids = [18,60,62,88,6,36,31,72], ax=ax, color='grey', edgecolor='black')
    plt.savefig('FCC111.pdf', dpi=500)
    
    # fig = plt.figure(figsize = (11,11))
    # ax = fig.add_subplot(111,
    #                       projection = '3d')
    # ax.set_xlim([0, xyz_lim[0]])
    # ax.set_ylim([0, xyz_lim[1]])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlim([0, xyz_lim[2]])
    # ax.set_zlabel('z')
    # ax.set_axis_off()
    # ax.view_init(elev = 15, azim = 11)

    # f_list = cc.get_many(2, cc.faces_slip) # [55,14,32,26,85,82,92,46]
    # ax.plot_trisurf([v.coord[0] for v in cc.vertices],
    #                 [v.coord[1] for v in cc.vertices],
    #                 [v.coord[2] for v in cc.vertices],
    #                 triangles = [[i-1 for i in f.v_ids] for f in f_list],
    #                 shade = False,
    #                 color = 'purple',
    #                 edgecolor = 'black',
    #                 linewidth = 2,
    #                 alpha = 0.5)
        
    # # for e in cc.get_many(1, [1,2,4,7,11,13,16,27,29,31,33,37,40,50,66,68,74,75,76,78,81,83,88,89,90]):
    # for e in cc.edges:
    #     v1, v2 = cc.get_many('v', e.v_ids)
    #     x_space = (v1.coord[0], v2.coord[0])
    #     y_space = (v1.coord[1], v2.coord[1])
    #     z_space = (v1.coord[2], v2.coord[2])
    #     ax.plot3D(x_space, y_space, z_space, c='black', linestyle = '--', zorder=1)
    
    # plt.savefig('FCC_2x2x2_slipplanes.pdf', dpi=500)
    # plt.show()
    
    print("Time elapsed: ", time.time() - t0, " s.") ; del t0



