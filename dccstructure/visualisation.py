# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 12:43 2022

Last edited on: 07/06/2022 14:50

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the DCC_Structure package. In here you will find a function that creates a visualisation of the a complex.

"""


# ----- #----- #  IMPORTS # ----- # ----- #


import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations # to avoid nested for loops
from dccstructure.iofiles import write_to_file


# ----- # ----- # FUNCTIONS # ------ # ----- #


def graph_complex(structure, cells_0D, cells_1D=None, cells_2D=None, azimuth=8, text=True):
    """
    Parameters
    ----------
    structure : str, optional
        A descriptor of the basic structure of the lattice.
    cells_0D : np array OR tuple of np arrays
        The spatial positions of points in 3D space.
    cells_1D : numpy array (M x 2) OR tuple of numpy arrays
        An array of edges connecting the points in 'cells_0D'. The default is None.
    cells_2D : np array
        An array whose rows list the indices of nodes which make up one face.
    azimuth: int
        The azimuthal angle for visualisation. The defualt is 8.
    text : bool
        Whether or not to write the node indices on the graph/image. The default is True.

    Returns
    -------
    Plots the complex with or without edges/faces between the nodes and saves it as a .png file.

    """
    
    def alpha_param(full_array, part_array):
        """
        Parameters
        ----------
        full_array : np array
            The x values of the nodes.
        part_array : np array
            The x values of the nodes of the edge/face we are considering.

        Returns
        -------
        a : float
            The appropriate alpha value for the edge/face we are considering.
        """
        
        max_value = np.max(full_array)
        
        a = np.average(part_array) / max_value
        
        if a == 0:
            
            a = 0.1
            
        elif 0 < a <= 0.9:
            
            a += 0.1
            
        elif a > 1:
                
            a = 1
        
        return a
            

    fig_name = STRUC + '_' + str(np.prod(SIZE))
    
    f_size = (10, 10)
    
    plt.figure(figsize = f_size)
    
    ax = plt.axes(projection = '3d')
        
    #---- SC ----#
    
    if structure == 'simple cubic':
                
        # Plot nodes
        
        ax.view_init(elev = 15, azim = azimuth)
                    
        ax.scatter3D(cells_0D[:,0], cells_0D[:,1], cells_0D[:,2], s = 70, c = 'black')
        
        if cells_1D != None:
            
            fig_name = fig_name + '_edges'

            for edge in cells_1D:
                
                edge = cells_0D[edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], 'green', alpha = alpha_param(edges))
        
    #---- BCC ----#
    
    elif structure == 'bcc':

        """ Required: cells_0D = (nodes, nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_virtual]);
                      cells_1D = (edges, edges[edges_sc], edges[edges_bcc], edges[edges_virtual]) """

        # Plot nodes
                
        ax.view_init(elev = 11, azim = azimuth)
        
        ax.scatter3D(cells_0D[1][:,0], cells_0D[1][:,1], cells_0D[1][:,2], s = 70, c = 'black')     # regular simple cubic nodes
        ax.scatter3D(cells_0D[2][:,0], cells_0D[2][:,1], cells_0D[2][:,2], s = 70, c = 'cyan')      # proper bcc nodes
        ax.scatter3D(cells_0D[3][:,0], cells_0D[3][:,1], cells_0D[3][:,2], s = 70, c = 'green')     # virtual fcc nodes

        # Plot edges
        
        if cells_1D != None:
            
            fig_name = fig_name + '_edges'
            
            for edge in cells_1D[1]: # SC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c='black', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[2]: # BCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c='blue', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[3]: # Virtual FCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], 'g--', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
            
        # Plot faces
        
        try:
            
            if cells_2D.any() != None:
            
                tri = np.empty((0,3))

                fig_name = fig_name + '_faces'
                
                # chosen_faces = [11,  9, 10, 39, 31, 33, 34, 43]
                # chosen_nodes = np.unique(cells_2D[chosen_faces])
                
                chosen_nodes = np.array([0,1,2,3,4,7])
                
                x = cells_0D[0][chosen_nodes][:,0]
                y = cells_0D[0][chosen_nodes][:,1]
                z = cells_0D[0][chosen_nodes][:,2]
                
                points = np.array(tuple(zip(x,y,z)))
                
                triangles = combinations(list(range(0,np.size(x))), 3)
                
                for i in triangles:
                    
                    possible_face = find_equal_rows(cells_0D[0], points[list(i)])[:,0]
                    
                    possible_face = list(find_equal_rows(cells_2D, np.array([np.sort(possible_face)]))[:,0])
                    
                    if possible_face != []: # If the face exists
                        
                        a = np.array([i[0], i[1], i[2]])
                        
                        tri = np.vstack((tri, a))
                    
                    else:
                        
                        pass
                                    
                ax.plot_trisurf(x, y, z, triangles = tri.astype(int), color='pink', edgecolor='purple', alpha=0.5, linewidth=3)
                                        
        except:
            pass
        
    #---- FCC ----#
    
    elif structure == 'fcc':
        
        """ Required: cells_0D = (nodes, nodes[nodes_sc], nodes[nodes_bcc], nodes[nodes_fcc]);
                      cells_1D = (edges, edges[edges_sc], edges[edges_bcc_fcc], edges[edges_fcc2], edges[edges_fcc_sc]) """

        # Plot nodes
                
        ax.view_init(elev = 11, azim = azimuth)
        
        ax.scatter3D(cells_0D[1][:,0], cells_0D[1][:,1], cells_0D[1][:,2], s = 70, c = 'black')     # regular simple cubic nodes
        ax.scatter3D(cells_0D[2][:,0], cells_0D[2][:,1], cells_0D[2][:,2], s = 70, c = 'cyan')      # proper bcc nodes
        ax.scatter3D(cells_0D[3][:,0], cells_0D[3][:,1], cells_0D[3][:,2], s = 70, c = 'green')     # virtual fcc nodes

        # Plot edges
        
        if cells_1D != None:
            
            fig_name = fig_name + '_edges'
            
            for edge in cells_1D[1]: # SC-SC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'black', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[2]: # BCC-FCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'blue', ls = '--', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[3]: # FCC-FCC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'turquoise', ls = '--', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
                
            for edge in cells_1D[4]: # FCC-SC edges
                
                edge = cells_0D[0][edge]
                
                ax.plot3D(edge[:,0], edge[:,1], edge[:,2], c = 'green', ls = '-.', alpha = alpha_param(cells_0D[0][:,0], edge[:,0]))
            
        # Plot faces
        
        try:
                        
            fig_name = fig_name + '_faces'
            
            tri = np.empty((0,3))
            
            chosen_faces = [33, 58, 56, 61, 21, 10, 17, 37]
            
            points = np.unique(cells_2D[chosen_faces])
            
            points = cells_0D[0][points]
                                    
            for face in chosen_faces:
                
                t = cells_0D[0][cells_2D[face]]
                
                t = find_equal_rows(points, t)[:,0]
                
                tri = np.vstack((tri, t))
                
            # ls = LightSource(azdeg = 250, altdeg = 45) (need lightsource=ls below)
                                
            ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                            triangles = tri.astype(int),
                            color='pink', edgecolor='purple', alpha=0.5, linewidth=3,
                            shade=True)
                                        
        except:
            pass
        
        # # Plot volumes
        
        # chosen_volumes = [5, 24, 23]
        
        # for volume in chosen_volumes:
        
        #     trios = np.sort(np.array([i for i in combinations(volume, 3)])) # These represent possible faces in the selected volume
            
        #     true_trios = find_equal_rows(faces, trios).astype(int) # these are the triplets above that do correspond to faces
            
        #     faces_in_volume = true_trios[:,0] # Gives indices of edges that constitute the face
            
        #     volumes_as_faces = np.vstack((volumes_as_faces, faces_in_volume)).astype(int)


    if azimuth != 8:
        
        fig_name = fig_name + str(azimuth)
    
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_xlim([0, np.max(cells_0D[0][:,0])])
    ax.set_ylim([0, np.max(cells_0D[0][:,1])])
    ax.set_zlim([0, np.max(cells_0D[0][:,2])])
    
    if text == True:
        
        for i in range(0, np.shape(cells_0D[0])[0]):
            
            ax.text(cells_0D[0][i,0] + 0.05, cells_0D[0][i,1] + 0.05, cells_0D[0][i,2] + 0.05,
                    str(i), color='black', weight = 'bold', size = 15,
                    zorder = 100)
            
    plt.axis('off')
    
    fig_name = fig_name + '.png'
    plt.savefig(fig_name, dpi=400)
        
    plt.show()
