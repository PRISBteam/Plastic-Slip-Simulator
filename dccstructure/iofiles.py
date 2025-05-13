# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2022

Last edited on: Mar 28 15:40 2024

Author: Afonso Barroso, The University of Manchester

This module is part of the DCC_Structure package. In here you will find a useful function for writing several variables onto .txt
files in one go.

"""


# ----- #----- #  IMPORTS # ----- # ----- #


import numpy as np
import os
from pathlib import Path

# ----- # ----- # FUNCTIONS # ------ # ----- #


def write_to_file(*args, out_name = r'd_output', new_folder = True):
    """
    Parameters
    ----------
    file : .txt file
        A file on which we want to write the output of the code.
    *args : np.array AND str
        The input arguments should be an alternating listing of np.arrays and strings, the first specifying an internal structure
        of the complex, namely nodes, edges, faces, and volumes, and the second simply stating the name of the preceding variable,
        e.g. *args = (nodes, 'nodes', edges, 'edges', faces, 'faces', volumes, 'volumes').

    Returns
    -------
    Several .txt files with the input *args written on them.
    """
        
    if new_folder:
    
        current_directory = os.getcwd()
            
        final_directory = os.path.join(current_directory, out_name)
        
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
           
        os.chdir(final_directory)
       
    
    for i in range(0, len(args)):
        
        # The point of the *args is that each i%2 == 0 is some variable np.array and each i%2 == 1 is a str, detailing the
        # name of the preceding variable.

        if i %2 == 0:
            
            file_name = args[i+1] + '.txt'
                        
            # if the input variable is a 2D array of floats 
            if type(args[i]) == np.ndarray and type(args[i][0,0]) in [np.float64, np.float32]:
                
                ft = '%1.8f'
                
            # if the input variable is a 2D array of integers
            if type(args[i]) == np.ndarray and type(args[i][0,0]) in [np.int64, np.int32]:
                
                ft = '%i'
            
            # if the input variable is a 1D array of floats 
            if type(args[i]) == np.ndarray and type(args[i][0]) in [np.float64, np.float32]:
                
                ft = '%1.8f'
                
            # if the input variable is a 1D array of integers
            if type(args[i]) == np.ndarray and type(args[i][0]) in [np.int64, np.int32]:
                
                ft = '%i'
                
            # if the input variable is a list of floats
            elif type(args[i]) == list and type(args[i][0]) == float:
                
                ft = '%1.8f'
            
            # if the input variable is a list of integers
            elif type(args[i]) == list and type(args[i][0]) == int:
                
                ft = '%i'

            # if the input variable is a list of arrays
            elif type(args[i]) == list and type(args[i][0]) == np.ndarray:
            
                ft = '%1.8f'
            
            np.savetxt(file_name, args[i], fmt = ft, delimiter = ' ', comments = '# ')




def import_complex_data(data_folder: Path):
    """
    Parameters
    ----------
    data_folder : Path
        The path name of the folder where the data files are located.

    Returns
    -------
    tuple
        Extracts information about the complex from the data files in data_folder.
        Order: nodes, edges, faces, faces_slip, faces_areas, faces_normals, volumes, volumes_vols, nr_cells, A0, A1, A2, A3, B1, B2, B3
    """
    
    # Import cell complex data files
    
    data = []
    
    if (data_folder / 'nodes.txt'):
        with open(data_folder / 'nodes.txt') as file:
            nodes = np.genfromtxt(file, delimiter = ' ')
        data.append(nodes)
        
    if (data_folder / 'edges.txt'):
        with open(data_folder / 'edges.txt') as file:
            edges = np.genfromtxt(file, delimiter = ' ')
        data.append(edges)
        
    if (data_folder / 'faces.txt'):
        with open(data_folder / 'faces.txt') as file:
            faces = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(faces)
        
    if (data_folder / 'faces_slip.txt'):
        with open(data_folder / 'faces_slip.txt') as file:
            faces_slip = list(np.genfromtxt(file, delimiter = ' ').astype(int))
        data.append(faces_slip)
        
    if (data_folder / 'faces_areas.txt'):
        with open(data_folder / 'faces_areas.txt') as file:
            faces_areas = list(np.genfromtxt(file, delimiter = ' '))
        data.append(faces_areas)

    if (data_folder / 'faces_normals.txt'):
        with open(data_folder / 'faces_normals.txt') as file:
            faces_normals = np.genfromtxt(file, delimiter = ' ')
        data.append(faces_normals)
        
    if (data_folder / 'volumes.txt').is_file():
        with open(data_folder / 'volumes.txt') as file:
            volumes = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(volumes)
        
    if (data_folder / 'volumes_vols.txt').is_file():
        with open(data_folder / 'volumes_vols.txt') as file:
            volumes_vols = list(np.genfromtxt(file, delimiter = ' '))
        data.append(volumes_vols)
        
    if (data_folder / 'nr_cells.txt'):   
        with open(data_folder / 'nr_cells.txt') as file:
            nr_cells = list(np.genfromtxt(file, delimiter = ' ').astype(int))
        data.append(nr_cells)
        
    if (data_folder / 'A0.txt'):
        with open(data_folder / 'A0.txt') as file:
            A0 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(A0)
        
    if (data_folder / 'A1.txt'):
        with open(data_folder / 'A1.txt') as file:
            A1 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(A1)
        
    if (data_folder / 'A2.txt'):
        with open(data_folder / 'A2.txt') as file:
            A2 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(A2)
        
    if (data_folder / 'A3.txt'):
        with open(data_folder / 'A3.txt') as file:
            A3 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(A3)
        
    if (data_folder / 'B1.txt'):
        with open(data_folder / 'B1.txt') as file:
            B1 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(B1)
        
    if (data_folder / 'B2.txt'):
        with open(data_folder / 'B2.txt') as file:
            B2 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(B2)
        
    if (data_folder / 'B3.txt'):
        with open(data_folder / 'B3.txt') as file:
            B3 = np.genfromtxt(file, delimiter = ' ').astype(int)
        data.append(B3)
        
    return data





