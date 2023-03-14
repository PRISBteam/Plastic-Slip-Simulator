# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2022

Last edited on: Mar 14 11:20 2023

Author: Afonso Barroso, 9986055, The University of Manchester

This module is part of the DCC_Structure package. In here you will find a useful function for writing several variables onto .txt
files in one go.

"""


# ----- #----- #  IMPORTS # ----- # ----- #


import numpy as np
import os


# ----- # ----- # FUNCTIONS # ------ # ----- #


def write_to_file(*args, new_folder = True, results=False):
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
    
    # This function will create one .txt file with the *args cleanly presented on it, easy for a human to read (if passed results=True),
    # as well as other files, with the *args more messily printed on them, but such that they are easier to read with another
    # Python script, for example.
    
    if new_folder:
    
        current_directory = os.getcwd()
            
        final_directory = os.path.join(current_directory, r'dccstructure_output')
        
        if not os.path.exists(final_directory):
            
           os.makedirs(final_directory)
           
        os.chdir(final_directory)
       
    # First file.
    
    if results == True:
    
        with open('results.txt', 'w') as f1:
                
            for i in range(0, len(args)):
                
                # The point of the *args is that each i%2 == 0 is some variable np.array and each i%2 == 1 is a str, detailing the
                # name of the preceding variable.
    
                if i %2 == 0:
                    
                    for row in args[i]:
                        
                        if np.all(row == args[i][0]): # first line
                            
                            f1.write(str(args[i + 1]) + ' = np.array([' + str(row) + ',\n')
                                                                        
                        elif np.all(row == args[i][-1]): # last line
                            
                            f1.write(' ' * (len(args[i + 1]) + 13) + str(row) + '])\n\n')
                            
                        else:
                            
                            f1.write(' ' * (len(args[i + 1]) + 13) + str(row) + ',\n')
                            
                else:
                    pass
            
    # Other files.
    
    for i in range(0, len(args)):
        
        # The point of the *args is that each i%2 == 0 is some variable np.array and each i%2 == 1 is a str, detailing the
        # name of the preceding variable.

        if i %2 == 0:
            
            file_name = args[i+1] + '.txt'
            
            ft = '%i'
                
            if type(args[i]) == np.ndarray and len(np.shape(args[i])) == 2 and type(args[i][0,0]) == np.float64:
                
                ft = '%1.8f'
                
            if type(args[i]) == np.ndarray and len(np.shape(args[i])) == 1 and type(args[i][0,0]) == np.float64:
                
                ft = '%1.8f'
            
            if type(args[i]) == list and type(args[i][0]) == float:
                
                ft = '%1.8f'
            
            np.savetxt(file_name, args[i], fmt = ft, delimiter = ' ', comments = '# ')




def import_complex_data(data_folder):
    """
    Parameters
    ----------
    data_folder : str
        The path name of the folder where the data files are located.

    Returns
    -------
    np.array
        Extracts information about the complex from the data files in data_folder.
    """

    # Import cell complex data files
            
    directory = os.getcwd() # str type
            
    os.chdir(data_folder)
    
    if os.path.isfile('./A0.txt') or os.path.isfile('.\\A0.txt'):
            
        with open('A0.txt') as file:
            
            A0 = np.genfromtxt(file, delimiter = ' ')
    
    if os.path.isfile('./A1.txt') or os.path.isfile('.\\A1.txt'):

        with open('A1.txt') as file:
    
            A1 = np.genfromtxt(file, delimiter = ' ')
    
    if os.path.isfile('./A2.txt') or os.path.isfile('.\\A2.txt'):

        with open('A2.txt') as file:
            
            A2 = np.genfromtxt(file, delimiter = ' ')
        
    if os.path.isfile('./A3.txt') or os.path.isfile('.\\A3.txt'):
        
        with open('A3.txt') as file:
            
            A3 = np.genfromtxt(file, delimiter = ' ')
        
    if os.path.isfile('./B1.txt') or os.path.isfile('.\\B1.txt'):
            
        with open('B1.txt') as file:
            
            B1 = np.genfromtxt(file, delimiter = ' ')
            
    if os.path.isfile('./B2.txt') or os.path.isfile('.\\B2.txt'):
            
        with open('B2.txt') as file:
            
            B2 = np.genfromtxt(file, delimiter = ' ')
            
    if os.path.isfile('./B3.txt') or os.path.isfile('.\\B3.txt'):
            
        with open('B3.txt') as file:
            
            B3 = np.genfromtxt(file, delimiter = ' ')
            
    if os.path.isfile('./faces_areas.txt') or os.path.isfile('.\\faces_areas.txt'):
            
        with open('faces_areas.txt') as file:
            
            faces_areas = np.genfromtxt(file, delimiter = ' ')
            
    if os.path.isfile('./faces_slip.txt') or os.path.isfile('.\\faces_slip.txt'):
            
        with open('faces_slip.txt') as file:
            
            faces_slip = np.genfromtxt(file, delimiter = ' ')
            
    if os.path.isfile('./node_degrees.txt') or os.path.isfile('.\\node_degrees.txt'):
            
        with open('node_degrees.txt') as file:
            
            node_degrees = np.genfromtxt(file, delimiter = ' ')
            
    if os.path.isfile('./normals.txt') or os.path.isfile('.\\normals.txt'):
            
        with open('normals.txt') as file:
            
            faces_normals = np.genfromtxt(file, delimiter = ' ')
            
    # if os.path.isfile('./number_of_cells.txt') or os.path.isfile('.\\number_of_cells.txt'):
            
    #     with open('number_of_cells.txt') as file:
            
    #         nr_cells = np.genfromtxt(file, delimiter = ' ')
              
    os.chdir(directory)
    
    return A0.astype(int), A1.astype(int), A2.astype(int), A3.astype(int), B1.astype(int), B2.astype(int), B3.astype(int), faces_areas, list(faces_slip.astype(int)), list(node_degrees.astype(int)), faces_normals #, list(nr_cells.astype(int))





