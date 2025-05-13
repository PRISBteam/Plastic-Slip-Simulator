#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:02 2024
Last edited on: Feb 12 11:33 2025

Author: Afonso Barroso, 9986055, The University of Manchester

Solves an Ising model of plastic slips in copper by employing a Metropolis-Hastings algorithm which minimises an energy function. This function
is taken from Naimark's work and comprises a self-energy term for the defects, a term for the mechanical energy of the externally applied stress
and a mean-field term for the effect of the ensemble of defects. All terms are calculated as inner products of vector-valued 2-cochains, wherein
the inner product of cochains is paired with the inner product of vectors in R^3. The slip cochain is computed as a local microscopic strain
measure, the stress cochain is obtained by transforming the global stress tensor field locally into vector-valued 2-cochains at each 2-cell,
and the mean-field cochain is obtained by (number-)averaging over all slip vectors assigned to 2-cells where slip events have taken place,
taking the dot product between this averaged vector and the direction of the slip being attempted, multiplying the direction of the slip being
attempted by the resulting number, and finally assigning the resulting vector to the 2-cell where the slip is being attempted.

"""


# ----- # ----- #  IMPORTS # ----- # ----- #

import numpy as np
import math
import random
from itertools import product
from collections import defaultdict
import multiprocessing as mp
import matplotlib as mpl ; mpl.rc('font',family='monospace')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys, os, shutil
from pathlib import Path
from time import time
# from datetime import datetime
from tqdm import tqdm


sys.path.append('./') ; sys.path.append('../') ; sys.path.append('../Voronoi_PCC_Analyser/')
from matgen.base import Vertex3D, Edge3D, Face3D, CellComplex
from dccstructure.iofiles import import_complex_data
from cellcomplex import createAllCells, scaleAllCells, createCellComplex
from cochain import Cochain

# ----- # ----- #  CLASSES # ----- # ----- #

class MHalgorithm:
    
    def __init__(self,
                 choices: list[int],
                 steps: int,
                 iterations: int,
                 cellcomplex: CellComplex):
            
        self.choices = choices
        self.steps = steps
        self.iterations = iterations
        self.cellcomplex = cellcomplex
        
    # -- #
    def setup(self,
              temperature: float,
              alpha: float,
              gamma: float,
              lambd: float,
              tau: float = None,
              stress: np.ndarray = None,
              starting_fraction: float = 0,
              starting_state: np.ndarray = None):
                
        self.stress = np.copy(stress)
        # self.stress_magnitude: float = np.sqrt(np.tensordot(stress_tensor, stress_tensor, axes = 2))
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        
        if starting_fraction != 0 and starting_state is not None:
            raise ValueError("Only one of [starting_fraction, starting_state] can be passed to MHalgorithm.setup().")
        self.starting_fraction = starting_fraction
        self.starting_state = starting_state
        
        
    # -- #
    def run_one_iteration(self, progress_bar: bool = False, get_acceptance_rate: bool = False):
        
        t0 = time()
        
        cellcomplex: CellComplex = self.cellcomplex
        Faces: list[Face3D] = cellcomplex.faces
        Faces_Slip: list[int] = self.choices
        Edges: list[Edge3D] = cellcomplex.edges
        Alpha: float = self.alpha
        Gamma: float = self.gamma
        Lambd: float = self.lambd
        Stress = self.stress
                
        boltzmann_constant: float = 1.380649e-23
        burgers: float = 2.556e-10 # (m) at 293 K (copper)
        
        probabilities_list: list[float] = []
        S2S2_list: list[float] = []
        P2S2_attempts_list: list[float] = []
        P2S2_successes_list: list[float] = []
        stress2S2_list: list[float] = []
        sys_energy_list: list[float] = [0]
        fraction_list: list[float] = []
        strain_list: list[float] = [0]
        activated_systems: dict[int, list] = {}
        state_vector_dict: dict[int, dict] = {}
        delta_pmt: list[float] = []
        system_changes: list[bool] = []
        meanfield_norm: list[float] = []
        if get_acceptance_rate == True:
            acceptance_rate_list: list[float] = []
        
        relaxation_energy: float = self.temperature * boltzmann_constant
        # relaxation: dict[int,float] = {p.id : temperature * boltzmann_constant for p in self.cellcomplex.polyhedra}
        
        """ Metropolis-Hastings algorithm """
        
        bins: dict[str,int] = {'A1':0,'A2':0,'A3':0,'A4':0,'A5':0,'A6':0,
                               'B1':0,'B2':0,'B3':0,'B4':0,'B5':0,'B6':0,
                               'C1':0,'C2':0,'C3':0,'C4':0,'C5':0,'C6':0,
                               'D1':0,'D2':0,'D3':0,'D4':0,'D5':0,'D6':0}
        
        slip_record: dict = defaultdict(int) # keeps track of which face-edge pairs have had their Ising state changed
        state_vector: dict = defaultdict(int) # keeps track of the current distribution of slipped face-edge pairs
        total_nr_pairs: int = len(Faces_Slip) * 3 # we consider both directionalities of one pair to be one and only one pair

        if self.starting_state is None:
            P2 = np.empty((0,4))
            if self.starting_fraction != 0:
                initial_slips: list[int] = random.sample(Faces_Slip, k = int(self.starting_fraction * len(Faces_Slip)))
                initial_slips.sort()
                
                # Compute an initial P2 value and initial distribution of activated slip systems
                for face in initial_slips:
                    edge: int = random.choice(Faces[face - 1].e_ids) # choose a slip direction at random
                    directionality: int = random.choice([1,-1])
                    slip_direction: np.ndarray = burgers * Faces[face - 1].measure**2 / Faces[face-1].support_volume * directionality * np.array(Edges[edge-1].unit_vector)
                    P2 = np.vstack((P2, np.array([face, slip_direction[0], slip_direction[1], slip_direction[2]])))
                    active_system: str = get_active_system(Faces[face - 1], Edges[edge - 1])
                    bins[active_system] += 1
                    state_vector[str(face) + ' ' + str(directionality * edge)] = 1
        
        else:
            P2 = np.empty((0,4))
            for row in self.starting_state:
                # Compute an initial P2 value and initial distribution of activated slip systems
                if int(row[2]) in [1,-1]:
                    state_vector[str(int(row[0])) + ' ' + str(int(row[1]))] = row[2]
                    face: Face3D = Faces[int(row[0]) - 1]
                    edge: Edge3D = Edges[abs(int(row[1])) - 1]
                    directionality = int(row[1])/abs(int(row[1]))
                    slip_direction: np.ndarray = burgers * face.measure**2 / face.support_volume * directionality * np.array(edge.unit_vector)
                    P2 = np.vstack((P2, np.array([face.id, slip_direction[0], slip_direction[1], slip_direction[2]])))
                    active_system: str = get_active_system(face, edge)
                    bins[active_system] += 1
        
        state_vector_dict.update({0 : state_vector.copy()})
        activated_systems.update({0 : bins.copy()})
        nr_slips: int = np.count_nonzero(list(state_vector.values()))
        delta_pmt.append(Alpha * total_nr_pairs / (Lambd * nr_slips)) if nr_slips > 0 else delta_pmt.append(None)
        fraction_list.append(nr_slips / total_nr_pairs)
        
        # Start the algorithm!
        
        steps: int = self.steps
        step: int = 1
        if progress_bar:
            progress = tqdm(total = steps, desc = 'Running simulation', miniters = 10_000, mininterval = 1)
        
        cyclic_counter = 1
        interval = 5_000
        
        while step <= steps:
                        
            # 1) Choose a random face-edge pair and invert the slip state of the pair (Ising model)
            
            face: int = random.choice(Faces_Slip) # choose a slip_face index at random (indices in 'choices' = cc.faces_slip start at 1)
            edge: int = random.choice(Faces[face - 1].e_ids) # choose a slip direction at random
            directionality: int = random.choice([1,-1])
            pair: str = str(face) + ' ' + str(directionality * edge)
                                    
            if state_vector[pair] == 1: # and slip_record[pair] == 0): # If this system has already slipped, unslip it
                state_vector[pair] = 0
                event_type: int = -1
                change_occurs = True
            elif state_vector[pair] == 0: # and slip_record[pair] == 0): # If this system has not slipped, slip it
                state_vector[pair] = 1
                event_type: int = 1
                change_occurs = True
            
            if change_occurs:
                
                # 2) Compute cochains
                
                edge_unit_vector: np.ndarray = np.array(Edges[edge-1].unit_vector)
                slip_direction: np.ndarray = burgers * Faces[face-1].measure**2 / Faces[face-1].support_volume * directionality * edge_unit_vector
                S2: Cochain = Cochain(cellcomplex, 2, np.array([[face, *slip_direction]])) # dimensions: L**2
                
                # 3) Compute change in energy of the system
                
                P2_on_cell: np.ndarray = np.copy(P2) # we will calculate the effect of the meanfield on the chosen 2-cell
                # if face in P2_on_cell[:,0].astype(int):
                #     P2_on_cell[np.argwhere(P2 == face)[0,0]] += np.array([0, slip_direction[0], slip_direction[1], slip_direction[2]])
                # else:
                #     P2_on_cell = np.vstack((P2_on_cell, np.array([face, slip_direction[0], slip_direction[1], slip_direction[2]])))
                P2_on_cell: Cochain = Cochain(cellcomplex, 2, np.array([[face, *meanfield(vector = edge_unit_vector, meanfield_array = P2_on_cell, N = total_nr_pairs)]])) # compute the resulting meanfield effect on the chosen 2-cell, considering the orientation of the slip
                S2S2: float = S2.inner_product_2022() # dimensions: L**3
                P2S2: float = P2_on_cell.inner_product_2022(S2) # dimensions: L**3
                stress2S2: float = np.dot(slip_direction, Stress @ np.array(Faces[face-1].normal) * Faces[face-1].measure) * Cochain(cellcomplex, 2, np.array([[face, 1]])).inner_product_2022()
                energy_diff = event_type * (Alpha * S2S2 - Gamma * stress2S2 - Lambd * P2S2)
                
                # 4) Acceptance criteria
                
                if energy_diff < 0:
                    change_occurs: bool = True
                    probabilities_list.append(1.0)
                    
                else:
                    prob = np.exp(- energy_diff / relaxation_energy)
                    probabilities_list.append(prob)
                    p_acc = - np.random.uniform(-1,0) # use this because np.random.random() does not include 1, while this function, as per the python documentation, includes -1.
                    
                    if prob < p_acc: # Whatever change we tried to make does not take effect, so we need to revert the slip state of the pair to its original value
                        change_occurs: bool = False
                        state_vector[pair] = 1 if state_vector[pair] == 0 else 0 # undo the change in part 1)
                        
                    else:
                        change_occurs: bool = True
                
                # 5) Take note of the changes to the system configuration if slip occurs
                
                if change_occurs:
                    
                    sys_energy_list.append(sys_energy_list[-1] + energy_diff) # the (un)slip event occurs and the system energy changes

                    if event_type == -1: # means that a face is being unslipped, so we need to remove it from the meanfield
                        P2[np.argwhere(P2 == face)[0,0]] -= np.array([0, slip_direction[0], slip_direction[1], slip_direction[2]]) # save changes to the mean-field
                        nr_slips -= 1
                        delta_pmt.append(Alpha * total_nr_pairs / (Lambd * nr_slips)) if nr_slips != 0 else None
                    elif event_type == 1: # means that a face is being slipped, so we need to add it to the meanfield
                        if face in P2[:,0]:
                            P2[np.argwhere(P2 == face)[0,0]] += np.array([0, slip_direction[0], slip_direction[1], slip_direction[2]]) # save changes to the mean-field
                        else:
                            P2 = np.vstack((P2, np.array([face, slip_direction[0], slip_direction[1], slip_direction[2]]))) # save changes to the mean-field
                        nr_slips += 1
                        delta_pmt.append(Alpha * total_nr_pairs / (Lambd * nr_slips))

                    slip_record[pair] += 1
                    fraction_list.append(nr_slips / total_nr_pairs)
                    strain_list.append(strain_list[-1] + event_type * np.linalg.norm(slip_direction))
                    active_system: str = get_active_system(cellcomplex.get_one('f', face), cellcomplex.get_one('e', edge))
                    bins[active_system] += event_type # This will be 1 if we are trying to slip a pair and -1 if we are trying to unslip a pair
                    
                    # To save a little bit of RAM, we only record the state_vector and the activated_systems at certain intervals during the simulation.
                    if cyclic_counter % interval == 0:
                        cyclic_counter = 1
                    elif (cyclic_counter % interval == interval - 1): #or step in list(range(int(0.99*steps), steps + 1))):
                        state_vector_dict.update({step : state_vector.copy()})
                        activated_systems.update({step : bins.copy()})
                        cyclic_counter += 1
                    else:
                        cyclic_counter += 1
                    
                    S2S2_list.append(S2S2)
                    P2S2_attempts_list.append(P2S2)
                    P2S2_successes_list.append(P2S2)
                    stress2S2_list.append(stress2S2)
                    
                elif not change_occurs: # here the trial change in the Ising model fails
                    
                    S2S2_list.append(None)
                    P2S2_attempts_list.append(P2S2)
                    P2S2_successes_list.append(None)
                    stress2S2_list.append(None)
                    delta_pmt.append(delta_pmt[-1])
                    fraction_list.append(fraction_list[-1])
                    strain_list.append(strain_list[-1])
                    sys_energy_list.append(sys_energy_list[-1])
                
            elif not change_occurs: # here the face chosen at random in 1) has already been flipped, so no change whatsoever occurs
                
                S2S2_list.append(None)
                P2S2_attempts_list.append(P2S2)
                P2S2_successes_list.append(None)
                stress2S2_list.append(None)
                delta_pmt.append(delta_pmt[-1])
                fraction_list.append(fraction_list[-1])
                strain_list.append(strain_list[-1])
                sys_energy_list.append(sys_energy_list[-1])
            
            meanfield_norm.append(np.linalg.norm(np.sum(P2[:,1:], axis = 0))/(len(Faces_Slip)*3))
            if get_acceptance_rate == True:
                system_changes.append(change_occurs)
                acceptance_rate_list.append(sum(system_changes)/step)
            step += 1
            if progress_bar:
                progress.update()
        
        if progress_bar:
            progress.close()
        
        if steps not in activated_systems.keys():
            activated_systems.update({steps : bins.copy()})
        if steps not in state_vector_dict.keys():
            state_vector_dict.update({steps : state_vector.copy()})
        
        self.probabilities: list[float] = probabilities_list
        self.S2S2: list[float] = S2S2_list
        self.P2S2_successes: list[float] = P2S2_successes_list
        self.P2S2_attempts: list[float] = P2S2_attempts_list
        self.stress2S2: list[float] = stress2S2_list
        self.delta: list[float] = delta_pmt
        self.fraction: list[float] = fraction_list
        self.strain: list[float] = strain_list
        self.sys_energy: list[float] = sys_energy_list
        self.state_vector: dict[int, dict] = state_vector_dict
        self.activated_systems: dict[int, dict] = activated_systems
        self.P2_final: dict[int, np.ndarray] = array_to_dict(P2)
        self.meanfield_norm: list[float] = meanfield_norm
        self.time: float = time() - t0
        if get_acceptance_rate == True:
            self.acceptance_rate: list[float] = acceptance_rate_list

        return self
    
    # -- #
    def run_no_meanfield(self, progress_bar: bool = False):
        
        t0 = time()
        
        cellcomplex: CellComplex = self.cellcomplex
        Faces: list[Face3D] = cellcomplex.faces
        Faces_Slip: list[int] = self.choices
        Edges: list[Edge3D] = cellcomplex.edges
        Alpha: float = self.alpha
        Gamma: float = self.gamma
        Lambd: float = 0
        Stress = self.stress
                
        boltzmann_constant: float = 1.380649e-23
        burgers: float = 2.556e-10 # (m) at 293 K (copper)
        
        probabilities_list: list[float] = []
        S2S2_list: list[float] = []
        stress2S2_list: list[float] = []
        sys_energy_list: list[float] = [0]
        fraction_list: list[float] = []
        strain_list: list[float] = [0]
        activated_systems: dict[int, list] = {}
        state_vector_dict: dict[int, dict] = {}
        system_changes: list[bool] = []
        
        relaxation_energy: float = self.temperature * boltzmann_constant
        # relaxation: dict[int,float] = {p.id : temperature * boltzmann_constant for p in self.cellcomplex.polyhedra}
        
        """ Metropolis-Hastings algorithm """
        
        bins: dict[str,int] = {'A2':0,'A3':0,'A6':0,
                               'B2':0,'B4':0,'B5':0,
                               'C1':0,'C3':0,'C5':0,
                               'D1':0,'D4':0,'D6':0}
        
        slip_record: dict = defaultdict(int) # keeps track of which face-edge pairs have had their Ising state changed
        state_vector: dict = defaultdict(int) # keeps track of the current distribution of slipped face-edge pairs
        total_nr_pairs: int = len(Faces_Slip) * 3 * 2

        if self.starting_state is None:
            if self.starting_fraction != 0:
                initial_slips: list[int] = random.sample(Faces_Slip, k = int(self.starting_fraction * len(Faces_Slip)))
                initial_slips.sort()
                
                # Compute an initial P2 value and initial distribution of activated slip systems
                for face in initial_slips:
                    edge: int = random.choice(Faces[face - 1].e_ids) # choose a slip direction at random
                    directionality: int = random.choice([1,-1])
                    active_system: str = get_active_system(Faces[face - 1], Edges[edge - 1])
                    bins[active_system] += 1
                    state_vector[str(face) + ' ' + str(directionality * edge)] = 1
        
        else:
            for row in self.starting_state:
                # Compute an initial P2 value and initial distribution of activated slip systems
                if int(row[2]) in [1,-1]:
                    state_vector[str(int(row[0])) + ' ' + str(int(row[1]))] = row[2]
                    face: Face3D = Faces[int(row[0]) - 1]
                    edge: Edge3D = Edges[abs(int(row[1])) - 1]
                    directionality = int(row[1])/abs(int(row[1]))
                    active_system: str = get_active_system(face, edge)
                    bins[active_system] += 1
        
        state_vector_dict.update({0 : state_vector.copy()})
        activated_systems.update({0 : bins.copy()})
        nr_slips: int = np.count_nonzero(list(state_vector.values()))
        fraction_list.append(nr_slips / total_nr_pairs)
        
        # Start the algorithm!
        
        steps: int = self.steps
        step: int = 1
        if progress_bar:
            progress = tqdm(total = steps, desc = 'Running simulation', miniters = 10_000, mininterval = 1)
        
        cyclic_counter = 1
        interval = 2000
        
        while step <= steps:
                        
            # 1) Choose a random face-edge pair and invert the slip state of the pair (Ising model)
            
            face: int = random.choice(Faces_Slip) # choose a slip_face index at random (indices in 'choices' = cc.faces_slip start at 1)
            edge: int = random.choice(Faces[face - 1].e_ids) # choose a slip direction at random
            directionality: int = random.choice([1,-1])
            pair: str = str(face) + ' ' + str(directionality * edge)
            
            # change_occurs: bool = False
                        
            if state_vector[pair] == 1: # and slip_record[pair] == 0): # If this system has already slipped, unslip it
                state_vector[pair] = 0
                event_type: int = -1
                change_occurs = True
            elif state_vector[pair] == 0: # and slip_record[pair] == 0): # If this system has not slipped, slip it
                state_vector[pair] = 1
                event_type: int = 1
                change_occurs = True
            
            if change_occurs:
                
                # 2) Compute cochains
                
                edge_unit_vector: np.ndarray = np.array(Edges[edge-1].unit_vector)
                slip_direction: np.ndarray = burgers * Faces[face-1].measure**2 / Faces[face-1].support_volume * directionality * edge_unit_vector
                S2: Cochain = Cochain(cellcomplex, 2, np.array([[face, *slip_direction]])) # dimensions: L**2
                
                # 3) Compute change in energy of the system
                
                # P2_on_cell: dict[int, np.ndarray] = P2.copy() # we will calculate the effect of the meanfield on the chosen 2-cell
                S2S2: float = S2.inner_product_2022() # dimensions: L**3
                P2S2: float = 0 # dimensions: L**3
                stress2S2: float = np.dot(slip_direction, Stress @ np.array(Faces[face-1].normal) * Faces[face-1].measure) * Cochain(cellcomplex, 2, np.array([[face, 1]])).inner_product_2022()
                energy_diff = event_type * (Alpha * S2S2 - Gamma * stress2S2 - Lambd * P2S2)
                                                
                # 4) Acceptance criteria
                
                if energy_diff < 0:
                    change_occurs: bool = True
                    probabilities_list.append(1.0)
                    
                else:
                    prob = np.exp(- energy_diff / relaxation_energy)
                    probabilities_list.append(prob)
                    p_acc = - np.random.uniform(-1,0) # use this because np.random.random() does not include 1, while this function, as per the python documentation, includes -1.
                    
                    if prob < p_acc: # Whatever change we tried to make does not take effect, so we need to revert the slip state of the pair to its original value
                        change_occurs: bool = False
                        state_vector[pair] = 1 if state_vector[pair] == 0 else 0 # undo the change in part 1)
                        
                    else:
                        change_occurs: bool = True
                
                # 5) Take note of the changes to the system configuration if slip occurs
                
                if change_occurs:
                    
                    sys_energy_list.append(sys_energy_list[-1] + energy_diff) # the (un)slip event occurs and the system energy changes

                    if event_type == -1: # means that a face is being unslipped, so we need to remove it from the meanfield
                        nr_slips -= 1
                    elif event_type == 1: # means that a face is being slipped, so we need to add it to the meanfield
                        nr_slips += 1

                    slip_record[pair] += 1
                    fraction_list.append(nr_slips / total_nr_pairs)
                    strain_list.append(strain_list[-1] + event_type * np.linalg.norm(slip_direction))
                    active_system: str = get_active_system(cellcomplex.get_one('f', face), cellcomplex.get_one('e', edge))
                    bins[active_system] += event_type # This will be 1 if we are trying to slip a pair and -1 if we are trying to unslip a pair
                    
                    # To save a little bit of RAM, we only record the state_vector and the activated_systems at certain intervals during the simulation.
                    if cyclic_counter % interval == 0:
                        cyclic_counter = 1
                    elif (cyclic_counter % interval == interval - 1 or step in list(range(int(0.99*steps), steps + 1))):
                        state_vector_dict.update({step : state_vector.copy()})
                        activated_systems.update({step : bins.copy()})
                        cyclic_counter += 1
                    else:
                        cyclic_counter += 1
                    
                    S2S2_list.append(S2S2)
                    stress2S2_list.append(stress2S2)
                    
                elif not change_occurs: # here the trial change in the Ising model fails
                    
                    S2S2_list.append(None)
                    stress2S2_list.append(None)
                    fraction_list.append(fraction_list[-1])
                    strain_list.append(strain_list[-1])
                    sys_energy_list.append(sys_energy_list[-1])
                
            elif not change_occurs: # here the face chosen at random in 1) has already been flipped, so no change whatsoever occurs
                
                S2S2_list.append(None)
                stress2S2_list.append(None)
                fraction_list.append(fraction_list[-1])
                strain_list.append(strain_list[-1])
                sys_energy_list.append(sys_energy_list[-1])
            
            system_changes.append(change_occurs)
            step += 1
            if progress_bar:
                progress.update()
        
        if steps not in activated_systems.keys():
            activated_systems.update({steps : bins.copy()})
        if steps not in state_vector_dict.keys():
            state_vector_dict.update({steps : state_vector.copy()})
        
        self.probabilities: list[float] = probabilities_list
        self.S2S2: list[float] = S2S2_list
        self.stress2S2: list[float] = stress2S2_list
        self.fraction: list[float] = fraction_list
        self.strain: list[float] = strain_list
        self.sys_energy: list[float] = sys_energy_list
        self.state_vector: dict[int, dict] = state_vector_dict
        self.activated_systems: dict[int, dict] = activated_systems
        self.acceptance_rate: tuple[float] = [sum(system_changes[0:k])/k for k in range(1,len(system_changes))]
        self.time: float = time() - t0
        return self
     
     # -- #
    """
    def run_with_thermodynamics(self,
                                density: float | list[float],
                                specific_heat: float | list[float],
                                temperature: float | list[float]):
        
        
        density_list: list = [density] * self.cellcomplex.vernb if isinstance(density, float) else density
        specific_heat_list: list = [specific_heat] * self.cellcomplex.vernb if isinstance(specific_heat, float) else specific_heat
        temperature_list: list = [temperature] * self.cellcomplex.vernb if isinstance(temperature, float) else temperature
        
        relaxation_energy: float = self.temperature * boltzmann_constant
    """
    
    # -- #
    def run_multiple_iterations(self, iterations: int = None):
        
        if iterations is None:
            iterations = self.iterations
            
        results: list = []
        # fraction_avg: list = []
        # sysenergy_avg: list = []
        
        with mp.Pool() as pool:
            for n in range(iterations):
                
                iteration = MHalgorithm(self.choices, self.steps, 1, self.cellcomplex)
                iteration.setup(self.temperature,
                                self.alpha,
                                self.gamma, 
                                self.lambd,
                                stress = self.stress,
                                starting_fraction = self.starting_fraction)
                function = iteration.run_one_iteration
                results.append(pool.apply_async(function))
            
            pool.close()
            pool.join()
            
            results = [r.get() for r in results]
        
        newdir: str = r'./multiple_iterations_out'
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        else:
            shutil.rmtree(newdir) # Removes all the subdirectories!
            os.makedirs(newdir)
        
        fname = newdir + '/'

        fractions = np.empty((0, self.steps+1))
        energies = np.empty((0, self.steps+1))
        
        # Plot fraction
        fig = plt.figure(figsize = (11, 11))
        ax = fig.add_subplot()
        ax.set_xlabel('Steps', size = 32, fontname = 'monospace', labelpad=15)
        ax.set_xlim([0, (self.steps+1)/1e3])
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.set_ylabel('n/N', size = 32, fontname = 'monospace', labelpad=15)
        ax.set_ylim([0, 0.8])
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.tick_params(axis = 'both', which = 'major', length = 10, labelfontfamily = 'monospace', labelsize = 25)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        for result in results:
            ax.scatter(np.arange(0, self.steps + 1, 1)/1e3, result.fraction, s = 6, marker = 'o')
            fractions = np.vstack((fractions, np.array([result.fraction])))
        plt.subplots_adjust(left=0.145, bottom=0.115, right=0.93, top=0.985, wspace=0, hspace=0)
        plt.savefig(fname = fname + 'fraction.png', dpi = 500)

        # Plot energy
        fig = plt.figure(figsize = (11,11))
        ax = fig.add_subplot()
        ax.set_xlabel('Steps', size = 32, fontname = 'monospace', labelpad=15)
        ax.set_xlim([0, (self.steps+1)/1e3])
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.set_ylabel('Dissipated energy (nJ)', size = 35, fontname = 'monospace', labelpad = 15) # ×
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.tick_params(axis = 'both', which = 'major', length = 10, labelfontfamily = 'monospace', labelsize = 25)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        for result in results:
            ax.scatter(np.arange(0, self.steps + 1, 1)/1e3, [x/1e-9 for x in result.sys_energy], s = 6, marker = 'o')
            energies = np.vstack((energies, np.array([result.sys_energy])))
        ax.set_ylim(bottom = 0)
        plt.subplots_adjust(left=0.145, bottom=0.115, right=0.93, top=0.985, wspace=0, hspace=0)
        plt.savefig(fname = fname + 'energy.png', dpi = 500)
                
        np.savetxt(fname + 'fractions.txt', fractions)
        np.savetxt(fname + 'energies.txt', energies)
        
        return results
        
    # -- #
    def plot_one_iteration(self, MAG: float, save_figure: tuple[bool, str] = (False, None)):
        
        fig = plt.figure(figsize = (30, 15))
        if MAG / 1e6 < 0.1:
            title = f'|σ| = {int(MAG)} Pa, '
        elif MAG / 1e6 >= 0.1:
            title = f'|σ| = {MAG/1e6:.1f} MPa, '
        title = title + f'α = {self.alpha:1.0e} Pa, γ = {self.gamma:1.0e}, λ = {self.lambd:1.0e} Pa'
        title = title + '   (time: {:.1f} min)'.format(self.time/60)
        fig.suptitle(title, fontsize = 20, y = 0.92)
        
        fig.subplots_adjust(wspace = 0.3)
        
        # eV = 1.602176634e-19 # (J)
        eV = 1e-9
        energy_units = 'nJ'
        
        # Plot S2:S2
        xx = self.steps
        yy = self.S2S2
        pmt = self.alpha
        ax = fig.add_subplot(341)
        ax.set_xlim([0, xx / 1e3])
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.set_xlabel('Steps (thousands)', size = 15)
        try:
            ax.set_ylim([0, 1.1 * pmt * max([x for x in yy if x is not None]) / eV])
        except:
            pass
        try:
            ax.yaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(pmt * max([y for y in yy if y is not None]) / eV))))
            ax.yaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(pmt * max([y for y in yy if y is not None]) / eV) - 1)))
        except:
            pass
        ax.set_ylabel(f'α⟨S$^{2}$, S$^{2}$⟩ ({energy_units})', size = 15) # (m$^{3}$)
        ax.scatter(np.array(range(xx)) / 1e3, [x * pmt / eV if x is not None else None for x in yy], s = 6, c = 'darkgreen', marker = 'o')
        
        # Plot stress2:S2
        yy = self.stress2S2
        pmt = self.gamma
        ax = fig.add_subplot(345)
        ax.set_xlim([0, xx / 1e3])
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.set_xlabel('Steps (thousands)', size = 15)
        try:
            ax.set_ylim([0, 1.1 * pmt * max([x for x in yy if x is not None]) / eV])
        except:
            pass
        ax.set_ylabel(f'γ⟨σ$^{2}$, S$^{2}$⟩ ({energy_units})', size = 15)
        try:
            ax.yaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(pmt * max([y for y in yy if y is not None]) / eV))))
            ax.yaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(pmt * max([y for y in yy if y is not None]) / eV) - 1)))
        except:
            pass
        ax.scatter(np.array(range(xx)) / 1e3, [x * pmt / eV if x is not None else None for x in yy], s = 6, c = 'firebrick', marker = 'o')
        
        # Plot P2:S2
        yy1 = self.P2S2_attempts
        yy2 = self.P2S2_successes
        pmt = self.lambd
        ax = fig.add_subplot(349)
        ax.set_xlim([0, xx / 1e3])
        ax.set_xlabel('Steps (thousands)', size = 15)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        try:
            ax.set_ylim([0, pmt * max([y for y in yy1 if y is not None]) / eV])
        except:
            pass
        try:
            ax.yaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(pmt * max([y for y in yy1 if y is not None]) / eV))))
            ax.yaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(pmt * max([y for y in yy1 if y is not None]) / eV) - 1)))
        except:
            pass
        ax.set_ylabel(f'λ⟨P$^{2}$, S$^{2}$⟩ ({energy_units})', size = 15) # (m$^{2}$)
        ax.scatter(np.array(range(xx)) / 1e3, [y * pmt / eV if y is not None else None for y in yy1], s = 8, marker = 'o', c = 'darkgray', label = 'Attempts', alpha=0.35)
        ax.scatter(np.array(range(xx)) / 1e3, [y * pmt / eV if y is not None else None for y in yy2], s = 6, marker = 'o', c = 'purple', label = 'Successful attempts')
        plt.legend(loc = 'lower right', fontsize = 'xx-large', markerscale = 4, framealpha = 0.5)
        
        # Plot fraction
        ax_f = fig.add_subplot(2,4,(2,3))
        ax_d = ax_f.twinx()
        ax_d.tick_params(axis = 'y', labelcolor = 'k')
        ax_d.set_ylabel('δ (dimensionless)', size = 15, c = 'k')
        ax_d.set_yscale('log')
        ax_d.scatter(np.arange(xx+1) / 1e3, [x if x is not None else None for x in self.delta], s = 2, c = 'k', marker = 'o')
        ax_f.set_xlim([0, xx / 1e3])
        ax_f.xaxis.set_major_locator(MultipleLocator(100))
        ax_f.xaxis.set_minor_locator(MultipleLocator(10))
        ax_f.set_xlabel('Steps (thousands)', size = 15)
        ax_f.set_ylim([0, 0.6])
        ax_f.tick_params(axis = 'y', labelcolor = 'b')
        ax_f.set_ylabel('Fraction of slipped 2-cells', size = 15, c = 'b')
        ax_f.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax_f.tick_params(axis = 'both', which = 'minor', bottom = True, left = True)
        ax_f.scatter(np.arange(xx + 1) / 1e3, self.fraction, s = 8, c = 'b', marker = 'o')
        # ax_f.plot(np.array(range(self.steps + 1)) / 1e3, [len(self.cellcomplex.faces_slip)/self.cellcomplex.facenb] * (self.steps + 1), 'k--')
        # ax_f.text(x = 3*int((self.steps + 1)/5) / 1e3, y = len(self.cellcomplex.faces_slip)/self.cellcomplex.facenb + 0.007, s = 'maximum allowed fraction', c = 'black', fontsize = 13)
        
        # Plot energy
        yy = self.sys_energy
        ax_e = fig.add_subplot(2,4,(6,7))
        ax_d = ax_e.twinx()
        ax_d.tick_params(axis = 'y', labelcolor = 'k')
        ax_d.set_ylabel('δ (dimensionless)', size = 15, c = 'k')
        ax_d.set_yscale('log')
        ax_d.scatter(np.arange(xx+1) / 1e3, [x if x is not None else None for x in self.delta], s = 2, c = 'k', marker = 'o')
        ax_e.set_xlim([0, xx / 1e3])
        ax_e.xaxis.set_major_locator(MultipleLocator(100))
        ax_e.xaxis.set_minor_locator(MultipleLocator(10))
        ax_e.set_xlabel('Steps (thousands)', size = 15)
        try:
            ax_e.set_ylim([(min(yy) * 1.01)  / eV, (max(yy) * 1.01) / eV])
        except:
            pass
        try:
            if max(yy) == 0:
                ax_e.yaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(-min(yy) / eV) - 2)))
            elif min(yy) == 0:
                ax_e.yaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(max(yy) / eV) - 2)))
        except:
            pass
        ax_e.set_ylabel(f'Cumulative energy of the system ({energy_units})', size = 15, c = 'r')
        ax_e.scatter(np.arange(xx + 1) / 1e3, [x / eV for x in yy], s = 8, c = 'r', marker = 'o')
        ax_e.tick_params(axis = 'y', labelcolor = 'r')
        
        # Plot activated systems
        yy = self.activated_systems
        rows = len(yy)
        counter = 1
        for key in yy.keys():
            ax = fig.add_subplot(rows, 4, 4*counter)
            ax.bar(x = list(yy[key].keys()),
                   height = list(yy[key].values()),
                   width = .4,
                   color = 'darkgoldenrod',
                   edgecolor = 'black')
            ax.set_ylabel(f'Events ({key})', size = 10)
            ax.get_xaxis().set_visible(False)
            counter += 1
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel('Slip systems', size = 20, labelpad = 13)
        ax.hlines(y = max(list(yy[key].values())), xmin = 0, xmax = len(yy[key].keys())-1, linestyles = 'dashed', color = 'black')

        if save_figure[0] == True:
            plt.savefig(fname = save_figure[1], dpi = 300)
    
    # -- #
    def plot_S2S2(self, save_figure: tuple[bool, str] = (False, None)):
        
        pmt = self.alpha
        xx = np.arange(self.steps) / 1e3
        yy = [pmt * y if y is not None else None for y in self.S2S2]
        scaling = orderOfMagnitude(max([y for y in yy if y is not None]))
        yy = [10**(-scaling) * y if y is not None else None for y in yy]
                
        fig = plt.figure(figsize = (11, 11))
        ax = fig.add_subplot()
        # x axis
        ax.set_xlim([0, max(xx)])
        ax.xaxis.set_major_locator(MultipleLocator(max(xx)/10))
        ax.xaxis.set_minor_locator(MultipleLocator(2*max(xx)/100))
        ax.set_xlabel('Steps (thousands)', size = 25, fontname = 'monospace', labelpad=10)
        # y axis
        try:
            ax.set_ylim([0, 1.1 * max([x for x in yy if x is not None])])
        except: pass
        try:
            ax.yaxis.set_major_locator(MultipleLocator(5 * 10 ** (orderOfMagnitude(max([y for y in yy if y is not None])) - 1)))
            ax.yaxis.set_minor_locator(MultipleLocator(5 * 10 ** (orderOfMagnitude(max([y for y in yy if y is not None])) - 2)))
        except: pass
        if scaling != 0:
            ax.set_ylabel('$\\alpha \\langle S^2, S^2 \\rangle$ ($\\times$10' + get_superscript(str(scaling)) + ' J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        else:
            ax.set_ylabel('$\\alpha \\langle S^2, S^2 \\rangle$ (J)', size = 20, fontname = 'monospace', labelpad = 10) # ×
        # tick parametres
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        # plot
        ax.scatter(xx, [y if y is not None else None for y in yy], s = 6, c = 'black', marker = 'o')
        # adjust
        plt.subplots_adjust(left=0.115, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        # save figure
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
    
    # -- #
    def plot_stress2S2(self, save_figure: tuple[bool, str] = (False, None)):
        
        pmt = self.gamma
        xx = np.arange(self.steps) / 1e3
        yy = [pmt * y if y is not None else None for y in self.stress2S2]
        scaling = orderOfMagnitude(max([y for y in yy if y is not None]))
        yy = [10**(-scaling) * y if y is not None else None for y in yy]
                
        fig = plt.figure(figsize = (11, 11))
        ax = fig.add_subplot()
        # x axis
        ax.set_xlim([0, max(xx)])
        ax.xaxis.set_major_locator(MultipleLocator(max(xx)/10))
        ax.xaxis.set_minor_locator(MultipleLocator(2*max(xx)/100))
        ax.set_xlabel('Steps (thousands)', size = 25, fontname = 'monospace', labelpad=10)
        # y axis
        try:
            ax.set_ylim([0, 1.1 * max([y for y in yy if y is not None])])
        except: pass
        try:
            ax.yaxis.set_major_locator(MultipleLocator(5 * 10 ** (orderOfMagnitude(max([y for y in yy if y is not None])) - 1)))
            ax.yaxis.set_minor_locator(MultipleLocator(5 * 10 ** (orderOfMagnitude(max([y for y in yy if y is not None])) - 2)))
        except: pass
        if scaling != 0:
            ax.set_ylabel('$\\langle \\sigma^2, S^2 \\rangle$ ($\\times$10' + get_superscript(str(scaling)) + ' J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        else:
            ax.set_ylabel('$\\langle \\sigma^2, S^2 \\rangle$ (J)', size = 20, fontname = 'monospace', labelpad = 10) # ×
        # tick parametres
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        # plot
        ax.scatter(xx, [y if y is not None else None for y in yy], s = 6, c = 'black', marker = 'o')
        # adjust
        plt.subplots_adjust(left=0.115, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        # save figure
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
        
    # -- #
    def plot_P2S2(self, save_figure: tuple[bool, str] = (False, None)):
        
        pmt = self.lambd
        xx = np.arange(self.steps) / 1e3
        yy1 = [pmt * y if y is not None else None for y in self.P2S2_attempts]
        yy2 = [pmt * y if y is not None else None for y in self.P2S2_successes]
        scaling = orderOfMagnitude(max([y for y in yy1 if y is not None]))
        yy1 = [10**(-scaling) * y if y is not None else None for y in yy1]
        yy2 = [10**(-scaling) * y if y is not None else None for y in yy2]
        
        fig = plt.figure(figsize = (11, 11))
        ax = fig.add_subplot()
        # x axis
        ax.set_xlim([0, max(xx)])
        ax.xaxis.set_major_locator(MultipleLocator(max(xx)/10))
        ax.xaxis.set_minor_locator(MultipleLocator(2*max(xx)/100))
        ax.set_xlabel('Steps (thousands)', size = 25, fontname = 'monospace', labelpad=10)
        # y axis
        try:
            ax.set_ylim([1.1 * min([y for y in yy1 if y is not None]), 1.1 * max([y for y in yy1 if y is not None])])
        except: pass
        try:
            ax.yaxis.set_major_locator(MultipleLocator(1 * 10 ** (orderOfMagnitude(max([y for y in yy1 if y is not None])))))
            ax.yaxis.set_minor_locator(MultipleLocator(2 * 10 ** (orderOfMagnitude(max([y for y in yy1 if y is not None])) - 1)))
        except: pass
        if scaling != 0:
            ax.set_ylabel('$\\lambda \\langle P^2, S^2 \\rangle$ ($\\times$10' + get_superscript(str(scaling)) + ' J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        else:
            ax.set_ylabel('$\\lambda \\langle P^2, S^2 \\rangle$ (J)', size = 20, fontname = 'monospace', labelpad = 10) # ×
        # tick parametres
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        # plot
        ax.scatter(xx, [y if y is not None else None for y in yy1], s = 8, marker = 'o', c = 'darkgray', label = 'Attempts', alpha=0.35)
        ax.scatter(xx, [y if y is not None else None for y in yy2], s = 6, marker = 'o', c = 'black', label = 'Successful attempts')
        # legend
        plt.legend(loc = 'lower left', fontsize = 'xx-large', markerscale = 4, framealpha = 0.5)
        # adjust
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        # save figure
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
            
    # -- #
    def plot_Acceptancerate(self, save_figure: tuple[bool, str] = (False, None)):
        
        fig = plt.figure(figsize = (11, 11))
        xx = self.steps
        ax = fig.add_subplot()
        ax.set_xlim([0, xx / 1e3])
        ax.xaxis.set_major_locator(MultipleLocator(xx/10000))
        ax.xaxis.set_minor_locator(MultipleLocator(2*xx/100000))
        ax.set_xlabel('Steps (thousands)', size = 26, fontname = 'monospace', labelpad=10)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Acceptance rate', size = 26, fontname = 'monospace', labelpad=10)
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.scatter(np.arange(len(self.acceptance_rate)) / 1e3, self.acceptance_rate, s = 8, c = 'black', marker = 'o')
        # tick parametres
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        plt.subplots_adjust(left=0.115, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)

    # -- #
    def plot_fraction(self, save_figure: tuple[bool, str] = (False, None)):
        
        fig = plt.figure(figsize = (11, 11))
        xx = self.steps
        ax_f = fig.add_subplot()
        # ax_d = ax_f.twinx()
        # ax_d.tick_params(axis = 'y', labelcolor = 'k')
        # ax_d.set_ylabel('δ (μm$^{3}$)', size = 15, c = 'k')
        # ax_d.set_yscale('log')
        # ax_d.scatter(np.arange(xx+1) / 1e3, [x / (1e-6)**3 if x is not None else None for x in self.delta], s = 2, c = 'k', marker = 'o')
        ax_f.set_xlim([0, xx / 1e3])
        ax_f.xaxis.set_major_locator(MultipleLocator(xx/10000))
        ax_f.xaxis.set_minor_locator(MultipleLocator(2*xx/100000))
        ax_f.set_xlabel('Steps (thousands)', size = 26, fontname = 'monospace', labelpad=10)
        ax_f.set_ylim([0, 0.7])
        ax_f.set_ylabel('Fraction of slipped 2-cells', size = 26, fontname = 'monospace', labelpad=10)
        ax_f.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax_f.scatter(np.arange(xx + 1) / 1e3, self.fraction, s = 8, c = 'black', marker = 'o')
        # tick parametres
        ax_f.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax_f.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        plt.subplots_adjust(left=0.115, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
    
    # -- #
    def plot_energy(self, save_figure: tuple[bool, str] = (False, None)):
        
        xx = np.arange(self.steps+1) / 1e3
        yy = [y if y is not None else None for y in self.sys_energy]
        scaling = orderOfMagnitude(max([abs(y) for y in yy if y is not None]))
        yy = [10**(-scaling) * y if y is not None else None for y in yy]
                
        fig = plt.figure(figsize = (11, 11))
        ax = fig.add_subplot()
        # x axis
        ax.set_xlim([0, max(xx)])
        ax.xaxis.set_major_locator(MultipleLocator(max(xx)/10))
        ax.xaxis.set_minor_locator(MultipleLocator(2*max(xx)/100))
        ax.set_xlabel('Steps (thousands)', size = 25, fontname = 'monospace', labelpad=10)
        # y axis
        try:
            ax.set_ylim([1.1 * min([y for y in yy if y is not None]), 0])
        except: pass
        try:
            ax.yaxis.set_major_locator(MultipleLocator(1 * 10 ** (orderOfMagnitude(max([abs(y) for y in yy if y is not None])))))
            ax.yaxis.set_minor_locator(MultipleLocator(2 * 10 ** (orderOfMagnitude(max([abs(y) for y in yy if y is not None])) - 1)))
        except: pass
        if scaling != 0:
            ax.set_ylabel('Cumulative dissipated energy ($\\times$10' + get_superscript(str(scaling)) + ' J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        else:
            ax.set_ylabel('Cumulative dissipated energy (J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        # tick parametres
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        # plot
        ax.scatter(xx, [y if y is not None else None for y in yy], s = 6, c = 'black', marker = 'o')
        # adjust
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        # save figure
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
    
    # -- #
    def plot_activated_systems(self, number: int = None, save_figure: tuple[bool, str] = (False, None)):
        
        fig = plt.figure(figsize = (22, 18))
        
        yy = self.activated_systems
        
        if number is None:
            rows = len(yy)
            counter = 1
            for key in yy.keys():
                ax = fig.add_subplot(rows, 4, 4*counter)
                ax.bar(x = list(yy[key].keys()),
                       height = list(yy[key].values()),
                       width = .4,
                       color = 'black',
                       edgecolor = 'black')
                ax.set_ylabel(f'Events ({key})', size = 10, fontname = 'monospace', labelpad=10)
                ax.get_xaxis().set_visible(False)
                counter += 1
            ax.get_xaxis().set_visible(True)
            ax.set_xlabel('Slip systems', size = 20, labelpad = 13, fontname = 'monospace')
            ax.hlines(y = max(list(yy[key].values())), xmin = 0, xmax = len(yy[key].keys())-1, linestyles = 'dashed', color = 'black')
        
        elif isinstance(number, int):
            ax = fig.add_subplot(rows, 4, 4*counter)
            ax.bar(x = list(yy[key].keys()),
                   height = list(yy[key].values()),
                   width = .4,
                   color = 'black',
                   edgecolor = 'black')
            ax.set_ylabel(f'Events ({key})', size = 10, fontname = 'monospace', labelpad=10)
            ax.set_xlabel('Slip systems', size = 20, labelpad = 13, fontname = 'monospace')
            ax.hlines(y = max(list(yy[key].values())), xmin = 0, xmax = len(yy[key].keys())-1, linestyles = 'dashed', color = 'black')
            
        plt.subplots_adjust(left=0.1, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
        
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
    
    # -- #
    def plot_Delta_vsFraction(self, save_figure: tuple[bool, str] = (False, None)):
                
        fig = plt.figure(figsize = (11, 11))
                
        ax = fig.add_subplot()
        ax.set_xlim([0, max(self.fraction)])
        ax.set_xlabel('Fraction of slipped 2-cells', size = 25, fontname = 'monospace', labelpad=10)
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.set_ylabel('$\\delta$', size = 25, fontname = 'monospace', labelpad=10)
        ax.set_yscale('log')
        ax.scatter([x if x is not None else None for x in self.fraction], [y if y is not None else None for y in self.delta], c = 'black', s = 4)
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        plt.subplots_adjust(left=0.11, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)
        
    # -- #
    def plot_Delta_vsEnergy(self, save_figure: tuple[bool, str] = (False, None)):
        
        xx = [x if x is not None else None for x in self.sys_energy]
        scaling = orderOfMagnitude(max([abs(x) for x in xx if x is not None]))
        xx = [10**(-scaling) * x if x is not None else None for x in xx]

        fig = plt.figure(figsize = (11, 11))
        ax = fig.add_subplot()
        ax.set_xlim([min(xx), 0])
        if scaling != 0:
            ax.set_xlabel('Cumulative system energy difference ($\\times$10' + get_superscript(str(scaling)) + ' J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        else:
            ax.set_xlabel('Cumulative system energy difference (J)', size = 25, fontname = 'monospace', labelpad = 10) # ×
        ax.invert_xaxis()
        try:
            ax.xaxis.set_major_locator(MultipleLocator(1 * 10 ** (orderOfMagnitude(max([abs(x) for x in xx if x is not None])))))
            ax.xaxis.set_minor_locator(MultipleLocator(2 * 10 ** (orderOfMagnitude(max([abs(x) for x in xx if x is not None])) - 1)))
        except: pass
        ax.set_ylabel('$\\delta$', size = 25, fontname = 'monospace', labelpad=10)
        ax.set_yscale('log')
        ax.scatter(xx, [y if y is not None else None for y in self.delta], c = 'black', s = 4)
        # tick parametres
        ax.tick_params(axis = 'both', which = 'major', length = 8, labelfontfamily = 'monospace', labelsize = 20)
        ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
        plt.subplots_adjust(left=0.11, bottom=0.1, right=0.97, top=0.97, wspace=0, hspace=0)
        
        if save_figure[0] == True:
            plt.savefig(save_figure[1], dpi = 700)

        

# ----- # ----- #  FUNCTIONS # ----- # ----- #


def get_permutations(list1: list,
                     list2: list,
                     list3: list = None) -> tuple:
    
    if list3 is not None:
        cartesian_product = product(list1, list2, list3)
        perms = tuple(perm for perm in cartesian_product)
    else:
        cartesian_product = product(list1, list2)
        perms = tuple(perm for perm in cartesian_product)
        
    return perms


def get_incident_faces_on_vertex(vertex: int | Vertex3D, cellcomplex: CellComplex) -> list[int]:
    
    if isinstance(vertex, int):
        vertex: Vertex3D = cellcomplex.get_one('v', vertex)
    
    # Create a list of indices of the faces incident on 'vertex'.
    incident_faces = set()
    for p_id in vertex.incident_polyhedra_ids:
        faces = cellcomplex.get_one('p', p_id).f_ids
        for f_id in faces:
            if vertex.id in cellcomplex.get_one('f', f_id).v_ids:
                incident_faces.add(f_id)
    
    return list(incident_faces)


def tensor22cochain(vertex: int | Vertex3D,
                    cellcomplex: CellComplex,
                    tensor: np.ndarray) -> tuple:
    
    if isinstance(vertex, int):
        vertex: Vertex3D = cellcomplex.get_one('v', vertex)
    
    # Create a list of indices of the faces incident on 'vertex'.
    incident_faces: list[int] = get_incident_faces_on_vertex(vertex, cellcomplex)
    
    # Create the transformation matrix A, whose rows are the area-normal vectors of the faces incident on 'vertex'.
    # By placing the vectors on the rows, A must be multiplied transposed on the right.
    A: np.ndarray = np.zeros((len(incident_faces), 3))
    row: int = 0
    for face in incident_faces:
        face: Face3D = cellcomplex.get_one('f', face)
        A[row,:] = np.array(face.normal) * face.measure
        row += 1
    
    # Multiply A and the tensor at the node together to obtain a matrix 
    Asigma = np.matmul(tensor, A.transpose())
    
    to_delete: list = []
    for col in range(np.shape(Asigma)[1]):
        if np.linalg.norm(Asigma[:,col]) == 0:
            to_delete.append(col)
    
    incident_faces = list(np.delete(incident_faces, to_delete))
    Asigma = np.delete(Asigma, to_delete, axis = 1)
    
    cochain: list = list(zip(incident_faces, Asigma.transpose()))
    cochain: dict = dict(cochain)
    
    return A, cochain


def cochain22tensor(vertex: int | Vertex3D,
                    cellcomplex: CellComplex,
                    cochain: Cochain) -> tuple:
    
    if isinstance(vertex, int):
        vertex: Vertex3D = cellcomplex.get_one('v', vertex)
    
    # The matrix A transforms a tensor on a node into vectors on the faces incident on that node. From it we can
    # define its Moore-Penrose pseudo-inverse, which transforms vectors on the incident faces into a tensor defined
    # at the vertex.
    A, _ = tensor22cochain(vertex, cellcomplex, np.eye(3,3))
    A_inv: np.ndarray = np.linalg.pinv(A)
    
    # Create a list of indices of the faces incident on 'vertex'.
    incident_faces: list[int] = get_incident_faces_on_vertex(vertex, cellcomplex)
    
    # The Moore-Penrose inverse of the matrix A obtained in tensor22cochain(), A_inv, is also a transformation relative
    # to a specific vertex. The tensor at the vertex will be the sum of the tensors obtained by applying the inverse to
    # each face incident on that vertex.
    
    tensor = np.array([cochain.cochain().get(cellcomplex.get_one('f', f_id), np.zeros(3)) for f_id in incident_faces])
    tensor = np.matmul(A_inv, tensor)
    
    return A_inv, tensor


def stress_cochain(cellcomplex: CellComplex,
                   stress_tensor: np.ndarray) -> Cochain:
    
    Asigma: dict = {}
    
    for v in cellcomplex.vertices:
        _, cochain = tensor22cochain(v, cellcomplex, stress_tensor)
        for k in cochain.keys():
            if k in Asigma.keys():
                Asigma[k] += cochain[k]
            else:
                Asigma.update({k : cochain[k]})
    
    return Cochain(cellcomplex, 2, Asigma)


def meanfield(vector: np.ndarray, meanfield_array: np.ndarray, N: int) -> np.ndarray:
    
    norm = np.linalg.norm(vector)
    if (not np.isclose(norm, 1) and not np.isclose(norm, 0)):
        vector = vector / norm
    
    values = meanfield_array[:,1:4]
    dot_products = np.dot(values, vector)
    meanfield_effect = np.sum(dot_products) / N * vector
    
    return meanfield_effect


def get_active_system(face: Face3D, edge: Edge3D) -> str:
    
    active_plane: str = list(face.slip_plane.keys())[0]
    active_direction: str = list(edge.slip_direction.keys())[0]
    return active_plane + active_direction


def orderOfMagnitude(number):
    if number == 0:
        return 0
    oom = math.floor(math.log(abs(number), 10))
    if abs(number / 10**oom) > 10:
        raise ValueError()
    else:
        return oom


def array_to_dict(a: np.ndarray) -> dict:
    return {int(a[i,0]) : a[i,1:] for i in range(np.shape(a)[0])}


def get_superscript(x): 
    normal = "0123456789+-"
    super_s = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
    res = x.maketrans(''.join(normal), ''.join(super_s)) 
    return x.translate(res) 


# ----- # ----- #  CODE # ----- # ----- #

# ---- # Get arguments from terminal # ---- #


if __name__ == '__main__':
    
    """ # ---- 1. Create cell complex ---- # """
    
    t0 = time()
    
    ccdata = '../Built_Complexes/FCC_10x10x10'
    data_folder = Path(ccdata)
    text = ccdata.split('x')
    complex_size = [int(text[-3][7:]), int(text[-2]), int(text[-1].split('/')[0])] # the number of computational unit cells in each 3D dimension
    nodes, edges, faces, faces_slip, faces_areas, faces_normals, volumes, volumes_vols, nr_cells, A0, A1, A2, A3, B1, B2, B3 = import_complex_data(data_folder)
    edges += 1 ; faces += 1 ; volumes += 1  ;  edges = edges.astype(int)
    faces_slip = [x + 1 for x in faces_slip]
    A0[:,0:2] += 1 ; A1[:,0:2] += 1 ; A2[:,0:2] += 1 ; A3[:,0:2] += 1 ; B1[:,0:2] += 1 ; B2[:,0:2] += 1 ; B3[:,0:2] += 1
    
    # Turns arrays into lists of instances of Vertex3D, Edge3D, Face3D and Polyhedra, respectively
    Nodes, Edges, Faces, Polyhedra = createAllCells((nodes + 0.5),
                                                    edges,
                                                    faces,
                                                    volumes,
                                                    faces_normals,
                                                    A0, A1, A2, A3, B1, B2, B3,
                                                    faces_areas,
                                                    volumes_vols,
                                                    perturb_vertices = False,
                                                    complex_size = complex_size)
    
    del A0, A1, A2, A3, B1, B2, B3
    del nodes, edges, faces, faces_areas, faces_normals, volumes, volumes_vols
    
    # dislocation_density = 1e12
    # if complex_size[0] %2 == 0:
    #     cube_side = 2 * np.sqrt(2) * complex_size[0] * (15 / np.sqrt(3 * dislocation_density))
    # else:
    #     n = complex_size[0]
    #     cube_side = (2 * np.sqrt(2) * n**2 * 15) /  np.sqrt(dislocation_density * (3*n**2-2*n+1))
    #     del n
        
    Nodes, Edges, Faces, Polyhedra = scaleAllCells(Nodes, Edges, Faces, Polyhedra, scaling = float(8e-6 / complex_size[0]))
    cc: CellComplex = createCellComplex(dim = 3,
                                        nodes = Nodes,
                                        edges = Edges,
                                        faces = Faces,
                                        faces_slip = faces_slip,
                                        polyhedra = Polyhedra,
                                        ip = 2022)
    
    del Nodes, Edges, Faces, faces_slip, Polyhedra
    
    # ax = create_ax_afonso()
    # cc.plot_polyhedra(ax = ax, alpha=0.2, edgecolor='purple')
    
    print("\nIt took {:.3f} s to create the cell complex.".format(time() - t0)) ; t0 = time()
    
    yieldstress = 10.09e9
    MAG = 10.09e9
    stress_tensor: np.ndarray = np.array([[0,0,0],[0,0,0],[0,0,1]])
    stress_tensor = stress_tensor * MAG / np.linalg.norm(stress_tensor)
    # stress2 = stress_cochain(cc, stress_tensor)
    
    # print("It took {:.3f} s to compute the stress 2-cochain.\n".format(time() - t0)) ; t0 = time()
    
    temperature: float = 293
    alpha: float | list[float] = [4.61e8 * yieldstress * 1e-6/complex_size[0]] # dimensions: FL**(-3)
    gamma: float | list[float] = [1] # dimensions: 0
    lambd: float | list[float] = [2.8 * alpha[0]] # dimensions: FL**(-3)
    
    test_run = MHalgorithm(choices = cc.faces_slip, steps = 10_000, iterations = 1, cellcomplex = cc)
    test_run.setup(temperature, alpha[0], gamma[0], lambd[0], stress = stress_tensor, starting_fraction = 0)
    
    """ ONE ITERATION """
    
    # print('Running simulation ... ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
    
    # one_iteration = test_run.run_one_iteration(progress_bar = True)
    
    # print("\nSimulation: complete! " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
    
    # if MAG/1e6 < 0.1:
    #     fname = 'sum_' +\
    #             str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) + '_' +\
    #             f'{one_iteration.alpha:1.0e}_{one_iteration.gamma:1.0e}_{one_iteration.lambd:1.0e}_' +\
    #             f'{int(MAG)}Pa.png'
    # elif MAG/1e6 >= 0.1:
    #     fname = 'sum_' +\
    #             str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) + '_' +\
    #             f'{one_iteration.alpha:1.0e}_{one_iteration.gamma:1.0e}_{one_iteration.lambd:1.0e}_' +\
    #             f'{MAG/1e6}MPa.png'
    # one_iteration.plot_one_iteration(MAG = MAG, save_figure = (True, fname))
    
    """ MULTIPLE ITERATIONS """
    
    # if MAG/1e6 < 0.1:
    #     fname = 'multisum_' +\
    #             str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) + '_' +\
    #             f'{test_run.alpha:1.2e}_{test_run.gamma:1.2e}_{test_run.lambd:1.2e}_' +\
    #             f'{int(MAG)}Pa'
    # elif MAG/1e6 >= 0.1:
    #     fname = 'multisum_' +\
    #             str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) + '_' +\
    #             f'{test_run.alpha:1.2e}_{test_run.gamma:1.2e}_{test_run.lambd:1.2e}_' +\
    #             f'{MAG/1e6}MPa'
    
    # multiple_iterations = test_run.run_multiple_iterations(iterations = 2)
        
    # counter = 0
    # for iteration in multiple_iterations:
    #     if MAG/1e6 < 0.1:
    #         fname = 'sum_' +\
    #                 str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) + '_' +\
    #                 f'{test_run.alpha:1.2e}_{test_run.gamma:1.2e}_{test_run.lambd:1.2e}_' +\
    #                 f'{int(MAG)}Pa_{counter}.png'
    #     elif MAG/1e6 >= 0.1:
    #         fname = 'sum_' +\
    #                 str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) + '_' +\
    #                 f'{test_run.alpha:1.2e}_{test_run.gamma:1.2e}_{test_run.lambd:1.2e}_' +\
    #                 f'{MAG/1e6}MPa_{counter}.png'
    #     iteration.plot_one_iteration(MAG = MAG, save_figure = (True, fname))
    #     counter += 1
    
    # print("\nIt took {:.3f} min to run the simulation.".format((time() - t0)/60)) ; del t0



""" ------------------------------------------------------------------------------------------- 

    # -- #
    def run_local_effects(self, progress_bar: bool = False):
        
        t0 = time()
        
        cellcomplex: CellComplex = self.cellcomplex
        Faces: list[Face3D] = cellcomplex.faces
        Faces_Slip: list[int] = cellcomplex.faces_slip
        Edges: list[Edge3D] = cellcomplex.edges
        Alpha: float = self.alpha
        Gamma: float = self.gamma
        Lambd: float = self.lambd
        Tau: float = self.tau
        Stress = self.stress
                
        boltzmann_constant: float = 1.380649e-23
        burgers: float = 2.556e-10 # (m) at 293 K (copper)
                
        S2S2_list: list[float] = []
        P2S2_list: list[float] = []
        T2S2_list: list[float] = []
        stress2S2_list: list[float] = []
        sys_energy_list: list[float] = [0]
        fraction_list: list[float] = []
        activated_systems: dict[int, list] = {}
        state_vector_dict: dict[int, dict] = {}
        delta_pmt: list[float] = []
        
        relaxation_energy: float = self.temperature * boltzmann_constant
        # relaxation: dict[int,float] = {p.id : temperature * boltzmann_constant for p in self.cellcomplex.polyhedra}
                
        # slip: list[int] = [0 for i in self.cellcomplex.faces] # to keep track of which faces have slipped (from the collection of all faces, not just slip faces)
        bins: dict[str,int] = {'A2':0,'A3':0,'A6':0,
                               'B2':0,'B4':0,'B5':0,
                               'C1':0,'C3':0,'C5':0,
                               'D1':0,'D4':0,'D6':0}
        
        slip_record: dict = defaultdict(int) # keeps track of which face-edge pairs have had their Ising state changed
        # for f in cellcomplex.faces:
        #     for e in f.e_ids:
        #         slip_record.update({(f.id,  e) : 0})
        #         slip_record.update({(f.id, -e) : 0})
        
        state_vector: dict = defaultdict(int) # keeps track of the current distribution of slipped face-edge pairs
        total_nr_pairs: int = len(Faces_Slip) * 3 * 2
        
        # P2: dict[int, np.ndarray] = {}
        P2: np.ndarray = np.empty((0,4))
        T2: Cochain = Cochain(cellcomplex, 2, np.array([[1,0,0,0]]))
        
        if self.starting_fraction != 0:
            
            initial_slips: list[int] = random.sample(Faces_Slip, k = int(self.starting_fraction * len(Faces_Slip)))
            initial_slips.sort()
            
            # Compute an initial P2 value and initial distribution of activated slip systems
            
            for face in initial_slips:
                
                edge: int = random.choice(Faces[face - 1].e_ids) # choose a slip direction at random
                directionality: int = random.choice([1,-1])
                slip_direction: np.ndarray = burgers * Faces[face - 1].measure**2 / Faces[face-1].support_volume * directionality * np.array(Edges[edge-1].unit_vector)
                # P2[face] = slip_direction
                P2 = np.vstack((P2, np.array([face, slip_direction[0], slip_direction[1], slip_direction[2]])))
                
                state_vector[str(face) + ' ' + str(directionality * edge)] = 1
                
                active_system: str = get_active_system(Faces[face - 1], Edges[edge - 1])
                bins[active_system] += 1
        
            T2 = T2 + P2.laplacian()
        
        state_vector_dict.update({0 : state_vector.copy()})
        activated_systems.update({0 : bins.copy()})
        nr_slips: int = np.count_nonzero(list(state_vector.values()))
        delta_pmt.append(Alpha * total_nr_pairs / (Lambd * nr_slips)) if nr_slips > 0 else delta_pmt.append(None)
        fraction_list.append(nr_slips / total_nr_pairs)
        
        # Start the algorithm!
        
        steps: int = self.steps
        step: int = 1
        if progress_bar:
            progress = tqdm(total = steps, desc = 'Running simulation', miniters = 10_000, mininterval = 1)
        
        cyclic_counter = 1
        interval = 2000
        
        while step <= steps:
                        
            # 1) Choose a random face-edge pair and invert the slip state of the pair (Ising model)
            
            face: int = random.choice(Faces_Slip) # choose a slip_face index at random (indices in 'choices' = cc.faces_slip start at 1)
            edge: int = random.choice(Faces[face - 1].e_ids) # choose a slip direction at random
            directionality: int = random.choice([1,-1])
            pair: str = str(face) + ' ' + str(directionality * edge)
            
            change_occurs: bool = False
                        
            if (state_vector[pair] == 1 and slip_record[pair] == 0): # If this system has already slipped, unslip it
                state_vector[pair] = 0
                event_type: int = -1
                change_occurs = True
            elif (state_vector[pair] == 0 and slip_record[pair] == 0): # If this system has not slipped, slip it
                state_vector[pair] = 1
                event_type: int = 1
                change_occurs = True
            
            if change_occurs:
                
                # 2) Compute cochains
                
                edge_unit_vector: np.ndarray = np.array(Edges[edge-1].unit_vector)
                slip_direction: np.ndarray = burgers * Faces[face-1].measure**2 / Faces[face-1].support_volume * directionality * edge_unit_vector
                S2: Cochain = Cochain(cellcomplex, 2, np.array([[face, *slip_direction]])) # dimensions: L**2
                
                # 3) Compute change in energy of the system
                
                # P2_on_cell: dict[int, np.ndarray] = P2.copy() # we will calculate the effect of the meanfield on the chosen 2-cell
                P2_on_cell: np.ndarray = np.copy(P2) # we will calculate the effect of the meanfield on the chosen 2-cell
                if face in P2_on_cell[:,0].astype(int):
                    # P2_on_cell[face] += slip_direction # add the new (un)slip to the meanfield
                    P2_on_cell[np.argwhere(P2 == face)[0,0]] += np.array([0, slip_direction[0], slip_direction[1], slip_direction[2]])
                else:
                    # P2_on_cell[face] = slip_direction
                    P2_on_cell = np.vstack((P2_on_cell, np.array([face, slip_direction[0], slip_direction[1], slip_direction[2]])))
                P2_on_cell: Cochain = Cochain(cellcomplex, 2, {face : meanfield(vector = edge_unit_vector, meanfield_array = P2_on_cell, N = total_nr_pairs)}) # compute the resulting meanfield effect on the chosen 2-cell, considering the orientation of the slip
                S2S2: float = S2.inner_product_2022() # dimensions: L**3
                P2S2: float = P2_on_cell.inner_product_2022(S2) # dimensions: L**3
                
                S2laplacian: Cochain = S2.laplacian()
                T2_on_cell =  T2 + S2laplacian
                T2S2: float = T2_on_cell.inner_product_2022(S2)
                
                stress2S2: float = np.dot(slip_direction, Stress @ np.array(Faces[face-1].normal) * Faces[face-1].measure) * Cochain(cellcomplex, 2, {face : 1}).inner_product_2022()
                energy_diff = event_type * (Alpha * S2S2 - Gamma * stress2S2 - Lambd * P2S2 - Tau * T2S2)
                                                
                # 4) Acceptance criteria
                
                if energy_diff < 0:
                    change_occurs: bool = True
                
                else:
                    prob = np.exp(- energy_diff / relaxation_energy)
                    p_acc = - np.random.uniform(-1,0) # use this because np.random.random() does not include 1, while this function, as per the python documentation, includes -1.
                    
                    if prob < p_acc: # Whatever change we tried to make does not take effect, so we need to revert the slip state of the pair to its original value
                        change_occurs: bool = False
                        state_vector[pair] = 1 if state_vector[pair] == 0 else 0 # undo the change in part 1)
                        
                    else:
                        change_occurs: bool = True
                
                # 5) Take note of the changes to the system configuration if slip occurs
                
                if change_occurs:
                    
                    sys_energy_list.append(sys_energy_list[-1] + energy_diff) # the (un)slip event occurs and the system energy changes
                    
                    T2 = T2 + event_type * S2laplacian
                    
                    # save changes to the mean-field
                    if event_type == -1: # means that a face is being unslipped, so we need to remove it from the meanfield
                        P2[np.argwhere(P2 == face)[0,0]] -= np.array([0, slip_direction[0], slip_direction[1], slip_direction[2]])
                        # if np.isclose(0, np.linalg.norm(P2[face]), rtol = 1e-5, atol = 0.0): # will only trigger if the difference is more than 0.001% of the value
                        #     del P2[face] # need to remove it so it doesn't affect the mean when calulating the mean-field effect
                    elif event_type == 1: # means that a face is being slipped, so we need to add it to the meanfield
                        if face in P2[:,0]:
                            # P2[face] += slip_direction
                            P2[np.argwhere(P2 == face)[0,0]] += np.array([0, slip_direction[0], slip_direction[1], slip_direction[2]])
                            # if np.isclose(0, np.linalg.norm(P2[face]), rtol = 1e-5, atol = 0.0): # will only trigger if the difference is more than 0.001% of the value
                            #     del P2[face] # need to remove it so it doesn't affect the mean when calulating the mean-field effect
                        else:
                            # P2[face] = slip_direction
                            P2 = np.vstack((P2, np.array([face, slip_direction[0], slip_direction[1], slip_direction[2]])))
                    
                    # Record face-edge pair as having (un)slipped, and must also account for topological and crystallographic constraints
                    
                    slip_record[pair] += 1
                    
                    # If we are adding a slip event, then we prevent the opposite slip event from occurring by changing its record to 1.
                    
                    if event_type == 1: # adding a slip event
                        # slip_record[(face, - directionality * edge)] += 1
                        nr_slips += 1
                        delta_pmt.append(Alpha * total_nr_pairs / (Lambd * nr_slips))
                    elif event_type == -1: # removing a slip event
                        nr_slips -= 1
                        delta_pmt.append(Alpha * total_nr_pairs / (Lambd * nr_slips)) if nr_slips != 0 else None
                    
                    fraction_list.append(nr_slips / total_nr_pairs)
                    
                    active_system: str = get_active_system(cellcomplex.get_one('f', face), cellcomplex.get_one('e', edge))
                    bins[active_system] += event_type # This will be 1 if we are trying to slip a pair and -1 if we are trying to unslip a pair
                    
                    # To save a little bit of RAM, we only record the state_vector and the activated_systems at certain intervals during the simulation.
                    if cyclic_counter % interval == 0:
                        cyclic_counter = 1
                    elif (cyclic_counter % interval == interval - 1 or step in list(range(int(0.99*steps), steps + 1))):
                        state_vector_dict.update({step : state_vector.copy()})
                        activated_systems.update({step : bins.copy()})
                        cyclic_counter += 1
                    else:
                        cyclic_counter += 1
                    
                    S2S2_list.append(S2S2)
                    P2S2_list.append(P2S2)
                    T2S2_list.append(T2S2)
                    stress2S2_list.append(stress2S2)
                    
                elif not change_occurs: # here the trial change in the Ising model fails
                    
                    S2S2_list.append(None)
                    P2S2_list.append(P2S2)
                    T2S2_list.append(T2S2)
                    stress2S2_list.append(None)
                    delta_pmt.append(delta_pmt[-1])
                    fraction_list.append(fraction_list[-1])
                    sys_energy_list.append(sys_energy_list[-1])
                
            elif not change_occurs: # here the face chosen at random in 1) has already been flipped, so no change whatsoever occurs
                
                S2S2_list.append(None)
                P2S2_list.append(None)
                T2S2_list.append(None)
                stress2S2_list.append(None)
                delta_pmt.append(delta_pmt[-1])
                fraction_list.append(fraction_list[-1])
                sys_energy_list.append(sys_energy_list[-1])
                
            step += 1
            if progress_bar:
                progress.update()
        
        # print("\nIt took {:.3f} min to run one iteration of the MH algorithm.".format((time() - t0)/60), flush = True)
        
        if steps not in activated_systems.keys():
            activated_systems.update({steps : bins.copy()})
        if steps not in state_vector_dict.keys():
            state_vector_dict.update({steps : state_vector.copy()})
            
        self.S2S2: list[float] = S2S2_list
        self.P2S2: list[float] = P2S2_list
        self.T2S2: list[float] = T2S2_list
        self.stress2S2: list[float] = stress2S2_list
        self.delta: list[float] = delta_pmt
        self.fraction: list[float] = fraction_list
        self.sys_energy: list[float] = sys_energy_list
        self.state_vector: dict[int, dict] = state_vector_dict
        self.activated_systems: dict[int, dict] = activated_systems
        self.P2_final: dict[int, np.ndarray] = array_to_dict(P2)
        self.time: float = time() - t0
        return self

"""
