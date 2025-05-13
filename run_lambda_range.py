#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:20 2024
Last edited on: Feb 27 16:30 2025

Author: Afonso Barroso, 9986055, The University of Manchester

"""

# ----- # ----- #  IMPORTS # ----- # ----- #

import numpy as np
import argparse
import sys, os, shutil
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from functools import partial
import multiprocessing as mp
from datetime import datetime

sys.path.append('./') ; sys.path.append('../') ; sys.path.append('../Voronoi_PCC_Analyser/')
from matgen.base import CellComplex
from dccstructure.iofiles import import_complex_data
from cellcomplex import createAllCells, scaleAllCells, createCellComplex
from metrohast_sum import MHalgorithm, orderOfMagnitude, get_superscript

""" ---------------------------------------------------------------------------------------------------------------- """

def worker(item: tuple, run, nr_processes: int | None = None) -> tuple:
    
    iteration = MHalgorithm(run.choices, run.steps, 1, run.cellcomplex)
    iteration.setup(temperature = run.temperature,
                    alpha = run.alpha,
                    gamma = run.gamma,
                    lambd = item[0],
                    stress = run.stress,
                    starting_fraction = run.starting_fraction)
    
    if isinstance(nr_processes, int):
        iteration = iteration.run_one_iteration(progress_bar=False, get_acceptance_rate=False)
    else:
        iteration = iteration.run_one_iteration(progress_bar=True, get_acceptance_rate=False)
        
    if len(iteration.P2_final) == 0:
        meanfield_value = np.array([0,0,0])
    else:
        meanfield_value = np.sum(np.array([np.array(p) for p in iteration.P2_final.values()]), axis=0)
    
    return iteration.fraction[-1], iteration.sys_energy[-1], meanfield_value #, iteration.activated_systems[iteration.steps]

def run_lambda_range(run, lambda_range: list | tuple | np.ndarray, nr_processes: int | None = None) -> tuple:
    
    fraction: list = []
    sysenergy: list = []
    meanfield: list = []
        
    function = partial(worker, run = run, nr_processes = nr_processes)
    items: list = []
    for lambda_value in lambda_range:
        for i in range(run.iterations):
            items.append([lambda_value * run.alpha, i])
    
    if isinstance(nr_processes, int):
        with mp.Pool(processes = nr_processes) as pool:
            for result in pool.map(function, items):
                fraction.append(result[0])
                sysenergy.append(result[1])
                meanfield.append(result[2])
    else:
        for item in items:
            result = function(item)
            fraction.append(result[0])
            sysenergy.append(result[1])
            meanfield.append(result[2])
    
    iters: int = run.iterations
    
    data_f = np.empty((0,4))
    data_e = np.empty((0,4))
    data_m = np.empty((0,5))

    for i, lambd in enumerate(lambda_range):
        f_values, f_counts = np.unique(np.array([fraction[i * iters : (i + 1) * iters]]).round(decimals=9), return_counts=True)
        e_values, e_counts = np.unique(np.array([sysenergy[i * iters : (i + 1) * iters]]).round(decimals=15), return_counts=True)
        
        data_f = np.vstack((data_f, np.array([[i, lambd, f_values[k], f_counts[k]] for k in range(len(f_values))])))
        data_e = np.vstack((data_e, np.array([[i, lambd, e_values[k], e_counts[k]] for k in range(len(e_values))])))
        
        meanfieldvalues_forthisstress = np.array(meanfield[i * iters : (i + 1) * iters])
        for row in meanfieldvalues_forthisstress:
            data_m = np.vstack((data_m, np.array([[i, lambd, *row]])))
    
    return data_f, data_e, data_m


def weighted_mean(measurements: list[tuple]) -> tuple[float]:
    numerator = sum([measurements[i][0] / measurements[i][1]**2 for i in range(len(measurements))])
    denominator = sum([1 / measurements[i][1]**2 for i in range(len(measurements))])
    return (numerator / denominator, 1 / denominator)


def plot_Fraction(data, name):
    
    data = np.copy(data)
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    # x axis
    ax.set_xlim([min(data[:,1]), max(data[:,1])])
    ax.xaxis.set_major_locator(1)
    ax.xaxis.set_minor_locator(0.5)
    ax.set_xlabel('$\\lambda / \\alpha$', size = 20, fontname = 'monospace', labelpad=10)
    # y axis
    ax.set_ylim([0, 0.8])
    ax.set_ylabel('Fraction of slipped 2-cells', size = 20, fontname = 'monospace', labelpad=10)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    # tick parametres
    ax.tick_params(axis = 'both', which = 'major', length = 5, labelfontfamily = 'monospace', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
    # plot
    ax.scatter(data[:,1], data[:,2], linewidth = 7, color = 'k')
    # ax.fill_between(x, y - yerr, y + yerr, color = 'grey', alpha = 0.2)
    # adjust
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
    # save figure
    plt.savefig(name, dpi = 1000)


def plot_Energy(data, name):
    
    data = np.copy(data)
    scaling = orderOfMagnitude(np.max(abs(data[:,2])))
    data[:,2] *= 10**(-scaling)
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    # x axis
    ax.set_xlim([min(data[:,1]), max(data[:,1])])
    ax.xaxis.set_major_locator(1)
    ax.xaxis.set_minor_locator(0.5)
    ax.set_xlabel('$\\lambda / \\alpha$', size = 20, fontname = 'monospace', labelpad=10)
    # y axis
    try:
        ax.yaxis.set_major_locator(MultipleLocator(1 * 10 ** (orderOfMagnitude(max([abs(i) for i in data[:,2] if i is not None])))))
        ax.yaxis.set_minor_locator(MultipleLocator(1 * 10 ** (orderOfMagnitude(max([abs(i) for i in data[:,2] if i is not None])) - 1)))
    except: pass
    if scaling != 0:
        ax.set_ylabel('Cumulative dissipated energy ($\\times$10' + get_superscript(str(scaling)) + ' J)', size = 20, fontname = 'monospace', labelpad = 10) # ×
    else:
        ax.set_ylabel('Cumulative dissipated energy (J)', size = 20, fontname = 'monospace', labelpad = 10) # ×
    # tick parametres
    ax.tick_params(axis = 'both', which = 'major', length = 5, labelfontfamily = 'monospace', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
    # plot
    ax.scatter(data[:,1], -data[:,2], linewidth = 7, color = 'k')
    # ax.fill_between(x, y - yerr, y + yerr, color = 'grey', alpha = 0.2)
    ax.set_ylim(bottom = 0)
    # adjust
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
    # save figure
    plt.savefig(name, dpi = 1000)


def write_to_file(fname: str, lst: list):
    
    with open(f'{fname}.txt', 'w') as f:
        for x in lst:
            f.write(f"{x}\n")


def main(id: str,
         ccdata: str,
         alpha: float,
         gamma: float,
         lambda_range: list,
         nr_points: float,
         stress_magnitude: float,
         stress_tensor: list,
         steps: int | float,
         iterations: int,
         side_length: float,
         temperature: float,
         starting_fraction: float,
         processes: int,
         plotp: str | list[str]):
        
    t0 = time()
    
    """ 1. Create cell complex """
    
    data_folder = Path(ccdata)
    text = ccdata.split('x')
    complex_size = [int(text[-3][7:]), int(text[-2]), int(text[-1].split('/')[0])] # the number of computational unit cells in each 3D dimension
    nodes, edges, faces, faces_slip, faces_areas, faces_normals, volumes, volumes_vols, nr_cells, A0, A1, A2, A3, B1, B2, B3 = import_complex_data(data_folder)
    edges += 1 ; faces += 1 ; volumes += 1  ;  edges = edges.astype(int)
    faces_slip = [x + 1 for x in faces_slip]
    A0[:,0:2] += 1 ; A1[:,0:2] += 1 ; A2[:,0:2] += 1 ; A3[:,0:2] += 1 ; B1[:,0:2] += 1 ; B2[:,0:2] += 1 ; B3[:,0:2] += 1

    Nodes, Edges, Faces, Polyhedra = createAllCells((nodes + 0.5),
                                                    edges,
                                                    faces,
                                                    volumes,
                                                    faces_normals,
                                                    A0, A1, A2, A3, B1, B2, B3,
                                                    faces_areas,
                                                    volumes_vols)
    
    del A0, A1, A2, A3, B1, B2, B3
    del nodes, edges, faces, faces_areas, faces_normals, volumes, volumes_vols
    
    Nodes, Edges, Faces, Polyhedra = scaleAllCells(Nodes, Edges, Faces, Polyhedra, scaling = float(side_length / complex_size[0]))

    cc: CellComplex = createCellComplex(dim = 3,
                                        nodes = Nodes,
                                        edges = Edges,
                                        faces = Faces,
                                        faces_slip = faces_slip,
                                        polyhedra = Polyhedra,
                                        ip = 2022)

    del Nodes, Edges, Faces, faces_slip, Polyhedra

    print("\nIt took {:.3f} s to create the cell complex.".format(time() - t0)) ; t0 = time()
    
    """ 2. Compute stress 2-cochain """
    
    stress_tensor: np.ndarray = np.array(stress_tensor).reshape((3,3))
    stress_tensor = stress_tensor / np.linalg.norm(stress_tensor) * stress_magnitude
    # stress2 = stress_cochain(cc, stress_tensor)
    
    # print("It took {:.3f} s to compute the stress 2-cochain.\n".format(time() - t0)) ; t0 = time()
    
    """ 3. Run the MH algorithm """
    
    print('\nRunning simulation ... ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
    
    test_run = MHalgorithm(choices = cc.faces_slip, steps = int(steps), iterations = iterations, cellcomplex = cc)
    test_run.setup(temperature = temperature,
                   alpha = alpha,
                   gamma = gamma,
                   lambd = None,
                   tau = None,
                   stress = stress_tensor,
                   starting_fraction = starting_fraction)

    lambda_range = np.linspace(lambda_range[0], lambda_range[1], nr_points) ; t0 = time()
            
    data_f, data_e, data_m = run_lambda_range(test_run, lambda_range, nr_processes = processes)
    
    """ 4. Plot graphs and save data """
    
    newdir: str = rf'./{id}_lambd_range_out'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    else:
        shutil.rmtree(newdir) # Removes all the subdirectories!
        os.makedirs(newdir)
    
    fname = newdir + '/' + str(id)
    
    try:
        plot_Fraction(data_f, fname + '_fraction.png')
    except Exception as e:
        print('\nFailed to plot Fraction vs. Lambda:', e)
    try:
        plot_Energy(data_e, fname + '_energy.png')
    except Exception as e:
        print('\nFailed to plot Energy vs. Lambda:', e)
    except Exception as e:
        print('\nFailed to plot Strain vs. Lambda:', e)
    
    np.savetxt(f'{id}_lambd_range_out/data_fraction.txt', data_f, header = 'number / lambda / unique value / counts')
    np.savetxt(f'{id}_lambd_range_out/data_sysenergy.txt', data_e, header = 'number / lambda / unique value / counts')
    np.savetxt(f'{id}_lambd_range_out/data_meanfield.txt', data_m, header = 'number / lambda / meanfield components')
    np.savetxt(f'{id}_lambd_range_out/lambda_values.txt', lambda_range)
    
    print("\nSimulation: complete! " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n') ; del t0
    
    
""" ---------------------------------------------------------------------------------------------------------------- """


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Some description')

    parser.add_argument(
        '-id',
        action = 'store',
        type = str,
        required = False,
        default = '',
        help = 'Unique ID to label the outputs.')

    parser.add_argument(
        '--ccdata',
        action = 'store',
        type = str,
        required = True,
        help = 'Path to the directory where the cell complex data files are.')

    parser.add_argument(
        '--alpha', '-a',
        action = 'store',
        type = float,
        required = True,
        help = 'The self-energy α of the microslips (Pa)')

    parser.add_argument(
        '--gamma', '-g',
        action = 'store',
        type = float,
        required = True,
        help = 'The strength γ of the coupling with the externally applied stress tensor.')

    parser.add_argument(
        '--range', '-r',
        action = 'store',
        nargs = 2,
        type = float,
        required = False,
        default = [0, 15],
        help = 'The range of magnitudes of the mean-field coupling strength λ in units of the self-energy α. The default is from 0 to 15.')

    parser.add_argument(
        '--nrpoints', '-n',
        action = 'store',
        type = int,
        required = False,
        default = 150,
        help = 'The number of data points desired. The default is 150.')

    parser.add_argument(
        '--magnitude', '-m',
        action = 'store',
        type = float,
        required = True,
        help = 'The magnitude of the externally applied stress (Pa).')

    parser.add_argument(
        '--stress',
        action = 'store',
        type = int,
        nargs = 9,
        required = True,
        help = 'The entries of the stress tensor matrix in the order xx xy xz yx yy yz zx zy zz as integers of relative strength (the stress tensor is normalised in the code).')

    parser.add_argument(
        '--steps', '-s',
        action = 'store',
        type = float,
        required = False,
        default = 500_000,
        help = 'The number of steps to run the simulation for. The default is 500,000.')

    parser.add_argument(
        '--iterations', '-i',
        action = 'store',
        type = int,
        required = False,
        default = 10,
        help = 'The number of iterations of the algorithm to run for each data point (and to take the average and standard deviation of). The default is 10.')

    parser.add_argument(
        '-d',
        action = 'store',
        type = float,
        required = False,
        default = 1e-6,
        help = 'Length of the edge of the (cubic) simulation space. The default is 1 micrometre')

    parser.add_argument(
        '--temp', '-t',
        action = 'store',
        type = float,
        required = False,
        default = 293,
        help = 'Temperature of the system. The default is 293 K.')

    parser.add_argument(
        '--fraction', '-f',
        action = 'store',
        type = float,
        required = False,
        default = 0,
        help = 'The starting fraction of slipped 2-cells. The default is 0.')

    parser.add_argument(
        '--processes',
        action = 'store',
        type = int,
        required = False,
        default = None,
        help = 'Number of processes to run the algorithm with. The default is None, which will make the script run in series.')

    parser.add_argument(
        '--plotp',
        action = 'store',
        type = str,
        required = False,
        default = None,
        nargs = '*',
        choices = ['x', 'X', 'y', 'Y', 'z', 'Z'],
        help = 'Which components of the mean-field term to plot against the magnitude of the applied stress. The default is None.')
    
    args = parser.parse_args()
    
    main(id = args.id,
         ccdata = args.ccdata,
         alpha = args.alpha,
         gamma = args.gamma,
         lambda_range = args.range,
         nr_points = args.nrpoints,
         stress_magnitude = args.magnitude,
         stress_tensor = args.stress,
         steps = args.steps,
         iterations = args.iterations,
         side_length = args.d,
         temperature = args.temp,
         starting_fraction = args.fraction,
         processes = args.processes,
         plotp = args.plotp)

    


