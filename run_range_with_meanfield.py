#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:15 2024
Last edited on: Feb 12 11:33 2025

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
                    lambd = run.lambd,
                    stress = item[0] * run.stress,
                    starting_fraction = run.starting_fraction)
    
    if isinstance(nr_processes, int):
        iteration = iteration.run_one_iteration(progress_bar=False, get_acceptance_rate=False)
    else:
        iteration = iteration.run_one_iteration(progress_bar=True, get_acceptance_rate=False)
        
    if len(iteration.P2_final) == 0:
        meanfield_value = np.array([0,0,0])
    else:
        meanfield_value = np.sum(np.array([np.array(p) for p in iteration.P2_final.values()]), axis=0)
    
    return iteration.fraction[-1], iteration.sys_energy[-1], iteration.delta[-1], iteration.strain[-1], meanfield_value #, iteration.activated_systems[iteration.steps]

def run_stress_range(run, stress_range: list | tuple | np.ndarray, nr_processes: int | None = None) -> tuple:
    
    fraction: list = []
    sysenergy: list = []
    delta: list = []
    strain: list = []
    meanfield: list = []
        
    function = partial(worker, run = run, nr_processes = nr_processes)
    items: list = []
    for stress_value in stress_range:
        for i in range(run.iterations):
            items.append([stress_value, i])
    
    if isinstance(nr_processes, int):
        with mp.Pool(processes = nr_processes) as pool:
            for result in pool.map(function, items):
                fraction.append(result[0])
                sysenergy.append(result[1])
                delta.append(result[2])
                strain.append(result[3])
                meanfield.append(result[4])
    else:
        for item in items:
            result = function(item)
            fraction.append(result[0])
            sysenergy.append(result[1])
            delta.append(result[2])
            strain.append(result[3])
            meanfield.append(result[4])
    
    iters: int = run.iterations
    
    data_f = np.empty((0,4))
    data_e = np.empty((0,4))
    data_d = np.empty((0,4))
    data_s = np.empty((0,4))
    data_m = np.empty((0,5))

    for i, stress in enumerate(stress_range):
        f_values, f_counts = np.unique(np.array([fraction[i * iters : (i + 1) * iters]]).round(decimals=9), return_counts=True)
        e_values, e_counts = np.unique(np.array([sysenergy[i * iters : (i + 1) * iters]]).round(decimals=15), return_counts=True)
        s_values, s_counts = np.unique(np.array([strain[i * iters : (i + 1) * iters]]).round(decimals=4), return_counts=True)
        
        d: list = [j for j in delta[i * iters : (i + 1) * iters] if j is not None]
        if len(d) == 0:
            d_values = np.array([None])
            d_counts = np.array([0])
        else:
            d_values, d_counts = np.unique(np.array(d).round(decimals=5), return_counts=True)

        data_f = np.vstack((data_f, np.array([[i, stress, f_values[k], f_counts[k]] for k in range(len(f_values))])))
        data_e = np.vstack((data_e, np.array([[i, stress, e_values[k], e_counts[k]] for k in range(len(e_values))])))
        data_d = np.vstack((data_d, np.array([[i, stress, d_values[k], d_counts[k]] for k in range(len(d_values))])))
        data_s = np.vstack((data_s, np.array([[i, stress, s_values[k], s_counts[k]] for k in range(len(s_values))])))
        
        meanfieldvalues_forthisstress = np.array(meanfield[i * iters : (i + 1) * iters])
        for row in meanfieldvalues_forthisstress:
            data_m = np.vstack((data_m, np.array([[i, stress, *row]])))
    
    return data_f, data_e, data_d, data_s, data_m


def weighted_mean(measurements: list[tuple]) -> tuple[float]:
    numerator = sum([measurements[i][0] / measurements[i][1]**2 for i in range(len(measurements))])
    denominator = sum([1 / measurements[i][1]**2 for i in range(len(measurements))])
    return (numerator / denominator, 1 / denominator)


def plot_stress_range(run,
                      x_range: list | np.ndarray,
                      fraction_avg: list,
                      fraction_std: list,
                      sysenergy_avg: list,
                      sysenergy_std: list,
                      delta_avg: list,
                      delta_std: list,
                      complex_size: list,
                      save_figure: tuple = (False, None)):
    
    fraction_avg: np.ndarray = np.array(fraction_avg)
    fraction_std: np.ndarray = np.array(fraction_std)
    sysenergy_avg: np.ndarray = np.array(sysenergy_avg)
    sysenergy_std: np.ndarray = np.array(sysenergy_std)
    delta_avg: np.ndarray = np.array(delta_avg)
    delta_std: np.ndarray = np.array(delta_std)
    
    fig = plt.figure(figsize = (15, 25))
    fig.suptitle(str(complex_size[0]) + 'x' + str(complex_size[1]) + 'x' + str(complex_size[2]) +\
                 f' α = {run.alpha:1.0e} Pa, γ = {run.gamma:1.0e}, λ = {run.lambd:1.0e} Pa' +\
                 ', 1 μm$^{3}$',
                 fontsize = 20, y = 0.92)
    
    # eV = 1.602176634e-19 # (J)
    eV = 1e-9
    energy_units = 'nJ'
    
    # FRACTION
    ax = fig.add_subplot(311)
    ax.set_xlim([x_range[0]/1e6, x_range[-1]/1e6])
    ax.set_ylim([0, 0.8])
    ax.set_xlabel('Stress (MPa)', size = 15)
    ax.set_ylabel('Fraction of slipped 2-cells', size = 15)
    ax.plot(x_range/1e6, fraction_avg, c = 'b', linewidth = 5)
    ax.fill_between(x_range/1e6, fraction_avg - fraction_std, fraction_avg + fraction_std, color = 'b', alpha = 0.2)
    
    # ENERGY
    ax = fig.add_subplot(312)
    ax.set_xlim([x_range[0]/1e6, x_range[-1]/1e6])
    try:
        ax.set_ylim([0, - 1.1 * min(sysenergy_avg)/eV])
    except:
        pass
    ax.set_xlabel('Stress (MPa)', size = 15)
    ax.set_ylabel(f'Cumulative energy difference ({energy_units})', size = 15)
    ax.plot(x_range/1e6, sysenergy_avg/eV, c = 'r', linewidth = 5)
    try:
        ax.fill_between(x_range/1e6, (sysenergy_avg - sysenergy_std)/eV, (sysenergy_avg + sysenergy_std)/eV, color = 'r', alpha = 0.2)        
    except:
        pass
    
    # DELTA
    ax = fig.add_subplot(313)
    ax.set_xlim([x_range[0]/1e6, x_range[-1]/1e6])
    # ax.set_ylim([0, 1.1 * min(delta_avg)])
    ax.set_xlabel('Stress (MPa)', size = 15)
    ax.set_ylabel('δ', size = 15)
    ax.set_yscale('log')
    ax.plot(x_range/1e6, delta_avg, c = 'k', linewidth = 5)
    try:
        ax.fill_between(x_range/1e6, delta_avg - delta_std, delta_avg + delta_std, color = 'k', alpha = 0.2)
    except:
        pass
    
    if save_figure[0] == True:
        plt.savefig(save_figure[1], dpi = 500)


# def plot_Acceptancerate(x, y, yerr, name):
    
#     x = 1e-6 * x
            
#     fig = plt.figure(figsize = (11, 11))
#     ax = fig.add_subplot()
#     # x axis
#     ax.set_xlim([min(x), max(x)])
#     ax.xaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(max(x)) - 1)))
#     ax.xaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(max(x)) - 2)))
#     ax.set_xlabel('Applied stress magnitude (MPa)', size = 20, fontname = 'monospace', labelpad=10)
#     # y axis
#     ax.set_ylabel('Acceptance rate', size = 20, fontname = 'monospace', labelpad=10)
#     #ax.yaxis.set_minor_locator(MultipleLocator(0.01))
#     # tick parametres
#     ax.tick_params(axis = 'both', which = 'major', length = 5, labelfontfamily = 'monospace', labelsize = 15)
#     ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
#     # plot
#     ax.plot(x, y, c = 'black', linewidth = 5)
#     ax.fill_between(x, y - yerr, y + yerr, color = 'grey', alpha = 0.2)
#     # adjust
#     plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
#     # save figure
#     plt.savefig(name, dpi = 1000)


def plot_Fraction(data, name):
    
    data = np.copy(data)
    data[:,1] = 1e-6 * data[:,1]
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    # x axis
    ax.set_xlim([min(data[:,1]), max(data[:,1])])
    ax.xaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 2)))
    ax.xaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 3)))
    ax.set_xlabel('Applied stress magnitude (MPa)', size = 20, fontname = 'monospace', labelpad=10)
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
    data[:,1] = 1e-6 * data[:,1]
    scaling = orderOfMagnitude(np.max(abs(data[:,2])))
    data[:,2] *= 10**(-scaling)
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    # x axis
    ax.set_xlim([min(data[:,1]), max(data[:,1])])
    ax.xaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 2)))
    ax.xaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 3)))
    ax.set_xlabel('Applied stress magnitude (MPa)', size = 20, fontname = 'monospace', labelpad=10)
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


def plot_Delta(data, name):
    
    data = np.copy(data)
    data[:,1] = 1e-6 * data[:,1]
    scaling = orderOfMagnitude(np.max(abs(data[:,2])))
    data[:,2] *= 10**(-scaling)
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    # x axis
    ax.set_xlim([min(data[:,1]), max(data[:,1])])
    ax.xaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 1)))
    ax.xaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 2)))
    ax.set_xlabel('Applied stress magnitude (MPa)', size = 20, fontname = 'monospace', labelpad=10)
    # y axis
    ax.set_ylabel('$\\delta$', size = 20, fontname = 'monospace', labelpad=10)
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


def plot_Strain(data, name):
    
    data = np.copy(data)
    data[:,1] = 1e-6 * data[:,1]
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    # x axis
    ax.set_xlim([min(data[:,1]), max(data[:,1])])
    ax.xaxis.set_major_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 1)))
    ax.xaxis.set_minor_locator(MultipleLocator(10 ** (orderOfMagnitude(max(data[:,1])) - 2)))
    ax.set_xlabel('Applied stress magnitude (MPa)', size = 20, fontname = 'monospace', labelpad=10)
    # y axis
    ax.set_ylabel('Strain', size = 20, fontname = 'monospace', labelpad=10)
    # ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    # tick parametres
    ax.tick_params(axis = 'both', which = 'major', length = 5, labelfontfamily = 'monospace', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
    # plot
    ax.scatter(data[:,1], data[:,2], linewidth = 7, color = 'k')
    # ax.fill_between(x, y - yerr, y + yerr, color = 'grey', alpha = 0.2)
    ax.set_ylim(bottom=0)
    # adjust
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
    # save figure
    plt.savefig(name, dpi = 1000)


def plot_PnormvsStress(x, y, yerr, name):
    
    x = 1e-6 * x
    scaling = orderOfMagnitude(np.max(abs(y)))
    y = y * 10**(-scaling)
    yerr = yerr * 10**(-scaling)
            
    fig = plt.figure(figsize = (11, 11))
    ax = fig.add_subplot()
    ax.errorbar(x/1e6, y, yerr, c = 'black', marker = 'o', linestyle = None, capsize = 4)
    ax.set_xlim([min(x)/1e6, max(x)/1e6])
    ax.set_xlabel('Stress (MPa)', size = 20, fontname = 'monospace', labelpad=10)
    ax.set_ylim([0,1.1*max(y)])
    ax.set_ylabel('$|\\vec{P}|$', size = 20, fontname = 'monospace', labelpad=10)
    ax.tick_params(axis = 'both', which = 'major', length = 5, labelfontfamily = 'monospace', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
    plt.savefig(name, dpi = 1000)
    
    
def plot_PvsStress(plotp, x, y, yerr, name):
    
    if isinstance(plotp, str):
        plotp = [plotp]
    if isinstance(plotp, list):
        for p in plotp:
            match p:
                case 'x' | 'X':
                    s = 0
                case 'y' | 'Y':
                    s = 1
                case 'z' | 'Z':
                    s = 2
            fig = plt.figure(figsize = (11,11))
            ax = fig.add_subplot()
            ax.errorbar(x/1e6, y[:,s], yerr[:,s], c = 'black', marker = 'o', linestyle = None, capsize = 4)
            ax.set_xlim([min(x)/1e6, max(x)/1e6])
            ax.set_xlabel('Stress (MPa)', size = 20, fontname = 'monospace', labelpad=10)
            ax.set_ylim([1.1 * min(y[:,s]), 1.1 * max(y[:,s])])
            ax.set_ylabel(f'$P_{p}$', size = 20, fontname = 'monospace', labelpad=10)
            ax.tick_params(axis = 'both', which = 'major', length = 5, labelfontfamily = 'monospace', labelsize = 15)
            ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
            plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0, hspace=0)
            plt.savefig(name + '_' + p + '.png', dpi = 1000)
            
    return None


def write_to_file(fname: str, lst: list):
    
    with open(f'{fname}.txt', 'w') as f:
        for x in lst:
            f.write(f"{x}\n")


def main(id: str,
         ccdata: str,
         alpha: float,
         gamma: float,
         lambd: float,
         stress_range: list,
         nr_points: float,
         stress: list,
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
    
    stress_tensor: np.ndarray = np.array(stress).reshape((3,3))
    stress_tensor = stress_tensor / np.linalg.norm(stress_tensor)
    # stress2 = stress_cochain(cc, stress_tensor)
    
    # print("It took {:.3f} s to compute the stress 2-cochain.\n".format(time() - t0)) ; t0 = time()
    
    """ 3. Run the MH algorithm """
    
    print('\nRunning simulation ... ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
    
    test_run = MHalgorithm(choices = cc.faces_slip, steps = int(steps), iterations = iterations, cellcomplex = cc)
    test_run.setup(temperature = temperature,
                   alpha = alpha,
                   gamma = gamma,
                   lambd = lambd,
                   tau = None,
                   stress = stress_tensor,
                   starting_fraction = starting_fraction)

    stress_range = np.linspace(stress_range[0], stress_range[1], nr_points) ; t0 = time()
            
    data_f, data_e, data_d, data_s, data_m = run_stress_range(test_run, stress_range, nr_processes = processes)
    
    """ 4. Plot graphs and save data """
    
    newdir: str = rf'./{id}_stress_range_out'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    else:
        shutil.rmtree(newdir) # Removes all the subdirectories!
        os.makedirs(newdir)
    
    fname = newdir + '/' + str(id)
    
    # try:
    #     plot_Acceptancerate(stress_range, np.array(acceptance_rate_avg), np.array(acceptance_rate_std), fname + '_acceptancerate.pdf')
    # except Exception as e:
    #     print('\nFailed to plot Fraction vs. Stress:', e)
    try:
        plot_Fraction(data_f, fname + '_fraction.png')
    except Exception as e:
        print('\nFailed to plot Fraction vs. Stress:', e)
    try:
        plot_Energy(data_e, fname + '_energy.png')
    except Exception as e:
        print('\nFailed to plot Energy vs. Stress:', e)
    try:
        plot_Delta(data_d, fname + '_delta.png')
    except Exception as e:
        print('\nFailed to plot Delta vs. Stress:', e)
    try:
        plot_Strain(data_s, fname + '_strain.png')
    except Exception as e:
        print('\nFailed to plot Strain vs. Stress:', e)
    # try:
    #     plot_PnormvsStress(stress_range, np.linalg.norm(np.array(meanfield_avg), axis = 1), np.linalg.norm(np.array(meanfield_std), axis = 1), name = newdir + '/PnormvsStress.png')
    # except Exception as e:
    #     print('\nFailed to plot P norm vs. Stress:', e)
    # try:
    #     _ = plot_PvsStress(plotp, stress_range, np.array(meanfield_avg), np.array(meanfield_std), name = newdir + '/PvsStress')
    # except Exception as e:
    #     print('\nFailed to plot P components vs. Stress:', e)
    
    np.savetxt(f'{id}_stress_range_out/data_fraction.txt', data_f, header = 'number / stress / unique value / counts')
    np.savetxt(f'{id}_stress_range_out/data_sysenergy.txt', data_e, header = 'number / stress / unique value / counts')
    np.savetxt(f'{id}_stress_range_out/data_delta.txt', data_d, header = 'number / stress / unique value / counts')
    np.savetxt(f'{id}_stress_range_out/data_strain.txt', data_s, header = 'number / stress / unique value / counts')
    np.savetxt(f'{id}_stress_range_out/data_meanfield.txt', data_m, header = 'number / stress / meanfield components')
    np.savetxt(f'{id}_stress_range_out/stress_values.txt', stress_range)
    
    print("\nSimulation: complete! " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n') ; del t0
    
""" ---------------------------------------------------------------------------------------------------------------- """

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
    '--lambd', '-l',
    action = 'store',
    type = float,
    required = True,
    help = 'The strength λ of the mean-field (Pa).')

parser.add_argument(
    '--range', '-r',
    action = 'store',
    nargs = 2,
    type = float,
    required = False,
    default = [0, 1e9],
    help = 'The range of magnitudes of the externally applied stress. The default is from 0 to 1 GPa.')

parser.add_argument(
    '--nrpoints', '-n',
    action = 'store',
    type = int,
    required = False,
    default = 200,
    help = 'The number of data points desired. The default is 200.')

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


if __name__ == '__main__':
    
    args = parser.parse_args()
    id = args.id
    ccdata = args.ccdata
    alpha = args.alpha
    gamma = args.gamma
    lambd = args.lambd
    stress_range = args.range
    nr_points = args.nrpoints
    stress = args.stress
    steps = args.steps
    iterations = args.iterations
    side_length = args.d
    temperature = args.temp
    starting_fraction = args.fraction
    processes = args.processes
    plotp = args.plotp
    
    main(id, ccdata, alpha, gamma, lambd, stress_range, nr_points, stress, steps, iterations, side_length, temperature, starting_fraction, processes, plotp)

    


