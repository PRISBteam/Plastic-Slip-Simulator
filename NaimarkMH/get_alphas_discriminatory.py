"""
Created on Wed Jul 17 12:15:36 2024

Last edited on: Oct 20 20:42 2024

@author: mbyxaad2
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math
from pathlib import Path
from tqdm import tqdm

sys.path.append('../Voronoi_PCC_Analyser/') ; sys.path.append('../') ; sys.path.append('./') ; sys.path.append('../dccstructure')
from matgen.base import CellComplex, Face3D, Edge3D
from dccstructure.iofiles import import_complex_data
from cellcomplex import createAllCells, scaleAllCells, createCellComplex
from cochain import Cochain
# from metrohast_sum import stress_cochain

""" -------------------------------------------------------------------------------------- """

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

def triaxiality_factor(tensor: np.ndarray) -> float:
    von_mises = np.sqrt(((tensor[0,0] - tensor[1,1])**2 + (tensor[1,1] - tensor[2,2])**2 + (tensor[2,2] - tensor[0,0])**2 + 6 * (tensor[0,1]**2 + tensor[1,2]**2 + tensor[2,0]**2)) / 2)
    return np.trace(tensor) / (3 * von_mises)

def round_small(number: float) -> float:
    return round(abs(number) / 10**orderOfMagnitude(abs(number)), 5) * 10**orderOfMagnitude(abs(number))

def euler_rot_x(theta):
    return np.array([[1,              0,             0],
                     [0,  np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])

def euler_rot_y(phi):
    return np.array([[ np.cos(phi), 0, np.sin(phi)],
                     [           0, 1,           0],
                     [-np.sin(phi), 0, np.cos(phi)]])

def euler_rot_z(psi):
    return np.array([[ np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [           0,           0, 1]])

def rotation(tensor, rotation):
    return (rotation @ (tensor @ rotation.transpose()))

def write_to_file(fname: str, lst: list):
    
    with open(f'{fname}.txt', 'w') as f:
        for x in lst:
            f.write(f"{x}\n")
            
def get_super(x): 
    normal = "0123456789+-"
    super_s = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"
    res = x.maketrans(''.join(normal), ''.join(super_s)) 
    return x.translate(res) 

""" -------------------------------------------------------------------------------------- """

data_folder = Path('../../Built_Complexes/FCC_5x5x5')
complex_size = [int(str(data_folder)[-5]), int(str(data_folder)[-3]), int(str(data_folder)[-1])] # the number of computational unit cells in each 3D dimension
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

dislocation_density = 1e12
if complex_size[0] %2 == 0:
    cube_side = 2 * np.sqrt(2/3) * complex_size[0] * (15 / np.sqrt(dislocation_density))
else:
    n = complex_size[0]
    cube_side = (2 * np.sqrt(2) * n**2 * 15 * np.sqrt(3*n**2-2*n+1)) / ((3*n**2-2*n+1) * np.sqrt(dislocation_density))
    del n
Nodes, Edges, Faces, Polyhedra = scaleAllCells(Nodes, Edges, Faces, Polyhedra, scaling = float(1e-6 / complex_size[0]))

cc: CellComplex = createCellComplex(dim = 3,
                                    nodes = Nodes,
                                    edges = Edges,
                                    faces = Faces,
                                    faces_slip = faces_slip,
                                    polyhedra = Polyhedra,
                                    ip = 2022)

del Nodes, Edges, Faces, faces_slip, Polyhedra

""" -------------------------------------------------------------------------------------- """

A2 = [] ; A3 = [] ; A6 = [] ; B2 = [] ; B4 = [] ; B5 = [] ;  C1 = [] ; C3 = [] ; C5 = [] ; D1 = [] ; D4 = [] ; D6 = []
vA2 = [] ; vA3 = [] ; vA6 = [] ; vB2 = [] ; vB4 = [] ; vB5 = [] ; vC1 = [] ; vC3 = [] ; vC5 = [] ; vD1 = [] ; vD4 = [] ; vD6 = []
burgers: float = 2.556e-10 # (m) at 293 K (copper)
yieldstress: float = 10.09e9
sample = [3054, 252, 6193, 435]

SA2l = Cochain(cc, 2, np.array([[1199, ]]))

for f in cc.get_many('f', sample):
    for e in cc.get_many('e', f.e_ids):
        vector = burgers * f.measure**2 / f.support_volume * np.array(e.unit_vector)
        S = Cochain(cc, 2, np.array([[f.id, *vector]]))
        slipsystem = get_active_system(f, e)
        if slipsystem == 'A2':
            A2.append(S)
        if slipsystem == 'A3':
            A3.append(S)
        if slipsystem == 'A6':
            A6.append(S)
        if slipsystem == 'B2':
            B2.append(S)
        if slipsystem == 'B4':
            B4.append(S)
        if slipsystem == 'B5':
            B5.append(S)
        if slipsystem == 'C1':
            C1.append(S)
        if slipsystem == 'C3':
            C3.append(S)
        if slipsystem == 'C5':
            C5.append(S)
        if slipsystem == 'D1':
            D1.append(S)
        if slipsystem == 'D4':
            D4.append(S)
        if slipsystem == 'D6':
            D6.append(S)

stress = np.array([[0,0,0],
                   [0,0,0],
                   [0,0,1]])

# stress = np.random.rand(3,3)
# stress = (stress + stress.T) / 2

angles = np.linspace(0, np.pi, 360)

for a in tqdm(angles, miniters = 30):
    new_stress = rotation(stress / np.linalg.norm(stress), euler_rot_x(a)) * yieldstress
    for S in A2:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vA2.append(alpha)
    for S in A3:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vA3.append(alpha)
    for S in A6:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        vA6.append(alpha)
    for S in B2:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vB2.append(alpha)
    for S in B4:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vB4.append(alpha)
    for S in B5:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        vB5.append(alpha)
    for S in C1:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vC1.append(alpha)
    for S in C3:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vC3.append(alpha)
    for S in C5:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        vC5.append(alpha)
    for S in D1:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vD1.append(alpha)
    for S in D4:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        if alpha > 0:
            vD4.append(alpha)
    for S in D6:
        stresscochain = Cochain(cc, 2, np.array([[f.id, *new_stress @ np.array(f.normal) * f.measure] for f in S.chain]))
        trial = stresscochain.inner_product_2022(S) / S.inner_product_2022()
        alpha = abs(trial) / 10**orderOfMagnitude(trial) # scale down
        alpha = np.round(alpha, 5) # round and get rid of similar values
        alpha = alpha * 10**orderOfMagnitude(trial) # scale up
        vD6.append(alpha)



        
scaling = (sum([p.measure for p in cc.polyhedra]) ** (1/3) / complex_size[0]) * yieldstress
xx = []
for a in angles:
    xx.extend([a])
    
# """ -------------------------------------------------------------------------------------- """

name = 'uniaxialtensionZaboutX'

# np.savetxt('values_' + name + '.txt', np.array([np.array(xx)*360/(2*np.pi), yy]).transpose())

# ALPHA VALUES
fig = plt.figure(figsize=(18,9))

# colors = ['#ffa600', '#ff6361', '#bc5090', '#58508d', '#003f5c']
colors = ['#66D671', '#D32784', '#13A48A', '#6F1706', '#001833']

ax = fig.add_subplot()
for yy, leg, clr, mkrstl, size in [(vA2, 'A2, B2, C1, D1', colors[0], 'o', 40),
                                    (vA3, 'A3, B4', colors[1], 'D', 40),
                                    (vC3, 'C3, D4', colors[2], 's', 40),
                                    (vA6, 'A6, B5', colors[3], '*', 60),
                                    (vC5, 'C5, D6', colors[4], '^', 60)]:
    ax.scatter(np.array(xx)[::3]*360/(2*np.pi),
                yy[::3] / (scaling * 10**orderOfMagnitude(max(yy[::3]) / scaling)),
                s = size,
                marker = mkrstl,
                label = leg,
                c = clr)
ax.set_xlabel('Angle of rotation (°)', size = 32, fontname = 'monospace', labelpad=15)
ax.set_xlim([0, 180])
ax.set_ylabel('$\\alpha \: / \: (\sigma_{\mathrm{y}} \sqrt[3]{V_{\mathrm{UC}}})$ ($\\times$10' + get_super(str(orderOfMagnitude(max(yy/scaling)))) + ' m$^{-1}$)',
              size = 32, fontname = 'monospace', labelpad = 15) # ×
ax.set_ylim([0, max(max(vA2), max(vA3), max(vA6),
                    max(vB2), max(vB4), max(vB5),
                    max(vC1), max(vC3), max(vC5),
                    max(vD1), max(vD4), max(vD6)) * 1.05 / (scaling * 10**orderOfMagnitude(max(yy) / scaling))])
ax.xaxis.set_major_locator(MultipleLocator(45))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.tick_params(axis = 'both', which = 'major', length = 10, labelfontfamily = 'monospace', labelsize = 25)
ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 22, markerscale = 2)
plt.subplots_adjust(left=0.095, bottom=0.135, right=0.8, top=0.98, wspace=0, hspace=0)
plt.savefig('values_' + name + '.pdf', dpi = 500)


""" For random stress """

# fig = plt.figure(figsize=(18,9))

# ax = fig.add_subplot()
# for yy, leg, clr, lnstl, size in [(vA2, 'A2', 'black', 'dotted', 5),
#                                   (vA3, 'A3', 'black', 'dashed', 5),
#                                   (vA6, 'A6', 'black', (0, (3, 5, 1, 5, 1, 5)), 5),
#                                   (vB2, 'B2', 'rebeccapurple', 'dotted', 5),
#                                   (vB4, 'B4', 'rebeccapurple', 'dashdot', 5),
#                                   (vB5, 'B5', 'rebeccapurple', (0, (5, 10)), 5),
#                                   (vC1, 'C1', 'mediumvioletred', 'solid', 5),
#                                   (vC3, 'C3', 'mediumvioletred', 'dashed', 5),
#                                   (vC5, 'C5', 'mediumvioletred', (0, (5, 10)), 5),
#                                   (vD1, 'D1', 'goldenrod', 'solid', 5),
#                                   (vD4, 'D4', 'goldenrod', 'dashdot', 5),
#                                   (vD6, 'D6', 'goldenrod', (0, (3, 5, 1, 5, 1, 5)), 5)]:
#     ax.plot(np.array(xx)*360/(2*np.pi),
#                 yy / (scaling * 10**orderOfMagnitude(max(yy[::3]) / scaling)),
#                 linewidth = size,
#                 linestyle = lnstl,
#                 label = leg,
#                 c = clr)
# ax.set_xlabel('Angle of rotation (°)', size = 32, fontname = 'monospace', labelpad=15)
# ax.set_xlim([0, 360])
# ax.set_ylabel('$\\alpha \: / \: (\sigma_{\mathrm{y}} \sqrt[3]{V_{\mathrm{UC}}})$ ($\\times$10' + get_super(str(orderOfMagnitude(max(yy/scaling)))) + ' m$^{-1}$)',
#               size = 32, fontname = 'monospace', labelpad = 15) # ×
# ax.set_ylim([0, max(max(vA2), max(vA3), max(vA6),
#                     max(vB2), max(vB4), max(vB5),
#                     max(vC1), max(vC3), max(vC5),
#                     max(vD1), max(vD4), max(vD6)) * 1.05 / (scaling * 10**orderOfMagnitude(max(yy) / scaling))])
# ax.xaxis.set_major_locator(MultipleLocator(45))
# ax.xaxis.set_minor_locator(MultipleLocator(5))
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(MultipleLocator(0.1))
# ax.tick_params(axis = 'both', which = 'major', length = 10, labelfontfamily = 'monospace', labelsize = 25)
# ax.tick_params(axis = 'both', which = 'minor', length = 8, labelcolor = 'black')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 22, markerscale = 2)
# plt.subplots_adjust(left=0.095, bottom=0.135, right=0.8, top=0.98, wspace=0, hspace=0)
# plt.savefig('values_' + name + '.pdf', dpi = 500)

