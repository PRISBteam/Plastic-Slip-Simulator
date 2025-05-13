# 1. Building a polyhedral cell complex (PCC) that reproduces the most common slip planes in FCC and BCC crystals

The package **dccstructure** contains Python modules to build a discrete cell complex (DCC) based on the slip planes of crystal nanostructures (simple cubic, FCC, BCC; HCP not yet available).

## Modules

### run.py

The script **run.py** is a summarised execution of the whole package. It takes the following input arguments on a terminal/command line:

Mandatory:

    --size int int int : specifies the number of unit cells in the directions x, y, z (default is 1 1 1);
    --struc str : specifies the crystallographic structure of the complex. Options: ['fcc', 'bcc'];

Optional:

    --dim int : specifies the dimension of the complex (default is 3);
    --basisv flt flt flt flt flt flt flt flt flt : specifies the 9 components of the 3 lattice basis vectors;
    --mp bool : if passed, the code will run with Python's multiprocessing package, i.e. paralellisation of operations. This will not work if all of the complex dimensions (as passed to --size) are smaller than the number of available CPUs;
    -d bool : if passed, the code will output the node degrees;
    -n bool : if passed, the code will output the unit normals to the 2-cells;
    -a bool : if passed, the code will output the areas of the 2-cells;
    -v bool : if passed, the code will output the volumes of the 3-cells;
    -s bool : if passed, the code will output the indices of 2-cells corresponding to slip planes.
        
This module is meant to be run from a command line/terminal that contains the dccstructure directory (*i.e.* one level above the dccstructure folder). It executes the whole package from scratch as intended, building a discrete (simplicial) cell complex with the parameters specified by the arguments above. It returns the adjacency and incidence matrices of the complex, as well as other topological and geometrical information (optional arguments). This file can be run from a terminal with the following command (ignore the + sign at the start and remove the angular brackets):
```diff
+    python -m dccstructure --size <int int int> --basisv <flt flt flt flt flt flt flt flt flt> --struc <str> (+ optional arguments)
```

### build.py

This module contains the functions necessary to construct the complex, from 0-cells to 3-cells. In here is also defined the function build_complex() which executes the whole module in sequence and returns the node coordinates and the edges, faces and volumes as lists of the constituent nodes.

### orientations.py

This module contains functions that gauge orientations on 2- and 1-cells based on a canonical outward orientation (positive) for every 3-cell and positive orientation for every 0-cell. It also cincludes functions that establish relative orientations between 3- and 2-cells, and 2- and 1-cells. The relative orientation between 1- and 0-cells is defined to follow the pattern [-1, 1], so that an edge always points from its first node to its second node as given by the find_edges() function in **build.py**.

### matrices.py

This module contains functions that compute the node degree distribution and topologically-relevant matrices such as adjacency matrices, oriented incidence matrices, and other matrices which relate 3-cells to 1-cells. Matrices between p-cells and 0-cells (p = 1,2,3) are encoded into the definitions of the volumes, faces and edges as given by the functions in **build.py**.

### geometry.py

This module contains functions which define geometric quantities such as normal vectors, areas, volumes, angles and steradian angles, useful, for example, for the functions defined in **operations.py**.

### operations.py

This module contains functions which define metric operations (and related quantities) on the complex, such as node weights, the metric tensor, an inner product, an adjoint coboundary operator and a star operator.

### iofiles.py

This module contains a useful function for automating outputs of matrices and other variables as .txt files.

## Known/unresolved issues:

1. The function build.find_neighbours() returns a TypeError for an asymmetrical BCC structure.


# 2. A Metropolis-Hastings (MH) algorithm for computing microscopic plastic slips in FCC crystals

The package **NaimarkMH** contains Python modules that define a *Cochain* and *MHalgorithm* classes. The first is used to define real-valued or vector-valued cochains on a cell complex, while the second is used run a MH algorithm to minimise the energy of a plastic system based on a combinatorial version of Naimark's model (1998).

## Modules

### cellcomplex.py

This module contains functions that take the data from a PCC, previously built using the **dccstructure** package, and the classes defined in the **base.py** module in *PRISBteam/Voronoi_PCC_Analyser/matgen* to restructure the data output by **dccstructure** into the data structure of the *CellComplex* class defined in **base.py**.

### cochain.py

This module defines the *Cochain* class as a subclass of the *CellComplex* class defined in the **base.py** module in *PRISBteam/Voronoi_PCC_Analyser/matgen*. The *Cochain* class contains a dictionary that assigns values (integer, floating-point or array-like) to the cells of a cell complex. It is furnished with several methods, including the dunder methods +, -, *, == and len. Other methods include the cup product and the inner product of cochains, all defined to reproduce the theory of Berbatov et al. (2022) and Berbatov (Thesis) (2023).

### metrohast_sum.py

This module defines the *MHalgorithm* class used to compute the energy minimisation of a system of plastic slips in an FCC complex, using the data output by **dccstructure** and restructured by **cellcomplex.py**. Using the *Cochain* class defined in **cochain.py**, slips are mathematically defined as vector-valued 2-cochains.

### get_alphas_discriminatory.py

This script was used to find the dependency of the self-energy of microslips on the rotation of the applied stress tensor, specifically in the case of uniaxial tension in the z direction rotating about the x axis. It is called 'discriminatory' because it discriminates between the slip systems.

### run_range_with_meanfield.py

This module employs the *MHalgorithm* class from **metrohast_sum.py** and runs several simulations (i.e. iterations) in parallel for a range of values of the applied stress magnitude. It can be run from a terminal/command line.

### run_lambda_range.py

This module employs the *MHalgorithm* class from **metrohast_sum.py** and runs several simulations (i.e. iterations) in parallel for a range of values of the mean-field coupling parameter. It can be run from a terminal/command line.

## Acknowledgements

This code has been created as a part of the EPSRC funded projects EP/V022687/1 _“Patterns recognition inside shear bands: tailoring microstructure against localisation”_ (PRISB).


## License

Distributed under the GNU General Public License v3.0.


Last updated on: 13 May 2025
