# PCC Discrete Plasticity

The package DCC_Structure contains Python modules to build a discrete cell complex (DCC) based on the slip planes of crystal nanostructures (simple cubic, FCC, BCC; HCP not yet available).

## Modules

### run.py

The script **run.py** is a summarised execution of the whole package. It takes the following input arguments on a terminal/command line:

Mandatory:

    --size int int int : specifies the number of unit cells in the directions x, y, z (default is 1 1 1);
    --struc str : specifies the crystallographic structure of the complex;

Optional:

    --dim int : specifies the dimension of the complex (default is 3);
    --basisv flt flt flt flt flt flt flt flt flt : specifies the 9 components of the 3 lattice basis vectors;
    --mp : if passed, the code will run with Python's multiprocessing package, i.e. paralellisation of operations. This will not work if all of the complex dimensions (as passed to --size) are smaller than the number of available CPUs;
    -d : if passed, the code will output the node degrees;
    -n : if passed, the code will output the unit normals to the 2-cells;
    -a : if passed, the code will output the areas of the 2-cells;
    -v : if passed, the code will output the volumes of the 3-cells;
    -s : if passed, the code will output the indices of 2-cells corresponding to slip planes.
        
This module is meant to be run from a command line/terminal in the dccstructure directory (*i.e.* inside the dccstructure folder). It executes the whole package from scratch as intended, building a discrete (simplicial) cell complex with the parameters specified by the arguments above. It returns the adjacency and incidence matrices of the complex, as well as other topological and geometrical information (optional arguments). This file can be run from a terminal with the following command (ignore the + sign at the start and remove the angular brackets):
```diff
+    python run.py --size <int int int> --basisv <flt flt flt flt flt flt flt flt flt> --mp (+ optional arguments)
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


## Acknowledgements

This code has been created as a part of the EPSRC funded projects EP/V022687/1 _“Patterns recognition inside shear bands: tailoring microstructure against localisation”_ (PRISB).


## License

Distributed under the GNU General Public License v3.0.


Last updated on: 25 March 2024
