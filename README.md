# DCC_Structure_Generator

The package DCC_Structure contains Python modules to build a discrete cell complex (DCC) based on the slip planes of crystal nanostructures (simple cubic, FCC, BCC; HCP not yet available). The script **execute.py** is a summarised execution of the whole package. It takes as input:
- the dimension of the complex: pass the argument as an int 2 or 3. The code should work fine for 2 dimensions but this has not been thoroughly tested;
- the desired structure: pass the argument as a str 'simple cubic', 'fcc', 'bcc', or 'hcp' (not yet available);
- the number of unit cells in each direction: pass the argument as three ints X Y Z. Here, a 'unit cell' is a volume bounded by 8 simple cubic-like vertices, and it is the division of this unit cell (or not, for simple cubic) into particular combinations of tetrahedra that makes the specified structure. For example, a bcc unit cell in this case has 24 tetrahedral 3-cells;
- the lattice basis vectors: pass the argument as nine ints a1, a2, a3, b1, b2, b3, c1, c2, c3. These are the vectors between corners of the unit cell as explained above. A complex with non-orthogonal basis vectors has not been tested;
- whether or not to also output the node degree matrix;
- whether or not to also output the unit normal vectors to 2-cells;
- whether or not to also output the areas of the 2-cells;
- whether or not to also output the indices of the 2-cells corresponding to slip planes;
- whether or not to also output the 'results.txt' file as described in the function iofiles.write_to_file().

## Modules

### __main__.py

This module is meant to be run from a command line/terminal in the directory containing the dccstructure package. It executes the whole package from scratch as intended, building a discrete (simplicial) cell complex with the parameters specified. This file can be run with the command
    python -m dccstructure (+ arguments)

### build.py

This module contains the functions necessary to construct the complex, from 0-cells to 3-cells.

### orientations.py

This module contains functions that gauge orientations on 1- and 2-cells and establish relative orientations between 3- and 2-cells, and 2- and 1-cells.

### matrices.py

This module contains functions that compute the node degree distribution and topologically-relevant matrices such as adjacency matrices, oriented incidence matrices, and other matrices which relate 3-cells to 1-cells. Matrices between p-cells and 0-cells (p = 1,2,3) are encoded into the definitions of the volumes, faces and edges as given by the functions in build.py.

### geometry.py

This module contains functions which define geometric quantities such as normal vectors, areas, volumes, angles and steradian angles, useful for the functions defined in operations.py

### operations.py

This module contains functions which define metric operations (and related quantities) on the complex, such as node weights, the metric tensor, an inner product, an adjoint coboundary oeprator and a star operator.

### io.py

This module contains a useful function for automating outputs of matrices and other variables.

### visualisation.py

This module contains a (non-optimised and not as-of-yet user-friendly) function that outputs pictures of the complex with selected 1-cells, 2-cells or 3-cells highlighted.

### execute.py

See top.

### build_complex.py

This module only has two functions: build_complex() summarises the whole process of building a complex, useful for quickly creating a complex to test out other functions, and bc_main() is a function which is callable via command line/terminal and which outputs several quantities into .txt files.

## Known/unresolved issues:

1. The function build.find_neighbours() returns a TypeError for an asymmetrical BCC structure.
2. The function visualisation.graph_complex() is currently unoperational, since it needs yet to be updated to the module-like version of the code (as it is written, it is still from the time when the whole code was written in a single .py file).
3. There might be issues with importing modules from within the the package itself.
