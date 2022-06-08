# DCC_Slip_Structure_Generator

The package DCC_Structure contains Python modules to build a discrete cell complex (DCC) based on the slip planes of crystal nanostructures (simple cubic, FCC, BCC; HCP not yet available). The script execute.py is a summarised execution of the whole package. It takes as input:
- the dimension of the complex: 2 or 3. The code should work fine for 2 dimensions but this has not been thoroughly tested;
- the desired structure: simple cubic, fcc, bcc, or hcp (not yet available);
- the number of unit cells in each direction: X Y Z. Here, a 'unit cell' is a volume bounded by 8 simple cubic-like vertices, and it is the division of this unit cell (or not, for simple cubic) into particular combinations of tetrahedra that makes the specified structure. For example, a bcc unit cell in this case has 24 tetrahedral 3-cells; and
- the lattice basis vectors: [a1, a2, a3] [b1, b2, b3] [c1, c2, c3]. These are the vectors between corners of the unit cell as explained above. A complex with non-orthogonal basis vectors has not been tested.

## Modules

### build.py

This module contains the functions necessary to construct the complex, from 0-cells to 3-cells.

### orientations.py

This module contains functions that gauge orientations on 1- and 2-cells and establish relative orientations between 3- and 2-cells, and 2- and 1-cells.

### matrices.py

This module contains functions that compute the node degree distribution and topologically-relevant matrices such as adjacency matrices, oriented incidence matrices, and other matrices which relate 3-cells to 1-cells. Matrices between p-cells and 0-cells (p = 1,2,3) are encoded into the definitions of the volumes, faces and edges as given by the functions in build.py.

### io.py

This module contains a useful function for automating outputs of matrices and other variables.

### visualisation.py

This module contains a (non-optimised and not as-of-yet user-friendly) function that outputs pictures of the complex with selected 1-cells, 2-cells or 3-cells highlighted.

