# Topology Optimization for Full-Cell Electrochemical Energy Storage Device (TOPEFC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15115948.svg)](https://doi.org/10.5281/zenodo.15115948)

This code is for the topology optimization of full-cell batteries and supercapacitor porous electrodes. It was used to produce the results in [Li et al., 2024](https://link.springer.com/article/10.1007/s00158-024-03901-z). Please see the paper for more details.

For half-cell optimization, please refer to [TOPE](https://github.com/LLNL/TOPE) and [Roy et al. 2022](https://link.springer.com/article/10.1007/s00158-022-03249-2).

LLNL Release Number: LLNL-CODE-2003445

## Requirements

Please install the open-source finite element library [Firedrake](https://www.firedrakeproject.org/download.html), and the MMA optimization package [pyMMAopt](https://github.com/LLNL/pyMMAopt). For automation of bash jobs, we use [signac-flow](https://github.com/glotzerlab/signac-flow) (0.12.0).
