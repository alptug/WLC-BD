# WLC-BD
## Overview
WLC-BD is a Python implementation of the Brownian Dynamics Simulation for the Worm-Like Chain (WLC) model. This repository provides tools to simulate and analyze the behavior of polymer chains using the WLC model.

## Model
The model assumes the worm-like chain is attached to a hard wall located at $$z=0$$, The other end of the chain is attached to a microsphere which is pulled along the positive $$z$$-axis with a force $$F_{mag}$$. The hydrodynamic effect of the hard wall is modelled by the Faxén correction of the Stoke's drag.

## Features
- Simulates the Brownian motion of the microsphere using Milstein method.
- Automatically choses the time step to be 2 orders of magnitude smaller than the charactristic time of the motion

## Usage
Modify the "Global Parameters for the simulation" given at the beginning of `WLC_BD_simulation.py` and to run a simulation, use the following command:

```bash
python WLC_BD_simulation.py
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or suggestions, please open an issue or contact the repository owner. 
