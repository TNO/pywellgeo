# PyWellGeo

**PyWellGeo** is a Python library for advanced well trajectory modeling, well data handling, and geothermal engineering workflows.  
It provides tools for representing, analyzing, and visualizing complex well architectures, including multi-branch wells, and supports a variety of input formats and engineering calculations.

## Features

- Flexible well trajectory modeling (vertical, deviated, multi-branch)
- Well data utilities (water properties, constants, DC1D well models)
- Well tree structures for advanced branching and perforation logic
- Geometric transformations (azimuth/dip, vector math)
- Integration with geothermal techno-economic workflows

## Installation

Install the latest version using pip:

```sh
pip install pywellgeo
```

Or install from source:

```sh
git clone https://github.com/TNO/pywellgeo.git
cd pywellgeo
pip install .
```

## Usage Examples

### Load and Work with a Well Trajectory

```python
from pywellgeo.welltrajectory.trajectory import Trajectory

# Create a trajectory from survey data or parameters
traj = Trajectory.from_xyz(
    x=[0, 100, 200],
    y=[0, 0, 0],
    z=[0, -500, -1000]
)

print(traj.length())
print(traj.get_md_tvd())
```

### Use Well Data Utilities

```python
from pywellgeo.well_data.names_constants import Constants

print(Constants.GRAVITY)
```

### Perform Azimuth/Dip Transformations

```python
from pywellgeo.transformations.azim_dip import AzimDip

azim, dip = AzimDip.vector_to_azim_dip([1, 1, -1])
print(f"Azimuth: {azim}, Dip: {dip}")
```

## Documentation

Full documentation is available at:  
[GitHub](https://github.com/TNO/pywellgeo)

---

For more examples and API details, see the [online documentation](https://tno.github.io/pywellgeo/).