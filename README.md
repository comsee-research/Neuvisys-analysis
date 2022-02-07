# Spiking Neural Network Analisys Library.

## Requirements

A python envrionment.
Alternatively, the library is made to be used with poetry.

### Installation with Poetry

- Install poetry:
``curl -sSL https://install.python-poetry.org | python3 -``

- Install the dependencies from the library folder:
``poetry install``

- You can then either launch a script with ``poetry run python your_script.py`` or activate the virtual environment with ``poetry shell``.

## Python Packages

Here is the list of packages and what they do:

- driving dataset: a package to work with the DDD17 driving dataset.
- events: a package to create, modify and convert event files.
- frames: a package to do traditional frame analysis.
- spiking_network: a package to modify, visualize and launch the SNN of the neuvisys project.

## Jupyter-Notebooks

There is two jupyter-notebooks which offer an interface to most of the packaged functions mentioned before:

- Neuvisys.ipynb: Used to visualize various information from a spiking network.
- Utils.ipynb: Used to launch and modify spiking networks.
