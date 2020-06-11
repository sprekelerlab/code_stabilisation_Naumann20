# Code for Naumann20
Publication: "Presynaptic inhibition rapidly stabilises recurrent excitation in the face of plasticity" by L. Naumann and H.Sprekeler

This repository provides the model code for the rate and spiking network and scripts showing how to run the models. Reproducing the publication figures requires to run the same model code over different parameters (beta, tau_c, w_ei and w_ee). Especially for the plasticity experiments, do NOT run the parameter sweeps in sequence on a standard laptop - it will take days. I highly recommend to use a compute cluster to run different parametrisations in paralell. 


## Rate network
The script `rate_network_cython.pyx` contains the core model code for the rate network. For simulation speed it is written in Cython and thus requires compilation so that the code can run purely in *C*.

### Compilation
The script `setup.py` can be used to compile the Cython code. To compile, navigate to the directory that contains both `setup.py` and `rate_network_cython.pyx` and run the following command in the terminal:  
  
`python setup.py build_ext --inplace`  
  
This will create a file `rate_network_cython.c`, which can now be imported in to python to run the functions it contains. It might also create other files or directories for the build that can be ignored.

### Running the model
Using the compiled model code you can now run simulations with different parameters (without recompiling). For an example of how to import the Cython function, run the code and plot the results look at `run_rate_network.py`. This script also contains a function that creates connection matrices with a fixed indegree that need to be passed to the model code and an overview of the parameters that should be passed to the model code.


## Spiking network


The spiking network is simulated using the Brian2 simulator. You can install Brian2 using `conda install`, `pip` or by fetching the latest release from Github.  
The skript `spiking_network.py` contains the model code as well as an example of how to run the model and plot the results. With the seeting in the run function Brian2 will compile the code and run it in *C* for simulation speed. Beware -- it's still 5000 neurons and will take time to run. How long depends on your machine, the length of the simulation, at which rate the network fires and whether you are monitoring spikes and weights. On an older Mac Laptop 10 seconds of simulation time take about 1-2 minutes.
