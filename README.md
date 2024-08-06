# Introduction

This project contains the code for the parallel simulation engine I developed in the context of my master's thesis. 
Its purpose is to speed up simulation work using PHIsim (https://sites.google.com/tue.nl/phisim/home). Note that this framework was designed specifically for PHIsimV3, and might require updates to work with different versions.

We achieve the parallelization by defining simulation objects that contain all necessary information for running a PHIsim instance; then we spin out these instances to separate processes.
The framework also allows you to define hooks to collect information from a simulation, so some of the data processing can also be parallelized.