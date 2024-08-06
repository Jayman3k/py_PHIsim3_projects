# Introduction

This project contains the code for the parallel simulation engine I developed in the context of my master's thesis. 
Its purpose is to speed up simulation work using PHIsim (https://sites.google.com/tue.nl/phisim/home) and was designed specifically for PHIsimV3.
We achieve this by defining simulation objects which contain all necessary information for running a PHIsim instance; and then spinning out these instances in separate processes.
The framework also allows you to define hooks to collect information from a simulation, so some of the data processing can also be parallelized.