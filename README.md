# Nursing-Home-POMDP
implemented using https://github.com/JuliaPOMDP/POMDPs.jl
The notebook was written in Julia (1.5.2) using the nteract (https://nteract.io/) notebook application.
To run the code you will need to have some packages imported. All packages related to juliaPOMDP can be found at https://github.com/JuliaPOMDP/POMDPs.jl. You will need:

 -POMDPs
 
 -POMDPModelTools
 
-POMDPPolicies

-BasicPOMCP

-POMDPSimulators

-ParticleFilters

any other POMDP related packages can be found in juliaPOMDP.jl. You will also need to have some general packages imported:

 -Distributions
 
-Random

-LinearAlgebra

-Statistics
 
-Plots

The code can be run in two ways:

1. Run this code in a nteract/jupyter interactive notebook. Download nteract (https://nteract.io/) and follow instructions at https://juliaacademy.com/courses/intro-to-julia/lectures/16882463 to start nteract and julia. Make sure all packages are installed and copy code from NursingHomeModel.ipynb into the nteract notebook and run.

2. Alternatively NursingHomeModel.jl is a julia script that can also be run.
