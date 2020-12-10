using POMDPs
using Distributions: Normal, Uniform, DiscreteUniform, Multinomial, Hypergeometric, Binomial
using Random
using LinearAlgebra
using Statistics
using Plots
import POMDPs: initialstate, gen, actions, discount, isterminal
Random.seed!(1);
