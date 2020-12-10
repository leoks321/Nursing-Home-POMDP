using POMDPs
using Distributions: Normal, Uniform, DiscreteUniform, Multinomial, Hypergeometric, Binomial
using Random
using LinearAlgebra
using Statistics
using Plots
import POMDPs: initialstate, gen, actions, discount, isterminal
Random.seed!(1);

mutable struct m <: POMDPs.POMDP{Array{Int64,1},Int,Int}  #initialize POMDP 
    discount_factor::Float64
end
m() = m(.95)
discount(p::m) = p.discount_factor

using POMDPModelTools #actions 1 = wait, 2 = lockdown
actions(::m) = [1, 2]


function gen(m::m, s::Array{Int64,1}, a::Int, rng::AbstractRNG) #function to generate (s,a) \to (s',r)

    if a == 1
       sp = waitT(s)
    end
    if a == 2
       sp = lockT(s)
    end

  
    if a == 1
    r = 10*s[1]+ -5 * s[2] + 10*s[3] - 50 * s[4] #reward for waiting
    end
  
    if a == 2
    r =  -100 #reward forlockdown
    end
  
    return (sp=sp, r=r)
end;

np = Uniform(0.3,0.6)
function waitT(s::Array{Int64,1})  #state transition if action "wait" is taken
c=3 #average number of social contact

p= rand(np)#infection probability
ri = 0.05 #probability of randomly getting infected 
  
A = zeros(Int8, s[1])
  
new_cases = 0
if s[1] != 0 && s[2] !=0
  for i = 1:s[2] #each infected person contacts 3 people and infects them with probability p
    for j = 1:c
      ind = rand(1:s[1])
      b = rand(Uniform(0,1)) < p
      A[ind]= b ? 1 : A[ind]
    end
  end
 end
if s[1] !=0
  for i=1:s[1] #assume low chance of outside infection 
    a = rand(Uniform(0,1)) > ri 
    new_cases= !a ? new_cases+1 : new_cases+0
  end
end
 

s2p = [0.2,0.5,0.25,0.05] #multinomial for infected people
s2 = Multinomial(s[2],s2p)
s2r = rand(s2)

s3p = [0.0,0.02,0.98,0.0] #multinomial for immune people
s3 = Multinomial(s[3],s3p)
s3r = rand(s3)
  

n = min(s[1],new_cases + sum(A))
s=  [s[1]-n + s2r[1],s2r[2] + n+s3r[2],s3r[3] + s2r[3] ,s[4] + s2r[4]+s3r[4]]
return s
  
end

function lockT(s::Array{Int64,1}) #state transition if action "lockdown" is taken

ri = 0.02
  new_cases = 0
if s[1] !=0
  for i=1:s[1]
    a = rand(Uniform(0,1)) > ri
    new_cases= !a ? new_cases+1 : new_cases+0
  end
end
 

s2p = [0.2,0.5,0.25,0.05]#multinomial for infected people
s2 = Multinomial(s[2],s2p)
s2r = rand(s2)

s3p = [0.0,0.02,0.98,0.0]#multinomial for immune people
s3 = Multinomial(s[3],s3p)
s3r = rand(s3)
  

n = min(s[1],new_cases)
s=  [s[1]-n + s2r[1],s2r[2] + n+s3r[2],s3r[3] + s2r[3] ,s[4] + s2r[4]+s3r[4]]
return s
  
end

using Distributions

function hyp(s::Array{Int64,1},sen::Float64,spe::Float64,smpl::Int,x::Int) #calculate PDF for observation distribution

  if s[2] < 0
    s[2]=0
  end
  h=Hypergeometric(s[2],s[1]+s[3],smpl)
  
    p=0
    for i=0:min(smpl,d.s[2]) #get anywhere from 0 to min(m,inf) infected people
        p = p + pdf(h,i) * bin(x,i,smpl-i,sen,spe) #probability of i infected people, times probability of x positives given i infected people
    end
    return p
end
function bin(t::Int, I::Int, NI::Int,sen::Float64,spe::Float64)
  bI = Binomial(I,sen)
  bNI = Binomial(NI,1-spe)
  tp =0
  for i=0:t
    #println(i)
    p = pdf(bI,i) * pdf(bNI,t-i)
    #print(p)
    tp = tp +p
  end
  return tp
end

function Obs(s::Array{Int64,1},sen::Float64,spe::Float64,smpl::Int) #generate observation in state s, sample size smpl, sensitivity sen and specificity spe
  inf = 0
 
  tinf = s[2]
  tnotinf = s[1]+s[3]
  
  posit = 0
  
  for i=1:smpl #choose random sample
    a = rand(Uniform(0,tinf+tnotinf))<tinf
    inf = a ? inf+1 : inf+0
    tinf = a ? tinf-1 : tinf+0
    tnotinf = !a ? tnotinf-1 : tnotinf+0
  end
  for i=1:inf #test infected people
    posit = rand(Uniform(0,1)) < sen ? posit+1 : posit+0
  end
  for i=1:smpl-inf #test uninfected people
    posit = rand(Uniform(0,1)) > spe ? posit+1 : posit+0
  end
  
  return posit
  
end

struct mDist #define observation distribution
    s::Array{Int,1}
end


function Base.rand(rng::AbstractRNG, d::Random.SamplerTrivial{mDist}) #generate observations
  return Obs(d[].s,0.98,0.98,d[].s[1]+d[].s[2]+d[].s[3])
end

function POMDPs.pdf(d::mDist, x::Int) #PDF of observation distribution
  return hyp(d.s,0.98,0.98,d.s[1]+d.s[2]+d.s[3],x)
end

function POMDPs.observation(p::m,a::Int,sp::Array{Int,1})
     return mDist(sp) 
end

d = mDist([10,20,30,40]) #example of observation distribution

struct initialD #initial state distribution
end


function Base.rand(rng::AbstractRNG, d::Random.SamplerTrivial{initialD}) #here initial state is deterministic
  return [90,10,0,0]
end

start = initialD()

using Distributions, LinearAlgebra, Statistics

POMDPs.initialstate(m::m) = start #set initial state

using POMDPPolicies: FunctionPolicy #heuristic policy to evaluate leaf nodes, always wait

function my_heuristic(b::Array{Int64,1})

return 1
 
end

heuristic_policy = FunctionPolicy(my_heuristic)

#initialize solver
using BasicPOMCP 
using POMDPSimulators
solver = POMCPSolver(tree_queries=100000, c=7000, max_depth=20, estimate_value = FORollout(heuristic_policy)) 
pomdp = m()
planner = solve(solver, pomdp);

Random.seed!(2); #run POMCP solver for 30 steps

using ParticleFilters
function runCP()
i=0
rw=0
bel = zeros(Float64, 30)
filter = SIRParticleFilter(pomdp, 2000)

for (s,a,r,sp,o,b) in stepthrough(pomdp, planner, filter, "s,a,r,sp,o,b") #show states(s,sp), rewards(r), observation(o), belief(b)
    @show (s,a,r,sp,o)
  println(mean(b))
    rw+=r
  bel[i+1] = mean(b)[2]
i+=1
  if i == 30
    break
  end
end
println(rw)
println(bel)
end
println("POMCP Run Through")
runCP()


using POMDPPolicies: FunctionPolicy #custom policies

function cutoff(b)
    if mean(b)[2] > 8
    return 2
  else
    return 1
  end
end



cutoff_p = FunctionPolicy(cutoff)



using ParticleFilters #example of simple strategy
function runSimp(k)

function simple(b)
  if i % k == 0
    return 2
  else
    return 1
  end
end


simple_p = FunctionPolicy(simple)
filter = SIRParticleFilter(pomdp, 2000)
i = 0

rwe = 0

for (s,a,r,sp,o,b) in stepthrough(pomdp, simple_p, filter, "s,a,r,sp,o,b")
    @show (s,a,r,sp,o)
  println(mean(b))
    rwe+=r
    i+=1
  if i == 30
    break
  end
end
println(rwe)
end
println("Simple Strategy Run Through")
runSimp(2)

using ParticleFilters #example of cutoff strategy
function runCUT()
filter = SIRParticleFilter(pomdp, 2000)
i = 0

rwe = 0

for (s,a,r,sp,o,b) in stepthrough(pomdp, cutoff_p, filter, "s,a,r,sp,o,b")
    @show (s,a,r,sp,o)
  println(mean(b))
    rwe+=r
    i+=1
  if i == 30
    break
  end
end
println(rwe)
end
println("Cutoff Heuristic Run Through")
runCUT()

Random.seed!(1);

#evaluate simple policy
using ParticleFilters
using POMDPPolicies: FunctionPolicy
function runSimp2(k)

function simple(b)
  if i % k == 0
    return 2
  else
    return 1
  end
end


simple_p = FunctionPolicy(simple)
filter = SIRParticleFilter(pomdp, 2000)
i = 0

rwe = 0

for (s,a,r,sp,o,b) in stepthrough(pomdp, simple_p, filter, "s,a,r,sp,o,b")
    #@show (s,a,r,sp,o)
  #println(mean(b))
    rwe+=r
    i+=1
  if i == 30
    break
  end
end
#println(rwe)
return rwe
end

function runSE()
for j = 1:6
tr  = 0
println("Every: ")
println(j)
for f = 1:10
tr += runSimp2(j)
end
println(tr/10)
end

end

runSE()





#cutoff heuristic
using POMDPPolicies: FunctionPolicy
using ParticleFilters
function evalC()


filter = SIRParticleFilter(pomdp, 2000)
k = -1
for c = 1:15
  
  k += 1 #try different k values
  print("Cutoff: ")
  println(k)
function cutoff(b) #lockdown if greater than k infected people in belief state
  if mean(b)[2] > k
    return 2
  else
    return 1
  end
end

cutoff_p = FunctionPolicy(cutoff)
  tr = 0
  for iter = 1:30
    i = 0
    rwe = 0

for (s,a,r,sp,o,b) in stepthrough(pomdp, cutoff_p, filter, "s,a,r,sp,o,b")
    #@show (s,a,r,sp,o)
  #println(mean(b))
    rwe+=r
    i+=1
  if i == 30
        #print(sp)
    break
  end
end
   tr += rwe
    #println(rwe)
  end
  println(tr/30)
end
end
evalC()
Random.seed!(2);
using ParticleFilters
function evalP()
#POMCP evaluation

filter = SIRParticleFilter(pomdp, 2000)
tr = 0
for iter = 1:10 #run 10 simulations
  i = 0
  rwe = 0

for (s,a,r,sp,o,b) in stepthrough(pomdp, planner, filter, "s,a,r,sp,o,b")
    #@show (s,a,r,sp,o)
  #println(mean(b))
    rwe+=r
    i+=1
  if i == 30
    break
  end
end
  println(rwe)
   tr += rwe
end
println("Average: ")
  println(tr/10)
end
evalP()