# SimpleSheet.jl

### Explicit solver
- `run_it_simple.jl`: Either fixed time step or CFL criterion; with option for Crank-Nicolson scheme. In any case it takes a long time because the time steps are so small, not possible to reach steady state in reasonable time even for 64x32 grid.
- `run_it_advection.jl`: Same as above but solving Bueler & Pelt (2015) equations. Requires BC as h needs the divergence now, but setting Dirichlet BC for h results in weird values. CFL criterion for h gives NaNs.

### Pseudo transient
- `run_it_simple_PT.jl`: Pseudo-transient method with option for split step scheme. The latter only works for the 64x32 grid, but there it goes to steady state very fast.

### DifferentialEquations.jl
- `run_it_diffeq_expl.jl`: also takes very long; tried ttot=1e6s, took 12min (64x32 grid) and result was still quite far away from the expected steady state solution.
- `run_it_diffeq_expl_two_e_v.jl`:

### Runke-Kutta
- `run_it_simple_RK4.jl`:
- `run_it_simple_RK4_1.jl`: