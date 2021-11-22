# SimpleSheet.jl
All the scripts have the options
- set_h_bc
- e_v_num

### Forward Euler, CN
`forward-euler_CN.jl`
- use_CFL              # time steps are a few orders of magnitude smaller than default dt=1s
- CN        -> CN=0: Forward Euler, CN=0.5: Crank-Nicolson

In any case this scheme takes a long time because the time steps are so small, not possible to reach steady state in reasonable time even for 64x32 grid.

### Pseudo transient
`pseudo-transient.jl`
- update_h_only       # true to use the split step scheme

The split step scheme only works for the 64x32 grid, but there it goes to steady state very fast.

### DifferentialEquations.jl
`diffeq_explicit.jl`

### Runke-Kutta
`runge-kutta_4`

Fourth order Runge Kutta


`run_it_simple_RK4_1.jl`: