# SimpleSheet.jl
Except for the *valley_forward-euler_CN.jl* script, everything in this repo concentrates on the SHMIP A1 case: ice sheet margin geometry, flat bet and sqrt function for surface.
All the scripts have the option to set h dirichlet boundary condition (`set_h_bc=true`). Additional options are listed below. `e_v_num` is an additional void ratio for regularisation that doesn't lead to actual storage of water.

### Forward Euler, Crank-Nicolson
*forward-euler_CN.jl*
- `e_v_num`
- `use_CFL`: time steps are a few orders of magnitude smaller than default dt=1s
- `CN`: CN=0: Forward Euler, CN=0.5: Crank-Nicolson

In any case this scheme takes a long time because the time steps are so small, not possible to reach steady state in reasonable time even for 64x32 grid.

### Pseudo transient
*pseudo-transient.jl*
- `e_v_num`
- `update_h_only`: true to use the split step scheme

The split step scheme only works for the 64x32 grid, but there it goes to steady state very fast.

### DifferentialEquations.jl
*diffeq_explicit.jl* : explicit ROCK4 solver
*diffeq_implicit.jl* : implicit TRBDF2 solver
*diffeq_impl-sundials.jl* : implicit sundial solvers
- `e_v_num`

### Runke-Kutta
*RK4.jl*
- `e_v_num`

Fourth order Runge Kutta



*RK4_piccard_loop.jl*

For each physical time step there are a few piccard type iterations carried out.

### Others

*original_PT_parallel.jl*

Initial pseudo-transient implementation from the SheetModel repo, does not include any of the options above.

*valley_forward-euler_CN.jl*

Same implementation as *forward-euler_CN.jl* just with the E1 SHMIP case instead of A1, so with a valley glacier topography. (A bit outdated though, not all of the options available.)