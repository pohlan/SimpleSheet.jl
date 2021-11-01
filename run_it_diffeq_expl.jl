# Solving the sheet using OrdinaryDiffEq solvers (ROCK4)
#
# TODO:
# - try the mass-conservation equation used by B&P as well
# - try on the GPU

# - also try implicit solvers, but that will be a lot more work

using Printf, LinearAlgebra, Statistics, Plots, Test, RecursiveArrayTools, OrdinaryDiffEq,
    Infiltrator
# pyplot()

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small    = eps(Float64)

function make_ode()
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    dt     = 1e0                          # physical time step
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-3                               # void ratio for englacial storage

    # numerics
    nx, ny = 64, 32
    itMax  = 1e7
    nout   = 1e4
    # derived
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)

    # initial conditions
    ϕ0     = 100. * ones(nx, ny)
    h0     = 0.04 * ones(nx, ny) # initial fields of ϕ and h
    get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1
    H      = [0.0; ones(nx-2); 0.0] * [0.0 ones(ny-2)' 0.0] .* get_H.(xc, yc') # ice thickness, rectangular ice sheet with ghostpoints
    ϕ0[1:2,:] .= 0.0 # Dirichlet BC

    # scaling factors
    H_     = 1000.0
    ϕ_     = 9.81 * 910 * H_
    h_     = 0.1
    x_     = max(Lx, Ly)
    q_     = 0.005 * h_^α * (ϕ_ / x_)^(β-1)
    t_     = h_ * x_ / q_
    Σ      = 5e-8 * x_ / q_
    Γ      = 3.375e-25 * ϕ_^3 * x_ / q_ * 2/27  # the last bit is 2/n^n from vc
    Λ      = m * x_ / q_

    # apply the scaling and convert to correct data type
    ϕ0     = ϕ0 ./ ϕ_
    h0     = h0 ./ h_
    dx     = dx / x_
    dy     = dy / x_
    dt     = dt / t_
    H      = H ./ H_

    # array allocationsize(Λ)
    dϕ_dx  = zeros(nx-1,ny  )
    dϕ_dy  = zeros(nx  ,ny-1)
    gradϕ  = zeros(nx-2,ny-2)
    d_eff  = zeros(nx  ,ny  )
    flux_x = zeros(nx-1,ny  )
    flux_y = zeros(nx  ,ny-1)
    vo     = zeros(nx  ,ny  )
    vc     = zeros(nx  ,ny  )

    # initialize all ϕ and h fields
    ϕ_old = copy(ϕ0)
    h_old = copy(h0)

    ode! = @views function (du,u,p,t)
        h, ϕ = u.x
        ## to avoid errors from Complex numbers:
        h .= max.(h,0)
        # alternative:
        #h .= abs.(h)

        # dirichlet boundary conditions to pw = 0
        ϕ[1:2,:] .= 0.0

        dhdt = du.x[1]
        dϕdt = du.x[2][2:end-1,2:end-1]
        # d_eff
        dϕ_dx  .= diff(ϕ, dims=1) ./ dx
        dϕ_dy  .= diff(ϕ, dims=2) ./ dy

        dϕ_dx[1,:] .= 0.0; dϕ_dx[end,:] .= 0.0
        dϕ_dy[:,1] .= 0.0; dϕ_dy[:,end] .= 0.0

        gradϕ  .= sqrt.( av_xi(dϕ_dx).^2 .+ av_yi(dϕ_dy).^2 )
        d_eff[2:end-1,2:end-1]  .= inn(h).^α .* (gradϕ .+ small).^(β-2)

        # rate if changes
        flux_x .= .- d_eff[2:end,:] .* max.(dϕ_dx, 0.0) .- d_eff[1:end-1,:] .* min.(dϕ_dx, 0.0)
        flux_y .= .- d_eff[:,2:end] .* max.(dϕ_dy, 0.0) .- d_eff[:,1:end-1] .* min.(dϕ_dy, 0.0)
        vo     .= (h .< 1.0) .* (1.0 .- h)
        vc     .=  h .* (0.91 .* H .- ϕ).^3

        dhdt   .= (Σ .* vo .- Γ .* vc)
        dϕdt   .= (.- (diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy) .- inn(dhdt) .+ Λ) ./ e_v

        return nothing
    end
    return ode!, ϕ0, h0, (;ϕ_, h_, x_, q_, t_, H_, Σ, Γ, Λ), H
end

ode!, ϕ0, h0, scales, H = make_ode()

const day = 24*3600
tspan = (0, 0.5day / scales.t_)
u0 = ArrayPartition(h0, ϕ0)
du0 = ArrayPartition(copy(h0), copy(ϕ0))


ode!(du0, u0, nothing, 0.0)
@time ode!(du0, u0, nothing, 0.0) # 1e-3s
@inferred ode!(du0, u0, nothing, 0.0)
## note there are a few Core.box around!
#@code_warntype ode!(du0, u0, nothing, 0.0)

prob = ODEProblem(ode!, u0, tspan)
# Note ROCK4 is an explicit alg which is good for stiff problems
# https://diffeq.sciml.ai/latest/solvers/ode_solve/#Stabilized-Explicit-Methods, https://epubs.siam.org/doi/pdf/10.1137/S1064827500379549
# ROCK4 seems to be the best of the lot.
# See also https://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/
# Time steps fo t>day:
# - 4s for tol=1e-8
# - 10s for tol=1e-7 (but solution is a bit unstable)
@time sol = solve(prob, ROCK4(), reltol=1e-8, abstol=1e-8) #, save_on=false) #, isoutofdomain=(u,p,t) -> any(u.x[1]<0));
# Note that about tol 1e-8 is needed to get a stable, non-oscillatory solution


hend = sol.u[end].x[1]*scales.h_;
ϕend = sol.u[end].x[2]*scales.ϕ_;
N = 910*9.81*H*scales.H_ - ϕend;

display(plot(heatmap(hend[2:end-1,2:end-1]),
             heatmap(ϕend[2:end-1,2:end-1])))

display(plot(plot(ϕend[2:end-1,end÷2]/1e6, xlabel="x (gridpoints)", ylabel="ϕ (MPa)"),
             plot(ϕend[2:end-1,end÷2]/scales.ϕ_, xlabel="x (gridpoints)", ylabel="ϕ ()"),
             reuse=false))

display(plot(sol.t*scales.t_/day, diff(sol.t*scales.t_), reuse=false, xlabel="t (day)", ylabel="timestep (s)"))#, yscale=:log10))
