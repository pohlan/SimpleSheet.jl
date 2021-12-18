# Solving the sheet using OrdinaryDiffEq solvers (ROCK4)
#
# TODO:
# - try on the GPU

using Printf, LinearAlgebra, Statistics, Test, RecursiveArrayTools, OrdinaryDiffEq,
    Infiltrator
import Plots; Plt=Plots
Plt.pyplot()

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small    = eps(Float64)
const day      = 3600*24

function make_ode_reg(; nx, ny,                 # grid size
                        e_v_num=0               # regularization void ratio
                        )
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-6                                 # void ratio for englacial storage

    # numerics
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)

    # ice thickness
    H           = zeros(nx, ny)
    get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1
    inn(H)     .= inn(get_H.(xc, yc'))

    # initial conditions
    ϕ0       = zeros(nx, ny)
    inn(ϕ0) .= 100.
    ϕ0[2,:] .= 0.0 # Dirichlet BC

    h0       = zeros(nx, ny)
    inn(h0) .= 0.04

    # scaling factors
    H_     = mean(H)
    ϕ_     = 9.8 * 910 * H_
    h_     = 0.1
    x_     = max(Lx, Ly)
    q_     = 0.005 * h_^α * (ϕ_ / x_)^(β-1)
    t_     = h_ * x_ / q_
    Ψ      = max(e_v .+ e_v_num, small) * ϕ_ / (1000 * 9.8 * h_)
    Σ      = 5e-8 * x_ / q_
    Γ      = 3.375e-25 * ϕ_^3 * x_ / q_ * 2/27  # the last bit is 2/n^n from vc
    Λ      = m * x_ / q_

    # apply the scaling
    ϕ0     = ϕ0 ./ ϕ_
    h0     = h0 ./ h_
    dx     = dx / x_
    dy     = dy / x_
    H      = H ./ H_

    # array allocationsize(Λ)
    dϕ_dx  = zeros(nx-1,ny  )
    dϕ_dy  = zeros(nx  ,ny-1)
    gradϕ  = zeros(nx-2,ny-2)
    d_eff  = zeros(nx  ,ny  )
    flux_x = zeros(nx-1,ny  )
    flux_y = zeros(nx  ,ny-1)
    div_q  = zeros(nx-2,ny-2)
    vo     = zeros(nx  ,ny  )
    vc     = zeros(nx  ,ny  )

    # initialize all ϕ and h fields
    ϕ_old = copy(ϕ0)
    h_old = copy(h0)

    ode! = let H=H,dx=dx,dy=dy
        @views function (du,u,p,t)
            h, ϕ = u.x
            ## to avoid errors from Complex numbers:
            h .= max.(h,0)
            # alternative:
            #h .= abs.(h)

            dhdt = du.x[1]
            dϕdt = du.x[2]

            # d_eff
            dϕ_dx  .= diff(ϕ, dims=1) ./ dx
            dϕ_dy  .= diff(ϕ, dims=2) ./ dy

            dϕ_dx[1,:] .= 0.0; dϕ_dx[end,:] .= 0.0
            dϕ_dy[:,1] .= 0.0; dϕ_dy[:,end] .= 0.0

            gradϕ  .= sqrt.( av_xi(dϕ_dx).^2 .+ av_yi(dϕ_dy).^2 )
            inn(d_eff)  .= inn(h).^α .* (gradϕ .+ small).^(β-2)

            # rate of changes
            flux_x .= .- d_eff[2:end,:] .* max.(dϕ_dx, 0.0) .- d_eff[1:end-1,:] .* min.(dϕ_dx, 0.0)
            flux_y .= .- d_eff[:,2:end] .* max.(dϕ_dy, 0.0) .- d_eff[:,1:end-1] .* min.(dϕ_dy, 0.0)
            div_q  .= (diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy)
            vo     .= (h .< 1.0) .* (1.0 .- h)
            vc     .=  h .* (H .- ϕ).^3

            dhdt      .= (Σ .* vo .- Γ .* vc)
            inn(dϕdt) .= (.- div_q .- inn(dhdt) .+ Λ) ./ Ψ
            dϕdt[2,:] .= 0 # BCs, fixes https://github.com/pohlan/SimpleSheet.jl/pull/4#issue-1041245216
            dhdt     .+= e_v_num.*dϕdt

            # ghost point boundaries
            dϕdt[[1,end],:] .= 0.0
            dϕdt[:,[1,end]] .= 0.0
            dhdt[[1,end],:] .= 0.0
            dhdt[:,[1,end]] .= 0.0

            return nothing
        end
    end
    return ode!, copy(ϕ0), copy(h0), (;ϕ_, h_, x_, q_, t_, H_, Σ, Γ, Λ), H
end

function simple_sheet(; nx, ny, tol=1e-8, e_v_num=0, do_plots=false, itMax=10^6)
    @printf("Running for e_v_num = %1.e \n", e_v_num)
    ode!, ϕ0, h0, scales, H = make_ode_reg(; nx, ny, e_v_num)
    tspan = (0, 8000day / scales.t_)
    u0 = ArrayPartition(h0, ϕ0)
    du0 = ArrayPartition(copy(h0), copy(ϕ0))

    ode!(du0, u0, nothing, 0.0)
    # @time ode!(du0, u0, nothing, 0.0) # 1e-3s
    # @inferred ode!(du0, u0, nothing, 0.0)
    ## note there are a few Core.box around!
    #@code_warntype ode!(du0, u0, nothing, 0.0)

    prob = ODEProblem(ode!, u0, tspan);
    # Note ROCK4 is an explicit alg which is good for stiff problems
    # https://diffeq.sciml.ai/latest/solvers/ode_solve/#Stabilized-Explicit-Methods, https://epubs.siam.org/doi/pdf/10.1137/S1064827500379549
    # ROCK4 seems to be the best of the lot.
    # See also https://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/
    # Time steps fo t>day:
    # - 4s for tol=1e-8
    # - 10s for tol=1e-7 (but solution is a bit unstable)
    # Note that about tol 1e-8 is needed to get a stable, non-oscillatory solution
    tic = Base.time()
    sol = solve(prob, ROCK4(), reltol=tol, save_everystep=true, abstol=tol, isoutofdomain=(u,p,t) -> any(u.x[1].<0), maxiters=itMax); #, dtmax=150/scales.t_) #, save_on=false) #, isoutofdomain=(u,p,t) -> any(u.x[1]<0));
    toc = Base.time() - tic

    h = sol.u[end].x[1]*scales.h_;
    ϕ = sol.u[end].x[2]*scales.ϕ_;1

    if do_plots
        Plt.display(Plt.plot(Plt.heatmap(inn(h')),
                     Plt.heatmap(inn(ϕ'))))

        Plt.display(Plt.plot(Plt.plot(ϕ[2:end-1,end÷2]/1e6, xlabel="x (gridpoints)", ylabel="ϕ (MPa)"),
                     Plt.plot(ϕ[2:end-1,end÷2]/scales.ϕ_, xlabel="x (gridpoints)", ylabel="ϕ ()"),
                     Plt.plot(h[2:end-1,end÷2], xlabel="x (gridpoints)", ylabel="h (m)"),
                     Plt.plot(h[2:end-1,end÷2]/scales.h_, xlabel="x (gridpoints)", ylabel="h ()"),
                     layout=(2,2), reuse=false))

        Plt.display(Plt.plot(sol.t*scales.t_/day, diff(sol.t*scales.t_), reuse=false, xlabel="t (day)", ylabel="timestep (s)"))#, yscale=:log10))
    end

    return ϕ, h, toc
end

# ϕ, h, toc = simple_sheet(nx=64, ny=32, tol=1e-8, e_v_num=1e-3, do_plots=true)
