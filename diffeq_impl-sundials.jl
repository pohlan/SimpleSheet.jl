# Solving the sheet using OrdinaryDiffEq solvers (ROCK4)
#
# TODO:
# - try the mass-conservation equation used by B&P as well
# - try on the GPU

# - also try implicit solvers, but that will be a lot more work

using Printf, LinearAlgebra, Statistics, Plots, Test, RecursiveArrayTools, OrdinaryDiffEq,
    Infiltrator, ForwardDiff
using DiffEqBase: dualcache, get_tmp
pyplot()

const gt = get_tmp
@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small    = eps(Float64)
get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1

function make_ode_reg(; use_masscons_for_h=false)
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-3                               # void ratio for englacial storage
    e_v_num= 0*1e-2                               # regularization void ratio

    # numerics
    nx, ny = 64, 32
    # nx *= 2
    # ny *= 2

    # derived
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)

    # initial conditions
    ϕ0     = 100. * ones(nx, ny)
    h0     = 0.04 * ones(nx, ny) # initial fields of ϕ and h
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
    H      = H ./ H_

    # array allocationsize(Λ)
    chunksize = ForwardDiff.pickchunksize(nx*ny)

    dϕ_dx_  = dualcache(zeros(nx-1,ny  ), chunksize)
    dϕ_dy_  = dualcache(zeros(nx  ,ny-1), chunksize)
    gradϕ_  = dualcache(zeros(nx-2,ny-2), chunksize)
    d_eff_  = dualcache(zeros(nx  ,ny  ), chunksize)
    flux_x_ = dualcache(zeros(nx-1,ny  ), chunksize)
    flux_y_ = dualcache(zeros(nx  ,ny-1), chunksize)
    div_q_  = dualcache(zeros(nx-2,ny-2), chunksize)
    vo_     = dualcache(zeros(nx  ,ny  ), chunksize)
    vc_     = dualcache(zeros(nx  ,ny  ), chunksize)


    @assert e_v + e_v_num > 0

    ode! = let H=H, dx=dx, dy=dy
        @views function (du,u,p,t)
            # caches
            dϕ_dx  = gt(dϕ_dx_, u)
            dϕ_dy  = gt(dϕ_dy_, u)
            gradϕ  = gt(gradϕ_, u)
            d_eff  = gt(d_eff_, u)
            flux_x = gt(flux_x_, u)
            flux_y = gt(flux_y_, u)
            div_q  = gt(div_q_, u)
            vo     = gt(vo_, u)
            vc     = gt(vc_, u)

            h, ϕ = reshape(u[1:end÷2], nx, ny), reshape(u[end÷2+1:end], nx, ny)
            ## to avoid errors from Complex numbers:
            h .= max.(h,0)
            # alternative:
            #h .= abs.(h)

            # dirichlet boundary conditions to pw = 0
            ϕ[1:2,:] .= 0.0

            # dhdt = du.x[1]
            # dϕdt = inn(du.x[2])
            dhdt, dϕdt = reshape(du[1:end÷2], nx, ny), reshape(du[end÷2+1:end],nx,ny)
            dϕdt[1,:] .= 0.0; dϕdt[end,:] .= 0.0
            dϕdt[:,1] .= 0.0; dϕdt[:,end] .= 0.0

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
            vo     .= (h .< 1.0) .* (1.0 .- h)
            vc     .=  h .* (0.91 .* H .- ϕ).^3

            dhdt   .= (Σ .* vo .- Γ .* vc)
            div_q  .= (diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy)
            inn(dϕdt)   .= (.- div_q .- inn(dhdt) .+ Λ) ./ (e_v .+ e_v_num)

            # update taking only non-regularization e_v into account
            if use_masscons_for_h
                # NOTE: above two are identical (modulus floating point errors)
                inn(dhdt) .= .-e_v.*inn(dϕdt) .- div_q .+ Λ
            else
                inn(dhdt) .= inn(dhdt) .+ e_v_num.*inn(dϕdt)
            end

            return nothing
        end
    end
    return ode!, copy(ϕ0), copy(h0), (;ϕ_, h_, x_, q_, t_, H_, Σ, Γ, Λ), H, chunksize
end

ode!, ϕ0, h0, scales, H, chunksize = make_ode_reg(;use_masscons_for_h=false)

const day = 24*3600
tspan = (0, 5day / scales.t_)

# Sundials does not work with ArrayPartition:
u0 = ArrayPartition(h0, ϕ0)[:]
du0 = ArrayPartition(copy(h0), copy(ϕ0))[:]

ode!(du0, u0, nothing, 0.0)
@time ode!(du0, u0, nothing, 0.0) # 1e-3s
@inferred ode!(du0, u0, nothing, 0.0)
## note there are a few Core.box around!
#@code_warntype ode!(du0, u0, nothing, 0.0)

prob = ODEProblem(ode!, u0, tspan);

using Sundials
@time sol = solve(prob, CVODE_BDF(linear_solver=:GMRES));

# ##############################################################
# if length(sol)>1
#     hend = sol.u[end].x[1]*scales.h_;
#     ϕend = sol.u[end].x[2]*scales.ϕ_;
#     N = 910*9.81*H*scales.H_ - ϕend;

#     display(plot(heatmap(inn(hend')),
#                  heatmap(inn(ϕend'))))

#     display(plot(plot(ϕend[2:end-1,end÷2]/1e6, xlabel="x (gridpoints)", ylabel="ϕ (MPa)"),
#                  plot(ϕend[2:end-1,end÷2]/scales.ϕ_, xlabel="x (gridpoints)", ylabel="ϕ ()"),
#                  plot(hend[2:end-1,end÷2], xlabel="x (gridpoints)", ylabel="h (m)"),
#                  plot(hend[2:end-1,end÷2]/scales.h_, xlabel="x (gridpoints)", ylabel="h ()"),
#                  layout=(2,2), reuse=false))

#     display(plot(sol.t*scales.t_/day, diff(sol.t*scales.t_), reuse=false, xlabel="t (day)", ylabel="timestep (s)"))#, yscale=:log10))
# end
