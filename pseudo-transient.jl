using Printf, LinearAlgebra, Statistics
import Plots; Plt = Plots

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small = eps(Float64)
const day   = 24*3600

@views function simple_sheet(;  nx, ny,                 # grid size
                                itMax=10^7,             # maximal number of iterations
                                tol=1e-6,               # tolerance, convergence criterion
                                γ=0.8,                  # damping parameter (γ_h = γ_ϕ)
                                dτ_h_=1.6e-5,             # pseudo-time step for h
                                do_monit=false,         # enable/disable plotting of intermediate results
                                e_v_num=0,              # regularisation void ratio
                                update_h_only=false     # true: split step scheme, only update h in the beginning
                                )

    @printf("Running for tol = %1.e. \n", tol)

    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    dt     = 8000day                      # physical time step
    dt_h   = 0.5
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-6                               # void ratio for englacial storage
    ev_num_kw = e_v_num

    # numerics
    nout   = 1000
    γ_h    = γ
    γ_ϕ    = γ

    # derived
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
    H_     = mean(H)        # the choice of H_ also has an impact on convergence and on optimal γ etc.; in SheetModel H_ = mean(H)
    ϕ_     = 9.8 * 910 * H_
    h_     = 0.1
    x_     = max(Lx, Ly)
    q_     = 0.005 * h_^α * (ϕ_ / x_)^(β-1)
    t_     = h_ * x_ / q_
    Ψ      = max(e_v + e_v_num, small) * ϕ_ / (1000 * 9.8 * h_)
    Σ      = 5e-8 * x_ / q_
    Γ      = 3.375e-25 * ϕ_^3 * x_ / q_ * 2/27  # the last bit is 2/n^n from vc
    Λ      = m * x_ / q_

    # apply the scaling and convert to correct data type
    ϕ0     = ϕ0 ./ ϕ_
    h0     = h0 ./ h_
    dx     = dx / x_
    dy     = dy / x_
    dt     = dt / t_
    dt_h   = dt_h / t_
    H      = H ./ H_

    # array allocation
    dϕ_dx   = zeros(nx-1,ny  )
    dϕ_dy   = zeros(nx  ,ny-1)
    gradϕ   = zeros(nx-2,ny-2)
    d_eff   = zeros(nx  ,ny  )
    flux_x  = zeros(nx-1,ny  )
    flux_y  = zeros(nx  ,ny-1)
    vo      = zeros(nx  ,ny  )
    vc      = zeros(nx  ,ny  )
    ux      = zeros(nx-2,ny-2)
    uy      = zeros(nx-2,ny-2)
    div_q   = zeros(nx-2,ny-2)
    Res_h   = zeros(nx-2,ny-2)
    Res_ϕ   = zeros(nx-2,ny-2)
    dhdτ    = zeros(nx-2,ny-2)
    dϕdτ    = zeros(nx-2,ny-2)
    dhdt    = zeros(nx-2,ny-2)
    dϕdt_ev = zeros(nx-2,ny-2)
    dτ_h    = zeros(nx-2,ny-2)
    dτ_ϕ    = zeros(nx-2,ny-2)

    iters  = []
    errs_ϕ = []
    errs_h = []

    # initialise all ϕ and h fields
    ϕ = copy(ϕ0)
    h = copy(h0)

    err_ϕ = 1.
    err_h = 1.
    iter  = 0.

    # PT iteration loop
    t_sol = @elapsed while iter<itMax && max(err_ϕ, err_h) > tol&& !any(isnan.([err_ϕ, err_h]))
        h .= max.(h, 0.0)

        if  iter < 100 && update_h_only # once update_h_only = false it cannot go back
            dτ_h = 1e-3
            e_v_num=0
        else
            dτ_h = dτ_h_
            update_h_only = false
            e_v_num = ev_num_kw
        end

        # boundary conditions
        ϕ[2, :] .= 0.0

        # d_eff
        dϕ_dx  .= diff(ϕ,dims=1) ./ dx
        dϕ_dy  .= diff(ϕ,dims=2) ./ dy

        dϕ_dx[1,:] .= 0.0; dϕ_dx[end,:] .= 0.0
        dϕ_dy[:,1] .= 0.0; dϕ_dy[:,end] .= 0.0

        gradϕ  .= sqrt.( av_xi(dϕ_dx).^2 .+ av_yi(dϕ_dy).^2 )
        inn(d_eff)  .= inn(h).^α .* (gradϕ .+ small).^(β-2)

        # fluxes and size evolution terms
        flux_x .= .- d_eff[2:end,:] .* max.(dϕ_dx, 0.0) .- d_eff[1:end-1,:] .* min.(dϕ_dx, 0.0)
        flux_y .= .- d_eff[:,2:end] .* max.(dϕ_dy, 0.0) .- d_eff[:,1:end-1] .* min.(dϕ_dy, 0.0)
        ux     .= av_xi(flux_x) ./ (inn(h) .+ small)  # dividing by zero gives NaN
        uy     .= av_yi(flux_y) ./ (inn(h) .+ small)
        vo     .= (h .< 1.0) .* (1.0 .- h)
        vc     .= h .* (H .- ϕ).^3            # for ϕ = +/-Inf this gives NaN even if h=0
        div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy

        # residuals
        dhdt         .= Σ .* inn(vo) .- Γ .* inn(vc)
        dϕdt_ev      .= (.- div_q .- dhdt .+ Λ)           # note: this is dϕdt * (e_v + e_vnum); for the case e_v == e_v_num == 0 it is more convenient this way
        dϕdt_ev[1,:] .= 0.                                # Dirichlet B.C. points, important for next line if e_v_num > 0
        dhdt        .+= e_v_num / Ψ .* dϕdt_ev

        # Res_ϕ  .= - Ψ * (inn(ϕ) .- inn(ϕ0)) ./ dt  .+ dϕdt_ev
        # Res_h  .= - (inn(h) .- inn(h0)) ./ dt .+ dhdt

        # solve for steady-state directly
        Res_ϕ  .= dϕdt_ev
        Res_h  .= dhdt

        Res_ϕ[1, :] .= 0.      # Dirichlet B.C. points, no update

        # rate of change
        dhdτ   .= Res_h .+ γ_h .* dhdτ
        dϕdτ   .= Res_ϕ .+ γ_ϕ .* dϕdτ

        # pseudo-timestep
        dτ_ϕ   .= min(dx, dy)^2 ./ inn(d_eff) ./ 4.1

        # updates
        if update_h_only
            # inn(h) .= inn(h)  .+ dt_h * dhdt    # explicit
             inn(h) .= inn(h)  .+ dτ_h .* dhdτ # PT
        else
            inn(ϕ) .= inn(ϕ)  .+ dτ_ϕ .* dϕdτ
            inn(h) .= inn(h)  .+ dτ_h .* dhdτ
        end

        iter += 1

        # errors and plot
        if iter % nout == 0 || update_h_only

            err_h = norm(Res_h) / sqrt(length(Res_h))
            err_ϕ = norm(Res_ϕ) / sqrt(length(Res_ϕ))
            push!(iters, iter)
            push!(errs_h, err_h)
            push!(errs_ϕ, err_ϕ)

            if do_monit
                @printf("it %d, err_h = %1.2e, err_ϕ = %1.2e \n", iter, err_h, err_ϕ)
                p1 = Plt.plot(ϕ[2:end-1, end÷2] .*ϕ_, label="ϕ", xlabel="x", title="ϕ cross-sec.")
                p2 = Plt.plot(h[2:end-1, end÷2] .*h_, label="h", xlabel="x", title="h cross-sec.")
                p3 = Plt.plot(abs.(Res_ϕ[:, end÷2]), label="abs(Res_ϕ)", xlabel="x", title="res cross-sec.")
                p4 = Plt.plot(abs.(Res_h[:, end÷2]), label="abs(Res_h)", xlabel="x", title="res cross-sec.")
                if max(err_ϕ, err_h) > tol && iter<itMax
                    Plt.display(Plt.plot(p1, p3, p2, p4))
                else
                    p5 = Plt.plot(iters, errs_ϕ, xlabel="# iterations", title="residual error", label="err_ϕ", yscale=:log10)
                    p6 = Plt.plot(iters, errs_h, xlabel="# iterations", title="residual error", label="err_h", yscale=:log10)
                    Plt.display(Plt.plot(p1, p3, p5, p2, p4, p6))
                end
            end
        end
    end

    #p1 = Plt.plot(iters, errs_ϕ, xlabel="# iterations", title="residual error", label="err_ϕ", yscale=:log10)
    #p2 = Plt.plot(iters, errs_h, xlabel="# iterations", title="residual error", label="err_h", yscale=:log10)
    #Plt.display(Plt.plot(p1,p2))

    return ϕ * ϕ_, h * h_, iter, t_sol
end

# ϕ, h, iter, t_sol = simple_sheet(; nx=64, ny=32, e_v_num=0., update_h_only=true,  γ=0.8, dτ_h_=1.6e-5, itMax=2*10^4, do_monit=true)
