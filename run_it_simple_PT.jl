using Printf, LinearAlgebra, Statistics, Plots

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small = eps(Float64)
const day   = 24*3600

@views function simple_sheet(; do_monit=true, use_masscons_for_h)
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    dt     = 1e7                          # physical time step
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-3                                # void ratio for englacial storage
    e_v_num= 0.                                 # regularization void ratio

    # numerics
    nx, ny = 64, 32
    nout   = 1000
    itMax  = 10^4
    γ_h    = 0.8
    γ_ϕ    = 0.8

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

    # array allocation
    dϕ_dx  = zeros(nx-1,ny  )
    dϕ_dy  = zeros(nx  ,ny-1)
    gradϕ  = zeros(nx-2,ny-2)
    d_eff  = zeros(nx  ,ny  )
    flux_x = zeros(nx-1,ny  )
    flux_y = zeros(nx  ,ny-1)
    vo     = zeros(nx  ,ny  )
    vc     = zeros(nx  ,ny  )
    ux     = zeros(nx-2,ny-2)
    uy     = zeros(nx-2,ny-2)
    div_q  = zeros(nx-2,ny-2)
    Res_h  = zeros(nx-2,ny-2)
    Res_ϕ  = zeros(nx-2,ny-2)
    dhdτ   = zeros(nx-2,ny-2)
    dϕdτ   = zeros(nx-2,ny-2)
    dhdt   = zeros(nx-2,ny-2)
    dϕdt   = zeros(nx-2,ny-2)
    dτ_h   = zeros(nx-2,ny-2)
    dτ_ϕ   = zeros(nx-2,ny-2)


    # initialise all ϕ and h fields
    ϕ = copy(ϕ0)
    h = copy(h0)

    # PT iteration loop
    for it = 1:itMax
        h .= max.(h, 0.0)

        # boundary conditions
        ϕ[1:2, :] .= 0.0

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
        vc     .= h .* (0.91 .* H .- ϕ).^3            # for ϕ = +/-Inf this gives NaN even if h=0
        div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy

        # residuals
        dhdt   .= Σ .* inn(vo) .- Γ .* inn(vc)
        dϕdt   .= (.- div_q .- dhdt .+ Λ) ./ (e_v .+ e_v_num)
        dϕdt[1, :] .= 0.   # no update at ϕ Dirichlet B.C. points, important since dϕdt is used to update dhdt and no Dirichlet B.C. are imposed on h
        if use_masscons_for_h
            # eq. as in B&P but additionally with storage term
            # should be equivalent to the version below (in practice not quite)
            dhdt .= .- dϕdt .* e_v .- div_q .+ Λ
        else
            # ODE, eq. as used in GlaDS but with regularisation e_v_num
            dhdt .= dhdt .+ e_v_num.* dϕdt
        end
        Res_ϕ  .= - e_v * (inn(ϕ) .- inn(ϕ0)) ./ dt  .+ dϕdt * e_v    # without the factor e_v in both terms it produces NaNs
        Res_h  .= - (inn(h) .- inn(h0)) ./ dt .+ dhdt

        # rate of change
        dhdτ   .= Res_h .+ γ_h .* dhdτ
        dϕdτ   .= Res_ϕ .+ γ_ϕ .* dϕdτ

        # pseudo-timestep as calculated so far:
        dτ_ϕ   .= min(dx, dy)^2 ./ inn(d_eff) ./ 4.1
        dτ_h   .= 1e-6 # min(dx, dy)   ./ (max.(abs.(ux), abs.(uy)) .+ small) ./ 4.1

        # exactly as it is in Büeler & van Pelt (just slightly different)
        #dτ_ϕ   .= 0.5 .* e_v ./ inn(d_eff) .* (1/dx^2 + 1/dy^2 + small)^(-1)
        #dτ_h   .= 0.5 .* min.((abs.(ux)./dx .+ abs.(uy)./dy .+ small).^(-1), dt)

        # updates
        inn(h) .= inn(h)  .+ dτ_h .* dhdτ
        inn(ϕ) .= inn(ϕ)  .+ dτ_ϕ .* dϕdτ

        # plot
        if (it % nout == 0) && do_monit
            # visu
            p1 = heatmap(inn(ϕ)')
            p2 = heatmap(inn(h)')
            display(plot(p1, p2))
            err_h = norm(Res_h) ./ length(Res_h)
            err_ϕ = norm(Res_ϕ) ./ length(Res_ϕ)
            @printf("it %d, err_h = %1.2e, err_ϕ = %1.2e \n", it, err_h, err_ϕ)
        end
    end
    return h, ϕ
end

do_monit = true
use_masscons_for_h = true
h, ϕ = simple_sheet(; do_monit=do_monit, use_masscons_for_h=use_masscons_for_h)

if !do_monit
    # visu
    p1 = heatmap(inn(ϕ)')
    p2 = heatmap(inn(h)')
    display(plot(p1, p2))
end
