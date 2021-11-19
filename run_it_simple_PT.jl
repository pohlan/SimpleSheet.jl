using Printf, LinearAlgebra, Statistics, Plots

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small = eps(Float64)
const day   = 24*3600

@views function simple_sheet(; do_monit=true, update_h_only = true)
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    dt     = 1e9                          # physical time step
    dt_h   = 0.5
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 0.                                  # void ratio for englacial storage

    # numerics
    nx, ny = 64, 32
    nout   = 1000
    itMax  = 10^5
    γ_h    = 0.935          # the third digit can help for saving ~2e3 iterations
    γ_ϕ    = 0.91
    tol    = 1e-6

    # derived
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)

    # initial conditions
    ϕ0     = 100. * ones(nx, ny)
    h0     = 0.04 * ones(nx, ny) # initial fields of ϕ and h
    get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1
    H      = [0.0; ones(nx-2); 0.0] * [0.0 ones(ny-2)' 0.0] .* get_H.(xc, yc') # ice thickness, rectangular ice sheet with ghostpoints
    ϕ0[2,:] .= 0.0 # Dirichlet BC

    # scaling factors
    H_     = maximum(H)        # the choice of H_ also has an impact on convergence and on optimal γ etc.; in SheetModel H_ = mean(H)
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
    dt_h   = dt_h / t_
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
    while max(err_ϕ, err_h) > tol && iter<itMax
        h .= max.(h, 0.0)

        if  err_h > 1e-3 && update_h_only # once update_h_only = false it cannot go back
            dτ_h = 1e-3
        else
            dτ_h = 9.8e-6                 # optimising it to the 1e-7 digit can save a few thousand iterations
            update_h_only = false
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
        vc     .= h .* (0.91 .* H .- ϕ).^3            # for ϕ = +/-Inf this gives NaN even if h=0
        div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy

        # residuals
        dhdt   .= Σ .* inn(vo) .- Γ .* inn(vc)
        dϕdt   .= .- div_q .- dhdt .+ Λ

        Res_ϕ  .= - e_v * (inn(ϕ) .- inn(ϕ0)) ./ dt  .+ dϕdt
        Res_h  .= - (inn(h) .- inn(h0)) ./ dt .+ dhdt

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

        # errors
        if update_h_only || iter % 1000 == 0
            err_h = norm(Res_h) / length(Res_h)
            err_ϕ = norm(Res_ϕ) / length(Res_ϕ)
            @printf("it %d, err_h = %1.2e, err_ϕ = %1.2e \n", iter, err_h, err_ϕ)

            push!(iters, iter)
            push!(errs_h, err_h)
            push!(errs_ϕ, err_ϕ)
        end

        # plot
        if (iter % nout == 0) && do_monit
            # visu
            p1 = heatmap(inn(ϕ .* ϕ_)')
            p2 = heatmap(inn(h .* h_)')
            p3 = plot(ϕ[2:end-1, end÷2] .* ϕ_, label="ϕ")
            p4 = plot(h[2:end-1, end÷2] .* h_, label="h")
            p5 = plot(abs.(Res_ϕ[:, end÷2]), label="abs(Res_ϕ)")
            p6 = plot(abs.(Res_h[:, end÷2]), label="abs(Res_h)")
            display(plot(p1, p3, p5, p2, p4, p6))
        end
    end
    return h * h_, ϕ * ϕ_, Res_ϕ, Res_h, iters, errs_h, errs_ϕ
end

do_monit = true
update_h_only = true
h, ϕ, Res_ϕ, Res_h, iters, errs_h, errs_ϕ = simple_sheet(; do_monit=do_monit, update_h_only=update_h_only)

p1 = plot(ϕ[2:end-1, end÷2], label="ϕ", xlabel="x", title="ϕ cross-sec.")
p2 = plot(h[2:end-1, end÷2], label="h", xlabel="x", title="h cross-sec.")
p3 = plot(iters, errs_ϕ, xlabel="# iterations", title="residual error", label="err_ϕ", yscale=:log10)
p4 = plot(iters, errs_h, xlabel="# iterations", title="residual error", label="err_h", yscale=:log10)
p5 = plot(abs.(Res_ϕ[:, end÷2]), label="abs(Res_ϕ)", xlabel="x", title="res cross-sec.")
p6 = plot(abs.(Res_h[:, end÷2]), label="abs(Res_h)", xlabel="x", title="res cross-sec.")
display(plot(p1, p3, p5, p2, p4, p6))