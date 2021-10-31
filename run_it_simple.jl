using Printf, LinearAlgebra, Statistics, Plots

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small = eps(Float64)
const day   = 24*3600
plot_out    = true
plot_err    = true

@views function simple_sheet()
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    dt     = 10.0                          # physical time step
    ttot   = 0.5day
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1.0#1e-3                               # void ratio for englacial storage

    # numerics
    nx, ny = 64, 32
    nout   = 1e4
    # derived
    nt     = Int(ttot ÷ dt)
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)

    # initial conditions
    ϕ0     = 100. * ones(nx, ny)
    h0     = 0.04 * ones(nx, ny) # initial fields of ϕ and h
    get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1
    H      = [0.0; ones(nx-2); 0.0] * [0.0 ones(ny-2)' 0.0] .* get_H.(xc, yc') # ice thickness, rectangular ice sheet with ghostpoints
    ϕ0[1:2,:] .= 0.0 # Dirichlet BC

    # scaling factors
    ϕ_     = 9.81 * 910 * mean(H)
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
    H      = H ./ mean(H)

    # array allocation
    dϕ_dx  = zeros(nx-1,ny  )
    dϕ_dy  = zeros(nx  ,ny-1)
    gradϕ  = zeros(nx-2,ny-2)
    d_eff  = zeros(nx  ,ny  )
    flux_x = zeros(nx-1,ny  )
    flux_y = zeros(nx  ,ny-1)
    vo     = zeros(nx  ,ny  )
    vc     = zeros(nx  ,ny  )
    dϕdt   = zeros(nx-2,ny-2)
    dhdt   = zeros(nx  ,ny  )
    dϕdt_old = zeros(nx-2,ny-2)
    dhdt_old = zeros(nx  ,ny  )

    # initialise all ϕ and h fields
    ϕ = copy(ϕ0)
    h = copy(h0)

    # Time loop
    println("Running nt = $nt time steps (dt = $(dt*t_) sec.)")
    for it = 1:nt

        # d_eff
        dϕ_dx  .= diff(ϕ,dims=1) ./ dx
        dϕ_dy  .= diff(ϕ,dims=2) ./ dy

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
        dϕdt   .= .- (diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy) .- inn(dhdt) .+ Λ

        # timestep
        # dtnum = e_v .* min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1

        # updates
        ϕ[2:end-1,2:end-1] .=  inn(ϕ) .+ dt ./ e_v .* (0.5 .* dϕdt .+ 0.5 .* dϕdt_old)
        h                  .= max.(h  .+ dt        .* (0.5 .* dhdt .+ 0.5 .* dhdt_old), 0.0)

        # dirichlet boundary conditions to pw = 0
        ϕ[1:2,:] .= 0.0

        # update old
        dϕdt_old .= dϕdt
        dhdt_old .= dhdt

        # check convergence criterion
        if it % nout == 0
            # visu
            p1 = heatmap(inn(ϕ)')
            p2 = heatmap(inn(h)')
            display(plot(p1, p2))
            @printf("it %d (dt = %1.3e), max(h) = %1.3f \n", it, dt, maximum(h))
        end
    end
    return
end

simple_sheet()
