using Printf, LinearAlgebra, Statistics, Plots

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small = eps(Float64)
const day   = 24*3600

@views function simple_sheet(; do_monit=true)
    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    dt     = 1.0                          # physical time step
    ttot   = 0.5day
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1.                                 # void ratio for englacial storage

    # numerics
    nx, ny = 64, 32
    nout   = 1e4
    CN     = 0. # Crank-Nicolson (CN=0.5), Forward Euler (CN=0)
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
    div_q  = zeros(nx-2,ny-2)
    ux     = zeros(nx-2,ny-2)
    uy     = zeros(nx-2,ny-2)
    dϕdt   = zeros(nx-2,ny-2)
    dhdt   = zeros(nx-2,ny-2)
    dϕdt_old = zeros(nx-2,ny-2)
    dhdt_old = zeros(nx-2,ny-2)

    # initialise all ϕ and h fields
    ϕ = copy(ϕ0)
    h = copy(h0)

    # Time loop
    println("Running nt = $nt time steps (dt = $(dt*t_) sec.)")
    t_sol=@elapsed for it = 1:nt

        h .= max.(h, 0.0)

        # dirichlet boundary conditions to pw = 0
        ϕ[1:2,:] .= 0.0

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
        div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy
        ux     .= av_xi(flux_x) ./ inn(h)
        uy     .= av_yi(flux_y) ./ inn(h)
        dhdt   .= .- div_q .+ Λ
        dϕdt   .= 1. / e_v * (- div_q .- (Σ .* inn(vo) .- Γ .* inn(vc)) .+ Λ)

        # timestep
        # dt_ϕ = e_v .* min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1
        # dt_h = min(dx, dy) ./ (max(maximum(ux), maximum(uy)) + small) ./ 4.1
        # dt = min(dt_ϕ, dt_h)

        # updates
        h[2:end-1,2:end-1] .= inn(h) .+ dt .* ((1-CN) .* dhdt .+ CN .* dhdt_old)
        ϕ[2:end-1,2:end-1] .= inn(ϕ) .+ dt .* ((1-CN) .* dϕdt .+ CN .* dϕdt_old)

        # update old (Crank Nicolson)
        dϕdt_old .= dϕdt
        dhdt_old .= dhdt

        # check convergence criterion
        if (it % nout == 0) && do_monit
            # visu
            p1 = heatmap(inn(ϕ)')
            p2 = heatmap(inn(h)')
            display(plot(p1, p2))
            @printf("it %d (dt = %1.3e), max(h) = %1.3f \n", it, dt, maximum(inn(h)))
        end
    end
    return h, ϕ, t_sol
end

do_monit = true
h, ϕ, t_sol = simple_sheet(; do_monit=do_monit)
@show t_sol

if !do_monit
    # visu
    p1 = heatmap(inn(ϕ)')
    p2 = heatmap(inn(h)')
    display(plot(p1, p2))
end
