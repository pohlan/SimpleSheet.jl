using Printf, LinearAlgebra, Statistics, Plots, Infiltrator

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
    Lx, Ly = 6e3, 1e3                  # length/width of the domain, starts at (0, 0)
    dt     = 1e-8                          # physical time step
    ttot   = 0.5day
    α      = 1.25
    β      = 1.5
    m      = 1.158e-6                           # source term for SHMIP A1 test case
    e_v    = 1                                  # void ratio for englacial storage

    # numerics
    nx, ny = 64, 32
    nout   = 100
    CN     = 0.5   # Crank-Nicolson (CN=0.5), Forward Euler (CN=0)
    # derived
    nt     = Int(ttot ÷ dt)
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-Ly/2-dy, Ly/2+dy, ny)

    # initial conditions
    ϕ0     = 100. * ones(nx, ny)
    h0     = 0.04 * ones(nx, ny) # initial fields of ϕ and h

    # surface elevation
    z_s(x, y) = 100(x+200)^(1/4) + 1/60*x - 2e10^(1/4) + 1

    # bed elevation
    para_bench = 0.05
    para_E1    = 0.05   # for SHMIP case E1
    f(x, para) = (z_s(6e3,0) - para*6e3)/6e3^2 * x^2 + para*x
    g(y) = 0.5e-6 * abs(y)^3
    r(x, para) = (-4.5*x/6e3 + 5) * (z_s(x, 0) - f(x, para)) /
                   (z_s(x, 0) - f(x, para_bench) + eps())
    z_b(x,y) = f(x, para_E1) + g(y) * r(x, para_E1)

    # ice thickness
    zb     = [0.0; ones(nx-2); 0.0] * [0.0 ones(ny-2)' 0.0] .* z_b.(xc, yc')
    H      = [0.0; ones(nx-2); 0.0] * [0.0 ones(ny-2)' 0.0] .* (z_s.(xc, yc') .- z_b.(xc, yc')) # ice thickness, rectangular ice sheet with ghostpoints
    H     .= max.(H, 0.)
    mask_H = H .> 0.
    bound_qx = diff(mask_H, dims=1) .!= 0.   # boundaries in x-direction
    bound_qy = diff(mask_H, dims=2) .!= 0.   # boundaries in y-direction

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
    zb     = zb ./ (H_*0.91)

    # array allocation
    dϕ_dx  = zeros(nx-1,ny  )
    dϕ_dy  = zeros(nx  ,ny-1)
    gradϕ  = zeros(nx-2,ny-2)
    d_eff  = zeros(nx  ,ny  )
    flux_x = zeros(nx-1,ny  )
    flux_y = zeros(nx  ,ny-1)
    div_q  = zeros(nx-2,ny-2)
    vo     = zeros(nx  ,ny  )
    vc     = zeros(nx  ,ny  )
    dϕdt   = zeros(nx-2,ny-2)
    dhdt   = zeros(nx  ,ny  )
    dϕdt_old = zeros(nx-2,ny-2)
    dhdt_old = zeros(nx  ,ny  )

    # initialise all ϕ and h fields
    ϕ0[2, :] .= zb[2, :] # Dirichlet BC
    ϕ = copy(ϕ0)
    h = copy(h0)

    # Time loop
    t = 0.
    t_sol=@elapsed for it = 1:10^4

        # d_eff
        dϕ_dx  .= diff(ϕ,dims=1) ./ dx
        dϕ_dy  .= diff(ϕ,dims=2) ./ dy

        # no flux b.c.
        dϕ_dx[bound_qx] .= 0.0;
        dϕ_dy[bound_qy] .= 0.0;

        gradϕ  .= sqrt.( av_xi(dϕ_dx).^2 .+ av_yi(dϕ_dy).^2 )
        d_eff[2:end-1,2:end-1]  .= inn(h).^α .* (gradϕ .+ small).^(β-2)
        d_eff[H .== 0.] .= 0.

        # rate if changes
        flux_x .= .- d_eff[2:end,:] .* max.(dϕ_dx, 0.0) .- d_eff[1:end-1,:] .* min.(dϕ_dx, 0.0)
        flux_y .= .- d_eff[:,2:end] .* max.(dϕ_dy, 0.0) .- d_eff[:,1:end-1] .* min.(dϕ_dy, 0.0)
        div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy
        vo     .= (h .< 1.0) .* (1.0 .- h)
        vc     .=  h .* (0.91 .* H .- (ϕ .- zb)).^3
        dhdt   .= (Σ .* vo .- Γ .* vc)
        dhdt[H .== 0.] .= 0.
        dϕdt   .= (- div_q .- inn(dhdt) .+ Λ) ./ e_v
        dϕdt[inn(H) .== 0.] .= 0.

        # timestep
        dtnum = e_v * min(dx, dy)^2 / maximum(d_eff .+ small) ./ 4.1

        # updates
        h      .=     h  .+ dtnum .* ((1-CN) .* dhdt .+ CN .* dhdt_old)
        inn(ϕ) .= inn(ϕ) .+ dtnum .* ((1-CN) .* dϕdt .+ CN .* dϕdt_old)

        h .= max.(h, 0.0)
        h[H .== 0.] .= 0.
        ϕ[H .== 0.] .= 0.

        # dirichlet boundary conditions to pw = 0
        ϕ[2, ny÷2+1] = zb[2, ny÷2+1]
        ϕ[2, ny÷2  ] = zb[2, ny÷2  ]

        # update old (Crank Nicolson)
        dϕdt_old .= dϕdt
        dhdt_old .= dhdt

        # check convergence criterion
        if (it % nout == 0) && do_monit
            # visu
            p1 = heatmap(inn(ϕ)')
            p2 = heatmap(inn(h)')
            p3 = plot(ϕ[2:end-1, end÷2], label="ϕ")
            p4 = plot(h[2:end-1, end÷2], label="h")
            display(plot(p1, p2, p3, p4))
            @printf("it %d (dt = %1.3e), max(h) = %1.3f \n", it, dtnum, maximum(inn(h)))
        end
        t += dtnum * t_
    end
    @printf("The odel stepped %1.3f s forward in time.\n", t)
    return h, ϕ, t_sol
end

do_monit = true
h, ϕ, t_sol = simple_sheet(; do_monit=do_monit)
@show t_sol

if !do_monit
    # visu
    p1 = heatmap(inn(ϕ)')
    p2 = heatmap(inn(h)')
    p3 = plot(ϕ[2:end-1, end÷2], label="ϕ")
    p4 = plot(h[2:end-1, end÷2], label="h")
    display(plot(p1, p2, p3, p4))
end
