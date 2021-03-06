using Printf, LinearAlgebra, Statistics, Infiltrator
import Plots; Plt = Plots

@views   inn(A) = A[2:end-1,2:end-1]
@views av_xa(A) = (0.5  .* (A[1:end-1,:] .+ A[2:end,:]))
@views av_ya(A) = (0.5  .* (A[:,1:end-1] .+ A[:,2:end]))
@views av_xi(A) = (0.5  .* (A[1:end-1,2:end-1] .+ A[2:end,2:end-1]))
@views av_yi(A) = (0.5  .* (A[2:end-1,1:end-1] .+ A[2:end-1,2:end]))
@views    av(A) = (0.25 .* (A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] .+ A[2:end,1:end-1]))

const small = eps(Float64)
const day   = 24*3600

@views function simple_sheet(;  nx, ny,          # grid size
                                itMax=10^7,      # maximal number of iterations
                                dt=1e-3,         # physical time step, fixed
                                do_monit=false,  # enable/disable plotting of intermediate results
                                e_v_num=0,       # regularisation void ratio
                                use_CFL=false,   # true: use CFL criterion for dt, false: use fixed dt=1s
                                CN=0             # Crank-Nicolson (CN=0.5), Forward Euler (CN=0)
                                )

    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    ttot   = 8000day
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                     # source term for SHMIP A1 test case
    e_v    = 1e-6                         # void ratio for englacial storage

    # numerics
    nout   = 10^3

    # derived
    nt     = min(Int(ttot ÷ dt), itMax)
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

    # apply the scaling and convert to correct data type
    ϕ0     = ϕ0 ./ ϕ_
    h0     = h0 ./ h_
    dx     = dx / x_
    dy     = dy / x_
    dt     = dt / t_
    ttot   = ttot / t_
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

    t  = 0.
    it = 0

    # Time loop
    @printf("Running for e_v_num = %1.e \n", e_v_num)
    t_sol=@elapsed while t<ttot && it<nt

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

        # rate of change
        flux_x .= .- d_eff[2:end,:] .* max.(dϕ_dx, 0.0) .- d_eff[1:end-1,:] .* min.(dϕ_dx, 0.0)
        flux_y .= .- d_eff[:,2:end] .* max.(dϕ_dy, 0.0) .- d_eff[:,1:end-1] .* min.(dϕ_dy, 0.0)
        vo     .= (h .< 1.0) .* (1.0 .- h)
        vc     .=  h .* (H .- ϕ).^3
        div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy
        # ux     .= av_xi(flux_x) ./ inn(h)
        # uy     .= av_yi(flux_y) ./ inn(h)

        dhdt      .= Σ .* inn(vo) .- Γ .* inn(vc)
        dϕdt      .= (- div_q .- dhdt .+ Λ) ./ Ψ
        dϕdt[1,:] .= 0                    # Dirichlet B.C. points
        dhdt     .+= e_v_num .* dϕdt

        # timestep
        # dt_ϕ = e_v .* min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1
        # dt_h = min(dx, dy) ./ (max(maximum(ux), maximum(uy)) + small) ./ 4.1   # is always smaller than dt_ϕ
        # dt = min(dt_ϕ, dt_h)

        if use_CFL
            dt = Ψ .* min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1   # time step much smaller than dt=1s
        end

        # updates
        inn(h) .= inn(h) .+ dt .* ((1-CN) .* dhdt .+ CN .* dhdt_old)
        inn(ϕ) .= inn(ϕ) .+ dt .* ((1-CN) .* dϕdt .+ CN .* dϕdt_old)

        # update old (Crank Nicolson)
        dϕdt_old .= dϕdt
        dhdt_old .= dhdt

        t  += dt
        it += 1

        # check convergence criterion
        if (it % nout == 0) && do_monit
            # visu
            p1 = Plt.plot(ϕ[2:end-1,end÷2] .* ϕ_)
            p2 = Plt.plot(h[2:end-1,end÷2] .* h_)
            Plt.display(Plt.plot(p1, p2))
            @printf("it %d, dt = %1.3e s, t = %1.3f day \n", it, dt*t_, t*t_/day)
        end

    end
    return ϕ .* ϕ_, h .* h_, t_sol
end

# ϕ, h, t_sol = simple_sheet(; nx=64, ny=32, use_CFL=true, e_v_num=10, do_monit=true, itMax=10^5)
