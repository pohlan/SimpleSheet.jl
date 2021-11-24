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

@views function compute_resid(Ki_h, Ki_ϕ, h, ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, e_v, e_v_num, dx, dy, set_h_bc, h_bc)

    h .= max.(h, 0.0)

    # dirichlet boundary conditions to pw = 0
    ϕ[2,:] .= 0.0
    if set_h_bc
        h[2,:] .= h_bc
    end

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
    div_q  .= diff(flux_x[:,2:end-1],dims=1) ./ dx .+ diff(flux_y[2:end-1,:],dims=2) ./ dy
    vo     .= (h .< 1.0) .* (1.0 .- h)
    vc     .=  h .* (0.91 .* H .- ϕ).^3

    Ki_h      .= Σ .* inn(vo) .- Γ .* inn(vc)
    Ki_ϕ      .= (- div_q .- Ki_h .+ Λ) ./ max(e_v + e_v_num, small)
    Ki_h     .+= e_v_num .* Ki_ϕ

    Ki_ϕ[1,:] .= 0.
    if set_h_bc
        Ki_h[1,:] .= h_bc
    end
    return
end

@views function cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, Ki_h, Ki_ϕ, fac)
    inn(Tmp_h) .= inn(h) .+ fac .* Ki_h
    inn(Tmp_ϕ) .= inn(ϕ) .+ fac .* Ki_ϕ
    return
end

@views function update_RK4!(h, ϕ, K1_h, K1_ϕ, K2_h, K2_ϕ, K3_h, K3_ϕ, K4_h, K4_ϕ, dt, set_h_bc, h_bc)
    inn(h) .= inn(h) .+ dt .* (K1_h .+ 2.0 .* K2_h .+ 2.0 .* K3_h .+ K4_h) ./ 6.0
    inn(ϕ) .= inn(ϕ) .+ dt .* (K1_ϕ .+ 2.0 .* K2_ϕ .+ 2.0 .* K3_ϕ .+ K4_ϕ) ./ 6.0

    h .= max.(h, 0.0)
    # dirichlet boundary conditions to pw = 0
    ϕ[2,:] .= 0.0
    if set_h_bc
        h[2,:] .= h_bc
    end
    return
end

@views function simple_sheet(;  nx, ny,          # grid size
                                itMax,           # maximal number of iterations
                                dt=1e-3,         # physical time step, fixed
                                do_monit=false,   # enable/disable plotting of intermediate results
                                set_h_bc=false,  # whether to set dirichlet bc for h (at the nodes where ϕ d. bc are set)
                                                 # note: false is only applied if e_v_num = 0, otherwise bc are required
                                e_v_num=0        # regularisation void ratio
    )

    # physics
    Lx, Ly = 100e3, 20e3                  # length/width of the domain, starts at (0, 0)
    ttot   = 1e9
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-6                               # void ratio for englacial storage

    # numerics
    nout   = 1000
    nt     = min(Int(ttot ÷ dt), itMax)
    dx, dy = Lx / (nx-3), Ly / (ny-3)     # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)
    if (e_v_num > 0.) set_h_bc=true end

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

    h_bc     = (Γ/Σ * (0.91 .* H[2,2] - ϕ0[2,2]).^3 .+ 1.).^(-1)   # solution for h if dhdt=0 (Σ vo = Γ vc)
    if set_h_bc
        h0[2,:] .= h_bc
    end

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
    K1_ϕ   = zeros(nx-2,ny-2)
    K2_ϕ   = zeros(nx-2,ny-2)
    K3_ϕ   = zeros(nx-2,ny-2)
    K4_ϕ   = zeros(nx-2,ny-2)
    Tmp_ϕ  = zeros(nx  ,ny  )
    K1_h   = zeros(nx-2,ny-2)
    K2_h   = zeros(nx-2,ny-2)
    K3_h   = zeros(nx-2,ny-2)
    K4_h   = zeros(nx-2,ny-2)
    Tmp_h  = zeros(nx  ,ny  )

    # initialise all ϕ and h fields
    ϕ = copy(ϕ0)
    h = copy(h0)

    # Time loop
    println("Running $nt iterations")
    t_sol=@elapsed for it = 1:nt

        # timestep
        # dt = min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1

        compute_resid(K1_h, K1_ϕ,     h,     ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, e_v, e_v_num, dx, dy, set_h_bc, h_bc)
        cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, K1_h, K1_ϕ, 0.5*dt)

        compute_resid(K2_h, K2_ϕ, Tmp_h, Tmp_ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, e_v, e_v_num, dx, dy, set_h_bc, h_bc)
        cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, K2_h, K2_ϕ, 0.5*dt)

        compute_resid(K3_h, K3_ϕ, Tmp_h, Tmp_ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, e_v, e_v_num, dx, dy, set_h_bc, h_bc)
        cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, K3_h, K3_ϕ, dt)

        compute_resid(K4_h, K4_ϕ, Tmp_h, Tmp_ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, e_v, e_v_num, dx, dy, set_h_bc, h_bc)
        update_RK4!(h, ϕ, K1_h, K1_ϕ, K2_h, K2_ϕ, K3_h, K3_ϕ, K4_h, K4_ϕ, dt, set_h_bc, h_bc)

        if (it % nout == 0) && do_monit
            # @show dtp = min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1
            # visu
            p1 = Plt.plot(ϕ[2:end-1,end÷2])
            p2 = Plt.plot(h[2:end-1,end÷2])
            Plt.display(Plt.plot(p1, p2))
            @printf("it %d (dt = %1.3e), max(h) = %1.3f \n", it, dt, maximum(inn(h)))
        end
    end
    return ϕ .* ϕ_, h .* h_, nt, t_sol
end

# ϕ, h, nt, t_sol = simple_sheet(; nx=64, ny=32, itMax=10^5, do_monit=true, set_h_bc=true, e_v_num=0.1, dt=10)
