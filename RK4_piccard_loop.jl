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

@views function compute_resid(Ki_h, Ki_ϕ, h, ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, Ψ, e_v_num, dx, dy)

    # BC
    h .= max.(h, 0.0)
    # dirichlet boundary conditions to pw = 0
    ϕ[2,:] .= 0.0

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
    vc     .=  h .* (H .- ϕ).^3

    Ki_h       .= Σ .* inn(vo) .- Γ .* inn(vc)
    Ki_ϕ       .= (- div_q .- Ki_h .+ Λ) ./ Ψ
    Ki_ϕ[1, :] .= 0.                         # Dirichlet B.C. points
    Ki_h      .+= e_v_num .* Ki_ϕ
    return
end

@views function cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, Ki_h, Ki_ϕ, fac)
    inn(Tmp_h) .= inn(h) .+ fac .* Ki_h
    inn(Tmp_ϕ) .= inn(ϕ) .+ fac .* Ki_ϕ
    return
end

@views function update_RK4!(h, ϕ, h_o, ϕ_o, K1_h, K1_ϕ, K2_h, K2_ϕ, K3_h, K3_ϕ, K4_h, K4_ϕ, dt)
    # inn(h) .= inn(h) .+ dt .* (K1_h .+ 2.0 .* K2_h .+ 2.0 .* K3_h .+ K4_h) ./ 6.0
    # inn(ϕ) .= inn(ϕ) .+ dt .* (K1_ϕ .+ 2.0 .* K2_ϕ .+ 2.0 .* K3_ϕ .+ K4_ϕ) ./ 6.0
    inn(h) .= inn(h_o) .+ dt .* (K1_h .+ 2.0 .* K2_h .+ 2.0 .* K3_h .+ K4_h) ./ 6.0
    inn(ϕ) .= inn(ϕ_o) .+ dt .* (K1_ϕ .+ 2.0 .* K2_ϕ .+ 2.0 .* K3_ϕ .+ K4_ϕ) ./ 6.0

    h .= max.(h, 0.0)
    # dirichlet boundary conditions to pw = 0
    ϕ[1:2,:] .= 0.0
    return
end

@views function simple_sheet(;  nx, ny,          # grid size
                                itMax=10^6,      # maximal number of iterations
                                dt=1e-3,         # physical time step, fixed
                                do_monit=false,  # enable/disable plotting of intermediate results
                                use_CFL=true,    # whether dt should be adapted according to CFL condition
                                e_v_num=0)       # regularisation void ratio
    # physics
    Lx, Ly = 100e3, 20e3                        # length/width of the domain, starts at (0, 0)
    ttot   = 8000day
    α      = 1.25
    β      = 1.5
    m      = 7.93e-11                           # source term for SHMIP A1 test case
    e_v    = 1e-6                               # void ratio for englacial storage

    # numerics
    nout   = 1000
    # derived
    nt     = min(Int(ttot ÷ dt), itMax)
    dx, dy = Lx / (nx-3), Ly / (ny-3)           # the outermost points are ghost points
    xc, yc = LinRange(-dx, Lx+dx, nx), LinRange(-dy, Ly+dy, ny)

    # ice thickness
    H           = zeros(nx, ny)
    get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1
    inn(H)     .= inn(get_H.(xc, yc'))

    # initial conditions
    ϕ0       = zeros(nx, ny)
    inn(ϕ0) .= 100.
    #ϕ0     = repeat(LinRange(10, 3.6e6, nx),inner=(1, ny))
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
    ϕ = copy(ϕ0); ϕ_o = copy(ϕ0)
    h = copy(h0); h_o = copy(h0)

    Err1 = zeros(nx  ,ny  )
    Err2 = zeros(nx  ,ny  )
    ittot    = 0
    it_outer = 0
    t        = 0.
    # Time loop
    @printf("Running for e_v_num = %1.e \n", e_v_num)
    t_sol=@elapsed while t < ttot

        # timestep
        if use_CFL && it_outer > 0
            dt = Ψ .* min(dx,dy)^2 ./ max(maximum(d_eff), small) ./ 4.1
        end

        h_o .= h
        ϕ_o .= ϕ

        err = 1.0; epsi = 1e-5; iter = 0
        while err > epsi && iter < 1e1
            Err1 .= ϕ
            Err2 .= h
            compute_resid(K1_h, K1_ϕ,     h,     ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, Ψ, e_v_num, dx, dy)
            cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, K1_h, K1_ϕ, 0.5*dt)

            compute_resid(K2_h, K2_ϕ, Tmp_h, Tmp_ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, Ψ, e_v_num, dx, dy)
            cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, K2_h, K2_ϕ, 0.5*dt)

            compute_resid(K3_h, K3_ϕ, Tmp_h, Tmp_ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, Ψ, e_v_num, dx, dy)
            cumul_RK4!(Tmp_h, Tmp_ϕ, h, ϕ, K3_h, K3_ϕ, dt)

            compute_resid(K4_h, K4_ϕ, Tmp_h, Tmp_ϕ, dϕ_dx, dϕ_dy, gradϕ, d_eff, flux_x, flux_y, div_q, vo, vc, α, small, β, H, Σ, Γ, Λ, Ψ, e_v_num, dx, dy)
            update_RK4!(h, ϕ, h_o, ϕ_o, K1_h, K1_ϕ, K2_h, K2_ϕ, K3_h, K3_ϕ, K4_h, K4_ϕ, dt)

            Err1 .= abs.(Err1 .- ϕ)
            Err2 .= abs.(Err2 .- h)
            err   = max(maximum(Err1), maximum(Err2))
            # if (iter % 100 == 0) @printf("iter = %d, err = %1.3e \n", iter, err) end
            iter  += 1
            ittot += 1
        end

        it_outer += 1
        t        += dt

        # check convergence criterion
        if (it_outer % nout == 0) && do_monit
            # @show dtp = min(dx,dy)^2 ./ maximum(d_eff) ./ 4.1
            # visu
            p1 = Plt.plot(ϕ[2:end-1,end÷2] .* ϕ_)
            p2 = Plt.plot(h[2:end-1,end÷2] .* h_)
            Plt.display(Plt.plot(p1, p2))
            @printf("it %d (piccard iter = %d), dt = %1.2e, t = %1.f days \n", ittot, iter, dt * t_, t/day * t_)
        end
    end
    return ϕ .* ϕ_, h .* h_, t_sol
end

 # ϕ, h, t_sol = simple_sheet(; nx=64, ny=32, use_CFL=true, do_monit=true, e_v_num=100)
 # e_v_num is not very helpful here
