using ParallelStencil, Printf, LinearAlgebra, Statistics, PyPlot, Infiltrator

const USE_GPU = true
# Initiate ParallelStencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

α = 1.25; β = 1.5
nx = 64; ny = 32
γ_ϕ = 0.8; γ_h = 0.8
dτ_ϕ_=1.0; dτ_h_= 7e-6

plot_output = true; plot_error = true;

small = eps(Float64)

macro N(ix, iy) esc(:(0.91*H[$ix, $iy] - ϕ[$ix, $iy])) end
macro vo(ix, iy) esc(:( h[$ix, $iy] < 1. ? 1 - h[$ix, $iy] : 0.)) end
macro vc(ix, iy) esc(:(h[$ix, $iy] * @N($ix, $iy)^3)) end
macro dϕ_dx(ix, iy) esc(:( ($ix == 1 || $ix == nx-1) ? 0. : (ϕ[$ix+1, $iy] - ϕ[$ix, $iy]) / dx)) end
macro dϕ_dy(ix, iy) esc(:( ($iy == 1 || $iy == ny-1) ? 0. : (ϕ[$ix, $iy+1] - ϕ[$ix, $iy]) / dy )) end
macro gradϕ(ix, iy) esc(:( sqrt(
    (0.5 * (@dϕ_dx($ix, $iy) + @dϕ_dx($ix-1, $iy)))^2
  + (0.5 * (@dϕ_dy($ix, $iy) + @dϕ_dy($ix, $iy-1)))^2
    ))) end
macro d_eff(ix, iy) esc(:( h[$ix, $iy]^α * (@gradϕ($ix, $iy) + small)^(β-2) )) end
macro dτ_ϕ(ix, iy) esc(:( dτ_ϕ_ * min(min(dx, dy)^2 / d_eff[$ix, $iy] / 4.1, dt))) end
macro flux_x(ix, iy) esc(:(
    - d_eff[$ix+1, $iy] * max(@dϕ_dx($ix, $iy), 0) +   # flux in negative x-direction
    - d_eff[$ix,   $iy] * min(@dϕ_dx($ix, $iy), 0.)    # flux in positive x-direction
    )) end
macro flux_y(ix, iy) esc(:(
    - d_eff[$ix, $iy+1] * max(@dϕ_dy($ix, $iy), 0.) +   # flux in negative y-direction
    - d_eff[$ix, $iy  ] * min(@dϕ_dy($ix, $iy), 0.)    # flux in positive y-direction
    )) end
macro Res_ϕ(ix, iy) esc(:(( H[$ix, $iy] > 0.) * (                                                      # only calculate at points with non-zero ice thickness
    - e_v * (ϕ[$ix, $iy] - ϕ_old[$ix, $iy]) / dt                                                            # dhe/dt = ev(ρw*g) * dϕ/dt
    - ( (@flux_x($ix, $iy) - @flux_x($ix-1, $iy)) / dx + (@flux_y($ix, $iy) - @flux_y($ix, $iy-1)) / dy )    # divergence
    - (Σ * @vo($ix, $iy) - Γ * @vc($ix, $iy))                                                                          # dh/dt
    + Λ                                                                                           # source term Λ * m (m=1)
    )
)) end
macro Res_h(ix, iy) esc(:(( H[$ix, $iy] > 0.) * (
    - (h[$ix, $iy] - h_old[$ix, $iy]) / dt
    + (Σ * @vo($ix, $iy) - Γ * @vc($ix, $iy))
   )
)) end

@parallel_indices (ix, iy) function update_deff!(d_eff, ϕ, h, dx, dy, α, β, H, small)
    nx, ny = size(ϕ)
    if (1 < ix < nx && 1 < iy < ny)
        d_eff[ix, iy] = @d_eff(ix, iy)
    end
    return
end
@parallel_indices (ix,iy) function residuals!(ϕ, ϕ_old, h, h_old, Res_ϕ, Res_h, Λ, Σ, Γ, d_eff,
                                              dx, dy, α, β, dt, H, small, e_v)
    nx, ny = size(ϕ)
    if (ix <= nx && iy <= ny)
        # residual of ϕ
        if ix == 2 # ϕ: without boundary points (divergence of q not defined there)
            Res_ϕ[ix, iy] = 0. # position where dirichlet b.c. are imposed
        elseif (1 < ix < nx && 1 < iy < ny)
            Res_ϕ[ix, iy] = @Res_ϕ(ix, iy)
        end

        # residual of h
        Res_h[ix, iy] = @Res_h(ix, iy)
    end
    return
end
@parallel_indices (ix,iy) function update_fields!(ϕ, ϕ2, ϕ_old, h, h2, h_old, Λ, Σ, Γ, d_eff,
                                                  dx, dy, α, β, dt, H, small, e_v,
                                                  dϕ_dτ, dh_dτ, γ_ϕ, γ_h, dτ_h_, dτ_ϕ_)
    nx, ny = size(ϕ)
    if (1 < ix < nx && 1 < iy < ny)
        # update ϕ
        dϕ_dτ[ix, iy] = @Res_ϕ(ix, iy) + γ_ϕ * dϕ_dτ[ix, iy]
        ϕ2[ix, iy] = ϕ[ix, iy] + @dτ_ϕ(ix, iy) * dϕ_dτ[ix, iy]
        # dirichlet boundary conditions to pw = 0
        if ix == 2
            ϕ2[ix, iy] = ϕ[ix, iy]
        end

        # update h
        dh_dτ[ix, iy] = @Res_h(ix, iy) + γ_h * dh_dτ[ix, iy]
        h2[ix, iy] = h[ix, iy] + dτ_h_ * dh_dτ[ix, iy]
    end
    return
end

# input parameters
dt = 1e9                               # physical time step
Lx = 100e3; Ly = 20e3                  # length/width of the domain, starts at (0, 0)
dx = Lx / (nx-3); dy = Ly / (ny-3)     # the outermost points are ghost points
xc = LinRange(-dx, Lx+dx, nx); yc = LinRange(-dy, Ly+dy, ny)
get_H(x, y) = 6 *( sqrt((x)+5e3) - sqrt(5e3) ) + 1
H = [0.0; ones(nx-2); 0.0] * [0.0 ones(ny-2)' 0.0] .* get_H.(xc, yc') # ice thickness, rectangular ice sheet with ghostpoints
m = 7.93e-11                           # source term for SHMIP A1 test case
e_v = 1e-3                             # void ratio for englacial storage

ϕ0 = 100. * @ones(nx, ny); h0 = 0.04 * @ones(nx, ny) # initial fields of ϕ and h

# scaling factors
ϕ_ = 9.81 * 910 * mean(H)
x_ = max(Lx, Ly)
q_ = 0.005 * 0.1^α * (ϕ_ / x_)^(β-1)
t_ = 0.1 * x_ / q_
Σ  = 5e-8 * x_ / q_
Γ  = 3.375e-25 * ϕ_^3 * x_ / q_ * 2/27  # the last bit is 2/n^n from vc
Λ  = m * x_ / q_

# Apply the scaling and convert to correct data type
ϕ0 = Data.Array(ϕ0 ./ ϕ_)
h0 = Data.Array(h0 ./ 0.1)
dx = dx / x_
dy = dy / x_
dt = dt / t_
H  = Data.Array(H ./ mean(H))

function run_the_model(ϕ0, h0, nx, ny, dx, dy, dt, H, Σ, Γ, Λ, small, e_v, γ_ϕ, γ_h, dτ_h_, dτ_ϕ_;
                       plot_output, plot_error)
    # array allocation
    d_eff = @zeros(nx, ny)
    dϕ_dτ = @zeros(nx, ny); dh_dτ = @zeros(nx, ny)
    Res_ϕ = @zeros(nx, ny); Res_h = @zeros(nx, ny)
    iters  = Int64[]; errs_ϕ = Float64[]; errs_h = Float64[]

    # initialise all ϕ and h fields
    ϕ_old = copy(ϕ0); ϕ = copy(ϕ0); ϕ2 = copy(ϕ0)
    h_old = copy(h0); h = copy(h0); h2 = copy(h0)

    # Pseudo-transient iteration
    iter = 0; itMax = 10^6
    tol  = 1e-6
    err_ϕ, err_h = 2*tol, 2*tol
    while !(max(err_ϕ, err_h) < tol) && iter<itMax
        @parallel update_deff!(d_eff, ϕ, h, dx, dy, α, β, H, small)
        @parallel update_fields!(ϕ, ϕ2, ϕ_old, h, h2, h_old, Λ, Σ, Γ, d_eff,
                                 dx, dy, α, β, dt, H, small, e_v,
                                 dϕ_dτ, dh_dτ, γ_ϕ, γ_h, dτ_h_, dτ_ϕ_)
        # pointer swap
        ϕ, ϕ2 = ϕ2, ϕ
        h, h2 = h2, h

        iter += 1

        # check convergence criterion
        if iter % 1000 == 0
            @parallel residuals!(ϕ, ϕ_old, h, h_old, Res_ϕ, Res_h, Λ, Σ, Γ, d_eff,
                                 dx, dy, α, β, dt, H, small, e_v)
            err_ϕ = norm(Res_ϕ) / length(Res_ϕ)
            err_h = norm(Res_h) / length(Res_h)

            # save error evolution in vector
            append!(iters, iter); append!(errs_ϕ, err_ϕ); append!(errs_h, err_h)

            @printf("iterations = %d, error ϕ = %1.2e, error h = %1.2e \n", iter, err_ϕ, err_h)
        end
    end

    pygui(true)
    if plot_output
        x_plt = [xc[1]; xc .+ (xc[2]-xc[1])]
        y_plt = [yc[1]; yc .+ (yc[2]-yc[1])]
        N = 0.91 * H .- ϕ
        N[H .== 0.0] .= NaN
        h[H .== 0.0] .= NaN

        figure()
        subplot(2, 2, 1)
        pcolor(x_plt, y_plt, h')#, edgecolors="black")
        colorbar()
        title("h")
        subplot(2, 2, 2)
        pcolor(x_plt, y_plt, N')#, edgecolors="black")
        colorbar()
        title("N")
        # cross-sections of ϕ and h
        subplot(2, 2, 3)
        ind = size(N,2)÷2
        plot(xc, h[:, ind])
        title(join(["h at y = ", string(round(yc[ind], digits=1))]))
        subplot(2, 2, 4)
        plot(xc, N[:, ind])
        title(join(["N at y = ", string(round(yc[ind], digits=1))]))
    end
    if plot_error
        figure()
        semilogy(iters, errs_ϕ, label="err_ϕ", color="darkorange")
        semilogy(iters, errs_h, label="err_h", color="darkblue")
        xlabel("# iterations")
        ylabel("error")
        legend()
    end

end

run_the_model(ϕ0, h0, nx, ny, dx, dy, dt, H, Σ, Γ, Λ, small, e_v, γ_ϕ, γ_h, dτ_h_, dτ_ϕ_,
              plot_output=true, plot_error=true);
