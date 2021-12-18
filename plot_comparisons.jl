using JLD2, DataFrames, PyPlot, LaTeXStrings, LinearAlgebra

# get SimpleSheet model results
if !isfile("comparison.jld2")
    println("The comparison file doesn't exist. Run comparison.jl!")
else
    df = load("comparison.jld2", "df")
end

# get reference fields from GlaDS
include("shmip_results.jl")
nx, ny = 64, 32
xs     = LinRange(0, 100e3, nx-2)
ϕ_ref = get_ϕ_ref(xs)
h_ref = get_h_ref(xs)
rms_ϕ_ref = norm(ϕ_ref) / sqrt(length(ϕ_ref))
rms_h_ref = norm(h_ref) / sqrt(length(h_ref))

rm_points(A, n) = A[1+n:end]

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 27

L2D = PyPlot.matplotlib.lines.Line2D
methods = [ "RK4",
            "pseudo-transient",
            "diffeq_explicit",
            "forward-euler_CN",
            "RK4_piccard_loop",
            "GlaDS"]

line_styles = Dict( "RK4"               => L2D([0], [0], color="cadetblue",      lw=3),
                    "pseudo-transient"  => L2D([0], [0], color="orange",         lw=3),
                    "diffeq_explicit"   => L2D([0], [0], color="indianred",      lw=3),
                    "forward-euler_CN"  => L2D([0], [0], color="hotpink",        lw=3),
                    "RK4_piccard_loop"  => L2D([0], [0], color="deepskyblue", lw=3),
                    "GlaDS"             => L2D([0], [0], color="black",   lw=3  ))

kw_lines    = Dict( "standard"          => L2D([0], [0], color="orange", lw=2.2, ls="-",  marker="o"),
                    "e_v_num"           => L2D([0], [0], color="orange", lw=2.2, ls=":",  marker="o"),
                    "split-step"        => L2D([0], [0], color="orange", lw=2.2, ls="--", marker="o"))

function get_style(method, kws)
    if method == "pseudo-transient"
        if kws.e_v_num > 0
            style = join([kw_lines["e_v_num"].get_ls(), kw_lines["e_v_num"].get_marker()])
        elseif kws.update_h_only == true
            style = join([kw_lines["split-step"].get_ls(), kw_lines["e_v_num"].get_marker()])
        else
            style = join([kw_lines["standard"].get_ls(), kw_lines["e_v_num"].get_marker()])
        end
    else
        style = "-o"
    end
    return style
end

fig, axs = subplots(1, 2, figsize=(18,10))
fig.subplots_adjust(hspace=0.3, wspace=0.5)
for method in methods
    i = findall(df.method .== method)
    col = line_styles[method].get_color()
    lw  = line_styles[method].get_lw()

    if method == "forward-euler_CN"
        axs[1].plot.(rm_points.(df.t_run[i], 3), rm_points.(df.rms_ϕ[i], 3) ./ rms_ϕ_ref, get_style(method, df.kwargs), color=col, lw=lw, label=method)
    else
       axs[1].plot.(df.t_run[i], df.rms_ϕ[i] ./ rms_ϕ_ref, get_style.(method, df.kwargs[i]), color=col, lw=lw, label=method)
    end

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Wall time (s)")
    axs[1].set_ylabel(L"$\mathrm{RMS}(\delta ϕ)$ / $\mathrm{RMS}(ϕ_\mathrm{ref})$") # RMS($ϕ_\text{test}$ - $ϕ_\text{ref}$)
    axs[1].set_title(L"ϕ")

    axs[1].legend([kw_lines[m] for m in keys(kw_lines)], collect(keys(kw_lines)),
                  title="Pseudo-transient",
                  title_fontsize="small",
                  fontsize="small",
                  loc=(0.45, 0.3))

    if method == "forward-euler_CN"
        axs[2].plot.(rm_points.(df.t_run[i], 3), rm_points.(df.rms_h[i], 3) ./ rms_h_ref, get_style(method, df.kwargs), color=col, lw=lw, label=method)
    else
        axs[2].plot.(df.t_run[i], df.rms_h[i] ./ rms_h_ref, get_style.(method, df.kwargs[i]), color=col, lw=lw)
    end
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Wall time (s)")
    axs[2].set_ylabel(L"$\mathrm{RMS}(\delta h)$ / $\mathrm{RMS}(h_\mathrm{ref})$")
    axs[2].set_title(L"h")

    axs[2].legend([line_styles[m] for m in methods], methods[1:end-1],
                   fontsize="small")
end

fig, axs = subplots(1, 2, figsize=(18,10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for method in methods
    i = findall(df.method .== method)
    ϕ_test = [df.ϕ_test[n][end] for n in i]
    h_test = [df.h_test[n][end] for n in i]
    col = line_styles[method].get_color()
    lw  = line_styles[method].get_lw()

    if method == "GlaDS"
        axs[1].plot(xs .* 1e-3, ϕ_ref, color="black", lw=3, label="GlaDS")
    else
        axs[1].plot.(repeat([xs .* 1e-3], length(i)), ϕ_test, color=col, lw=lw, label=method)
    end
    axs[1].set_xlabel("x (km)")
    axs[1].set_ylabel("ϕ (Pa)")
    #axs[1].set_title("ϕ (Pa)")

    if method == "GlaDS"
        axs[2].plot(xs .* 1e-3, h_ref, color="black", lw=3, label="GlaDS")
    else
        axs[2].plot.(repeat([xs .* 1e-3], length(i)), h_test, color=col, lw=lw, label=method)
    end
    axs[2].set_xlabel("x (km)")
    axs[2].set_ylabel("h (m)")
    #axs[2].set_title("h (m)")

    axs[2].legend([line_styles[m] for m in methods], methods, fontsize="small")

    fig.suptitle("Fields after maximum wall time")
end

# all variants of one method

# method = "RK4_piccard_loop"
# i = df.method .== method
# figure(figsize=(12, 12))
# subplot(1,2,1)
# plot.(df.nit[i], df.rms_ϕ[i], markers[1:sum(i)], color=colors[method])   # label is not broadcastable
# title(L"ϕ")

# subplot(1,2,2)
# plot.(df.nit[i], df.rms_h[i], markers[1:sum(i)], color=colors[method])
# title(L"h")
