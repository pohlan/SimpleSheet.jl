using JLD2, DataFrames, PyPlot, LaTeXStrings

# get SimpleSheet model results
if !isfile("comparison.jld2")
    println("The comparison file doesn't exist. Run comparison.jl!")
elseif !@isdefined(df)
    df = load("comparison.jld2", "df")
end

# get reference fields from GlaDS
include("shmip_results.jl")
nx, ny = 64, 32
xs     = LinRange(0, 100e3, nx-2)
ϕ_ref = get_ϕ_ref(xs)
h_ref = get_h_ref(xs)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 18

function get_marker(kwargs::NamedTuple)
    if kwargs == (set_h_bc = false, e_v_num = 0.0, dt = 0.001)
        sym   = "-o"
        label = "standard"
    elseif kwargs == (set_h_bc = true, e_v_num = 0.1, dt = 10)
        sym = "-s"
        label = "h b.c.; e_v_num"
    end
    return sym
end

function get_lab(kwargs::NamedTuple)
    if kwargs.set_h_bc == false && kwargs.e_v_num == 0 && haskey(kwargs, update_h_only) ? kwargs.update_h_only == false : true
        label = "standard"
    else
        label = string(kwargs)
    end
    return label
end

markers = ["-o", "-s", "-+", "-x", "-d"]

L2D = PyPlot.matplotlib.lines.Line2D
methods = [ "RK4",
            "pseudo-transient",
            "diffeq_explicit",
            "forward-euler_CN",
            "RK4_piccard_loop",
            "GlaDS"]

line_styles = Dict( "RK4"               => L2D([0], [0], color="green",   lw=2.2),
                    "pseudo-transient"  => L2D([0], [0], color="orange",  lw=2.2),
                    "diffeq_explicit"   => L2D([0], [0], color="blue",    lw=2.2),
                    "forward-euler_CN"  => L2D([0], [0], color="magenta", lw=2.2),
                    "RK4_piccard_loop"  => L2D([0], [0], color="cyan",    lw=2.2),
                    "GlaDS"             => L2D([0], [0], color="black",   lw=3  ))

fig, axs = subplots(1, 2, figsize=(15,10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for method in methods
    i = findall(df.method .== method)
    col = line_styles[method].get_color()
    lw  = line_styles[method].get_lw()

    # axs[1].plot.(df.nit[i], df.rms_ϕ[i], markers[1], color=col, lw=lw, label=method)
    # axs[1].set_xlabel("# iterations")
    # axs[1].set_ylabel("RMS(ϕ_test - ϕ_ref)")
    # axs[1].set_yscale("log")
    # axs[1].legend(custom_legend, methods)
    # axs[1].set_title(L"ϕ")

    axs[1].plot.(df.t_run[i]./60, df.rms_ϕ[i], markers[1], color=col, lw=lw, label=method)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Wall time (min)")
    axs[1].set_ylabel("RMS(ϕ_test - ϕ_ref)")
    #axs[1].set_title(L"ϕ")

    axs[1].legend([line_styles[m] for m in methods], methods)

    # axs[3].plot.(df.nit[i], df.rms_h[i], markers[1], color=colors[method])
    # axs[3].set_xlabel("# iterations")
    # axs[3].set_ylabel("RMS(h_test - h_ref)")
    # axs[3].set_yscale("log")
    # axs[3].set_title(L"h")


    axs[2].plot.(df.t_run[i]./60, df.rms_h[i], markers[1], color=col)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Wall time (min)")
    axs[2].set_ylabel("RMS(h_test - h_ref)")
    #axs[2].set_title(L"h")
end

fig, axs = subplots(1, 2, figsize=(15,10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for method in ["RK4_piccard_loop"] #methods
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
    #axs[1].set_ylabel("ϕ")
    axs[1].legend([line_styles[m] for m in methods], methods)
    axs[1].set_title("ϕ (Pa)")

    if method == "GlaDS"
        axs[2].plot(xs .* 1e-3, h_ref, color="black", lw=3, label="GlaDS")
    else
        axs[2].plot.(repeat([xs .* 1e-3], length(i)), h_test, color=col, lw=lw, label=method)
    end
    axs[2].set_xlabel("x (km)")
    axs[2].set_title("h (m)")
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
