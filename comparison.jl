using DataFrames, JLD2, Printf, Infiltrator

# helper functions
section(A) = mean(A, dims=2)[2:end-1]
rms(V)     = norm(V) /sqrt(length(V))

# reference fields
include("shmip_results.jl")

nx, ny = 64, 32
xs     = LinRange(0, 100e3, nx-2)
ϕ_ref = get_ϕ_ref(xs)
h_ref = get_h_ref(xs)

test_sets = Dict("diffeq_explicit"  => [(set_h_bc=false, e_v_num=0   ),
                                        #(set_h_bc=true,  e_v_num=0   ),
                                        (set_h_bc=true,  e_v_num=1e-1)],
                 #"diffeq_impl"     =>   [],
                "forward-euler_CN"  => [(set_h_bc=false, e_v_num=0,    CN=0  , dt=1e-4),
                                        (set_h_bc=false, e_v_num=0,    CN=0.5, dt=1e-4),
                                        #(set_h_bc=true,  e_v_num=1e-1, CN=0  , dt=10),
                                        (set_h_bc=true,  e_v_num=1e-1, CN=0.5, dt=10  )],

                "pseudo-transient"  => [(set_h_bc=false, e_v_num=0,    update_h_only=false, γ=0.91, dτ_h_=7.3e-6),
                                        (set_h_bc=false, e_v_num=0,    update_h_only=true,  γ=0.91, dτ_h_=1.9e-5),
                                        (set_h_bc=true,  e_v_num=0,    update_h_only=false, γ=0.91, dτ_h_=7.2e-6),
                                        (set_h_bc=true,  e_v_num=1e-1, update_h_only=false, γ=0.90,  dτ_h_=1.1e-5),
                                        (set_h_bc=true,  e_v_num=1e-1, update_h_only=true,  γ=0.90, dτ_h_=2.5e-5)],

                "RK4"               => [(set_h_bc=false, e_v_num=0., dt=1e-3),
                                        #(set_h_bc=true,  e_v_num=0, dt=1e-3),
                                        (set_h_bc=true,  e_v_num=0.1, dt=10)],

                "RK4_piccard_loop"  => [(set_h_bc=false, e_v_num=0, dt=1e-3)]
)

df    = DataFrame(method=String[], kwargs=NamedTuple[], ϕ_test = Array[], h_test = Array[], rms_ϕ=Array{Float64}[], rms_h=Array{Float64}[], nit=Array{Int64}[], t_run=Array{Float64}[])
tic = Base.time()
for method in keys(test_sets)
    include(method * ".jl")
    i = 0
    for kwargs in test_sets[method]
        i += 1

        ϕ_test = Array[]
        h_test = Array[]
        rms_ϕ  = []
        rms_h  = []
        nit    = []
        t_run  = []

        if startswith(method, "diffeq")
            itMaxs = [1e3, 5e3, 1e4, 2e4]
        else
            itMaxs = [1e3, 5e3, 1e4, 2e4, 5e4, 1e5]
        end

        @printf("Running %s for test_set %d out of %d \n", method, i, length(test_sets[method]))

        for itMax in itMaxs
            ϕ, h, it, t = simple_sheet(; nx, ny, itMax, kwargs...)
            ϕs = section(ϕ)
            hs = section(h)

            push!(ϕ_test, ϕs)
            push!(h_test, hs)
            push!(rms_ϕ, rms(ϕs .- ϕ_ref))
            push!(rms_h, rms(hs .- h_ref))
            push!(nit, it)
            push!(t_run, t)
        end
        push!(df, (;method, kwargs, ϕ_test, h_test, rms_ϕ, rms_h, nit, t_run))
    end
end
toc = Base.time() - tic
@printf("Executed in %f minutes.", toc/60)  # taking 52 minutes at the moment

save("comparison.jld2", "df", df)

# basic plotting with Plots
# all h and ϕ cross sections

#Plt.plot(df.h_test)
#Plt.plot!(h_ref, color="black", linewidth=2)
#Plt.plot(df.ϕ_test)
#Plt.plot!(ϕ_ref, color="black", linewidth=2)

# rms against iterations

#Plt.plot(df.nit, df.rms_h)
#Plt.plot(df.nit, df.rms_ϕ)
