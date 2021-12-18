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

test_sets = Dict(#"diffeq_explicit"  => (kwargs = ((tol=1e-8,),),
                 #                       work   = [1, 1e-1]), #, 1e-2, 5e-3, 1e-3]),                   # e_v_num

                "forward-euler_CN"  => (kwargs = ((use_CFL=true,),),
                                        work   = [100, 50]), #, 20, 10]),                           # e_v_num

                "pseudo-transient"  => (kwargs = ((e_v_num=0,    update_h_only=false),
                                                  (e_v_num=0,    update_h_only=true),
                                                  #(e_v_num=0.2,  update_h_only=true),
                                                  (e_v_num=0.2,  update_h_only=false,)),
                                        work   = [1e-3]), #, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),                         # tolerance

                "RK4"               => (kwargs = ((use_CFL=true,),),
                                        work   = [60, 50]), #, 40, 15]),                               # e_v_num

                "RK4_piccard_loop"  => (kwargs = ((use_CFL=true,),),
                                        work   = [300, 200, 150, 100, 80])                                                  # e_v_num
)

df    = DataFrame(method=String[], kwargs=NamedTuple[], ϕ_test = Array[], h_test = Array[], rms_ϕ=Array{Float64}[], rms_h=Array{Float64}[], work=Array{Float64}[], t_run=Array{Float64}[])
tic = Base.time()
for method in keys(test_sets)
    include(method * ".jl")
    i = 0
    for kwargs in test_sets[method].kwargs
        i += 1

        ϕ_test = Array[]
        h_test = Array[]
        rms_ϕ  = []
        rms_h  = []
        work   = []
        t_run  = []

        @printf("Running %s for test_set %d out of %d \n", method, i, length(test_sets[method].kwargs))

        for w in test_sets[method].work
            if method == "pseudo-transient"
                work_kw = (;tol = w)
            else
                work_kw = (;e_v_num = w)
            end

            ϕ, h, t = simple_sheet(; nx, ny, work_kw..., kwargs...)

            ϕs = section(ϕ)
            hs = section(h)

            push!(ϕ_test, ϕs)
            push!(h_test, hs)
            push!(rms_ϕ, rms(ϕs .- ϕ_ref))
            push!(rms_h, rms(hs .- h_ref))
            push!(work, w)
            push!(t_run, t)
        end
        push!(df, (;method, kwargs, ϕ_test, h_test, rms_ϕ, rms_h, work, t_run))
    end
end
toc = Base.time() - tic
@printf("Executed in %f minutes.", toc/60)  # taking 52 minutes at the moment

save("comparison_test.jld2", "df", df)

# basic plotting with Plots
# all h and ϕ cross sections

#Plt.plot(df.h_test)
#Plt.plot!(h_ref, color="black", linewidth=2)
#Plt.plot(df.ϕ_test)
#Plt.plot!(ϕ_ref, color="black", linewidth=2)

# rms against iterations

#Plt.plot(df.nit, df.rms_h)
#Plt.plot(df.nit, df.rms_ϕ)
