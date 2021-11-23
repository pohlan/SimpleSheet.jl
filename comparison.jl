using DataFrames, Infiltrator

# helper functions
section(A) = mean(A, dims=2)[2:end-1]
rms(V)     = norm(V) /sqrt(length(V))

# reference fields
include("shmip_results.jl")

nx, ny = 64, 32
xs     = LinRange(0, 100e3, nx-2)
ϕ_ref = get_ϕ_ref(xs)
h_ref = get_h_ref(xs)

test_sets = Dict("diffeq_explicit" =>   [(set_h_bc=false, e_v_num=0   ),
                                         (set_h_bc=true,  e_v_num=0   ),
                                         (set_h_bc=true,  e_v_num=1e-1)],
                 #"diffeq_impl"     =>   [],
                "forward-euler_CN" =>   [(set_h_bc=false, e_v_num=0,    use_CFL=false, CN=0  ),
                                         (set_h_bc=false, e_v_num=0,    use_CFL=true,  CN=0  ),
                                         (set_h_bc=false, e_v_num=0,    use_CFL=false, CN=0.5),
                                         (set_h_bc=true,  e_v_num=0,    use_CFL=false, CN=0  ),
                                         (set_h_bc=true,  e_v_num=1e-1, use_CFL=false, CN=0  ),
                                         (set_h_bc=true,  e_v_num=1e-1, use_CFL=false, CN=0.5)],
                #"PT"          =>   [],
                #"RK4"         =>   [],
                #"RK4_loop"    =>   []
)

ttots = [5e3, 1e4]
df    = DataFrame(method=String[], kwargs=NamedTuple[], ϕ_test = Array[], h_test = Array[], rms_ϕ=Array{Float64}[], rms_h=Array{Float64}[], nit=Array{Int64}[], t_run=Array{Float64}[])
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

        @printf("Running %s for test_set %d out of %d \n", method, i, length(test_sets[method]))

        for ttot in ttots
            ϕ, h, it, t = simple_sheet(; nx, ny, ttot, kwargs...)
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
