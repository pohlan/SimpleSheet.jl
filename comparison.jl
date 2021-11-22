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

include("diffeq_explicit.jl")
keywords = [(set_h_bc=false, e_v_num=0   ),
            #(set_h_bc=true,  e_v_num=0   ),
            (set_h_bc=true,  e_v_num=1e-1)]

test_output = Dict("diffeq_expl" => Dict(),
                   "diffeq_impl" => Dict(),
                   "forward"     => Dict(),
                   "PT"          => Dict(),
                   "RK4"         => Dict(),
                   "RK4_loop"    => Dict(),
                   )

ttots = [5e3, 1e4]
df    = DataFrame(kwargs=NamedTuple[], ϕ_test = Array[], h_test = Array[], rms_ϕ=Array{Float64}[], rms_h=Array{Float64}[], nit=Array{Int64}[], t_run=Array{Float64}[])
for kwargs in keywords
    ϕ_test = Array[]
    h_test = Array[]
    rms_ϕ  = []
    rms_h  = []
    nit    = []
    t_run  = []

    for ttot in ttots
        ϕ, h, it, t = simple_sheet(;ttot, nx, ny, kwargs...)
        ϕs = section(ϕ)
        hs = section(h)

        push!(ϕ_test, ϕs)
        push!(h_test, hs)
        push!(rms_ϕ, rms(ϕs .- ϕ_ref))
        push!(rms_h, rms(hs .- h_ref))
        push!(nit, it)
        push!(t_run, t)
    end
    push!(df, (;kwargs, ϕ_test, h_test, rms_ϕ, rms_h, nit, t_run))
end
