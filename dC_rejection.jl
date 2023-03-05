
 

"""
uses rejection sampling to sample from a `dims` dimensional distribution with density `f`.
Currently an upper bound is specified by a single line at `f_max`.
`x_min` and `x_max` specify the rectangular domain and `n` is the number of tries
"""
function rejection_sampling(dim::Int64, f, f_max::Float64; x_min=0, x_max=10, n=10000)
    xs = x_min .+ (x_max - x_min)*rand(n,dim)
    ps = f_max*rand(n,1)
    return xs[[f(xs[i,:]) .>= ps[i] for i=1:n],:]
end

 

"""
Calculate the expectation of exp(max (u+eps)/lambda) over eps.
Neeeds a file called "gumbel_draws.csv" in the current path.
"""
function ExpH(u::Array{Float64},lambda::Float64,file::String)
    draws = CSV.read(file,DataFrame,header=false);
    draws = Matrix(draws);
    return(mean(mapslices(x -> exp(maximum(u .+ x)/lambda),draws,dims=2)))
end

 
#########################################################################################################
"""
Creates a file "gumbel_draws.csv" containing `n` million draws from `dims` independent
distributions `d`.
"""
function ErrorDraws(d::Distribution,dims::Int64,n::Int64)
    #if file not exists
    if !(isfile(string(typeof(d))*"_draws_scale_" * string(scale(d)) *"_location_"*string(location(d))*".csv"))
        for i=1:n
            eps = Float32.(rand(d,(1_000_000,dims)))
            CSV.write(string(typeof(d))*"_draws_scale_" * string(scale(d)) *"_location_"*string(location(d))*".csv",  Tables.table(eps), writeheader=false,append=true)
        end
    end
end
 
"""
Objective function for optimal Lagrange parameter `lambda` and given `u` and `rho`.
"""
function OptDRO(lambda::Float64,u::Matrix{Float64},rho::Float64,file::String)
    if lambda <= 0.0
        return (1000000)
    end
    return(lambda*rho + lambda * log(ExpH(u,lambda,file)) )
end


function robustSampling(dc_model::dC_model,rho::Float64)
    ErrorDraws(dc_model.d,length(dc_model.u),50)
    file = string(typeof(dc_model.d))*"_draws_scale_" * string(scale(dc_model.d)) *"_location_"*string(location(dc_model.d))*".csv"        #needed for the correct calculation of lambda

    res = optimize(x->OptDRO(first(x),dc_model.u,rho,file), [1.2*scale(dc_model.d)], show_trace=true, g_tol=1e-12)     # file is used as a global variable in res
    dc_model.lambda = first(Optim.minimizer(res))    #get optimal lambda > beta for gumbel
    #@assert lambda > beta

    dc_model.H = ExpH(dc_model.u,dc_model.lambda,file)  # get expectation term for optimal lambda
    dc_model.rho = rho
    return(0)
end

function choiceProbs(dc_model::dC_model,n=100_000_000)
    dims = length(dc_model.u)
    f_DRO(x) = prod(pdf(dc_model.d,y) for y in x) *exp(maximum(dc_model.u .+ x)/dc_model.lambda) / dc_model.H
    result = optimize(x -> (-1)*f_DRO(x), ones(dims) ,show_trace=true, g_tol=1e-12)     # get maximum of f_DRO //make sure the initial value has right dimension
    f_max = 1.2*(-Optim.minimum(result))

    xs = Matrix{Float64}(undef,0,dims); m=0;
    while m < 10_000_000
        xs = Float32.(rejection_sampling(dims,f_DRO, f_max, x_min=dc_model.domain[1],x_max=dc_model.domain[2], n=n))
        folder = string(typeof(dc_model.d))*"/"
        CSV.write(folder*string(dc_model.u)*"_loc"*string(location(dc_model.d))*"_scale"*string(scale(dc_model.d))*"_rho"*string(dc_model.rho)*"_draws.csv",  Tables.table(xs), writeheader=false,append=true)
        m = m + length(xs)
    end

    Auswahl = mapslices(argmax,dc_model.u .+ xs, dims=2)
    tmp = Array{Float64}(undef,dims) .* 0.0
    for i=1:dims
        tmp[i] = sum(Auswahl .==i)/length(Auswahl)
    end
    dc_model.selection = vec(tmp)
    return(tmp')
end

 