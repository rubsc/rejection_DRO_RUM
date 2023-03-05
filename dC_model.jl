using Distributions
using Random
using LinearAlgebra
using DataFrames
using Tables
using CSV
using Optim

#########################################################################################################
mutable struct dC_model{T<: UnivariateDistribution}
    domain::Tuple{Int64,Int64}
    d::T
    u::Matrix{Float64}
    lambda::Float64
    H::Float64
    rho::Float64
    selection::Vector{Float64}
    function dC_model(d::T,u::Matrix{Float64}) where {T<:UnivariateDistribution}
        new{T}((-10,10),d::T,u::Matrix{Float64},0.0,0.0,0.0,zeros(length(u)))
    end
    function dC_model(domain::Tuple{Int64,Int64},d::T,u::Matrix{Float64}) where {T<:UnivariateDistribution}
        new{T}(domain::Tuple{Int64,Int64},d::T,u::Matrix{Float64},0.0,0.0,0.0,zeros(length(u)))
    end
    function dC_model(domain::Tuple{Int64,Int64},d::T,u::Matrix{Float64},lambda::Float64,H::Float64,rho::Float64) where {T<:UnivariateDistribution}
        new{T}(domain::Tuple{Int64,Int64},d::T,u::Matrix{Float64},lambda,H,rho,zeros(length(u)))
    end
end
 



function save_print(dc_model::dC_model)
    ms2 = DataFrame()    
    insertcols!(ms2, 1, :u => string(dc_model.u))
    insertcols!(ms2, 2, :dim => length(dc_model.u))
    insertcols!(ms2, 3, :domain=> string(dc_model.domain))
    insertcols!(ms2, 4, :loc => location(dc_model.d))
    insertcols!(ms2, 5, :scale => scale(dc_model.d))
    insertcols!(ms2, 6, :rho => dc_model.rho)
    insertcols!(ms2, 7, :lambda => dc_model.lambda)
    insertcols!(ms2, 8, :Expectation => dc_model.H)

    ms2=hcat(ms2,DataFrame(Matrix(dc_model.selection'),:auto)) 
    CSV.write("result.csv",  ms2, writeheader=true,append=true)    
end


include("dC_rejection.jl")