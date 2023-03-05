include("dC_model.jl")


u = [0.0 1.0 2.0 2.1]
rho_vector = [0.1, 0.7, 1.3,2.2,4.3]

for rho in rho_vector
    mu = 0.0; beta = 1.0;
    dc_model = dC_model((-1,25),Gumbel(mu,beta),u)
    dc_model.rho = rho
        
    robustSampling(dc_model,rho)
    choiceProbs(dc_model,10_000_000)
    save_print(dc_model)
end
