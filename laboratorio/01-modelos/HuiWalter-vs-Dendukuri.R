library(rjags)



data = read.csv("datos/Covariance_Simulated_Data.csv")
data = data[-c(1,12)]
N_tests = 9


combinatorial_maping <- function(t1,t2,N){
    # Maps the combination of tests to a number.
    # With N = 4:
    #     1 2 -> 1
    #     1 3 -> 2
    #     1 4 -> 3
    #     2 3 -> 4
    #     2 4 -> 5
    #     3 4 -> 6
    return( (N-1)*N/2 - (N-t1)*(N-t1+1)/2 + t2 - t1 )
}


joint_diagnosis_from_individual_data <- function(individual_data){
    # individual_data = simlated_data_1 # from: sampler_from_joint()
    # columns(individual_data) == c("Population", "Test1", "Test2", "State")
    res = c(
        sum((1-individual_data[,2]) & (1-individual_data[,3])),# 00
        sum((  individual_data[,2]) & (1-individual_data[,3])),# 10
        sum((1-individual_data[,2]) & (  individual_data[,3])),# 01
        sum((  individual_data[,2]) & (  individual_data[,3])))# 11
    return(res)
}


T1=numeric(0)
T2=numeric(0)
N = numeric(0)
D = matrix(0, nrow=4, ncol=((N_tests)*(N_tests)-N_tests)/2 )
for (t1 in seq(1,N_tests-1)){
    for (t2 in seq(t1+1,N_tests)){
        test1 = paste0("Test",t1)
        test2 = paste0("Test",t2)
        i = combinatorial_maping(t1=t1,t2=t2,N=N_tests)
        T1[i] = t1; T2[i] = t2
        N[i] = dim(data[, c("Population",test1, test2)])[1]
        D[,i] = joint_diagnosis_from_individual_data(data[, c("Population",test1, test2)])
    }
}

cantidad_de_marginales = 36
dim(D) == c(4, cantidad_de_marginales)

# ##################
# Inference

n.chains = 2
n.adapt = 5000
n.burn = 5000
n.iter = 20000
thin = 4

observable = list(
    "Tests" = N_tests,
    "Combinations" = length(N),
    "t1"=T1,
    "t2"=T2,
    "N" = N,
    "D" = D
)

inits = list()
for (c in seq(n.chains )){

    inits[[c]] = list(
        "s" = runif(N_tests, 0.5, 1.0),
        "x" = runif(N_tests, 0.5, 1.0),
        "p" = 0.5,
        "pm" = 0.5,
        "covs" = numeric(length(N)),
        "covx" = numeric(length(N))
    )
}


model.especification = 'HuiWalter-vs-Dendukuri.rjags'

model.engine <- jags.model(
    file = model.especification,
    data = observable,
    inits = inits,
    n.chains = n.chains,
    n.adapt = n.adapt
)

# Burn
update(model.engine , n.burn = 5000)

# Sample
chains <- coda.samples(
    model = model.engine ,
    variable.names = c('s','x','p', 'covs', 'covx', 'm', 'pm'),
    n.iter = n.iter,
    thin = thin
)


variables_names = colnames(chains[[1]])
brief = matrix(nrow=length(variables_names), ncol=5)
colnames(brief) = c("2.5%", "Median", "97.5%", "psrf.point", "psrf.upper")
rownames(brief) = variables_names
colapsed = rbind(chains[[1]],chains[[2]])
for (vn in variables_names ){
    brief[vn,c("2.5%", "Median", "97.5%")] = quantile(colapsed[,vn], p=c(0.025, 0.5,0.975))
}

brief


