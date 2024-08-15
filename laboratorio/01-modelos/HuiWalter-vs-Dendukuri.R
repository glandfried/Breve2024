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
    "D" = D,
    "m" = 0
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



plot_estimates <- function(real,estimates,ylim,xlim,name, max=NA, min=NA){

    grilla = seq(ylim[1],ylim[2],by=0.05)

    if (is.na(max[1])){
        max = rep(-10, length(real))
        min = rep(-10, length(real))
    }

    plot(0,0,col=rgb(1,1,1,0),  ylim=ylim, xlim=xlim, axes=F, xlab="", ylab="")
    for (is in seq(length(real))){#is=1
        points(is, real[is], pch=19)
        text(is, ylim[2],paste0(name,toString(is)), cex=1.5, col=rgb(0,0,0,0.6))
        if (length(real)>1){
            segments(x0=is, x1=is, y0=estimates[is,1], y1=estimates[is,3],lwd=1)
            segments(x0=is-0.05, x1=is+0.05, y0=estimates[is,2], y1=estimates[is,2])
        }
        if (length(real)==1){
            segments(x0=is, x1=is, y0=estimates[1], y1=estimates[3],lwd=1)
            segments(x0=is-0.05, x1=is+0.05, y0=estimates[2], y1=estimates[2])
        }
        segments(x0=is-0.1, x1=is+0.1, y0=max[is], y1=max[is], lty=2)
        segments(x0=is-0.1, x1=is+0.1, y0=min[is], y1=min[is], lty=2)
    }

    abline(v=3.5)
    axis(side=2, at= grilla ,labels=NA,cex.axis=0.6,tck=0.015)
    #axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
    #axis(lwd=0,side=1, cex.axis=1.5,line=-0.45)
    axis(lwd=0,side=2,at= grilla, cex.axis=1.5,line=-0.45)
    abline(h=grilla, col=rgb(0,0,0,0.1))
}

load("datos/real.RData")
cov_real = numeric(36)
c = 1
for (i in seq(8)){
    for (j in seq(i+1,9)){
    cov_real[c] = cov[i,j]
    c = c + 1
    }
}



s = brief[seq(36*2+4, 36*2+4+8),seq(3)]
covs = brief[seq(36),seq(3)]


plot_estimates(real=S,estimates=s,ylim=c(0.65,1), xlim=c(0.75, length(S)+0.25), name="s")


observable_HuiWalter = list(
    "Tests" = N_tests,
    "Combinations" = length(N),
    "t1"=T1,
    "t2"=T2,
    "N" = N,
    "D" = D,
    "m" = 0
)
model.engine.HW <- jags.model(
    file = model.especification,
    data = observable,
    inits = inits,
    n.chains = n.chains,
    n.adapt = n.adapt
)

# Burn
update(model.engine.HW , n.burn = 5000)

# Sample
chains_HW <- coda.samples(
    model = model.engine ,
    variable.names = c('s','x','p', 'covs', 'covx', 'm', 'pm'),
    n.iter = n.iter,
    thin = thin
)


variables_names = colnames(chains_HW[[1]])
brief_HW = matrix(nrow=length(variables_names), ncol=5)
colnames(brief_HW) = c("2.5%", "Median", "97.5%", "psrf.point", "psrf.upper")
rownames(brief_HW) = variables_names
colapsed_HW = rbind(chains_HW[[1]],chains_HW[[2]])
for (vn in variables_names ){
    brief_HW[vn,c("2.5%", "Median", "97.5%")] = quantile(colapsed_HW[,vn], p=c(0.025, 0.5,0.975))
}

s_HW = brief_HW[seq(36*2+4, 36*2+4+8),seq(3)]

plot_estimates(real=S,estimates=s_HW,ylim=c(0.65,1), xlim=c(0.75, length(S)+0.25), name="s")


