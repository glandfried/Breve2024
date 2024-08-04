library(rjags)

# Verdaderos valores
#
prevalence = c(0, 0.15, 0.3, 0.45, 0.7)
sensitivity = c(0.9, 0.6)
specificity = c(0.95, 0.9)

# Datos simulados con verdaderos valores
#
r <- matrix(nrow=4, ncol=length(prevalence)) # DATA (simulated)
r[,1] <- c(839, 104, 44, 7)
r[,2] <- c(754, 92, 85, 92)
r[,3] <- c(598, 99, 134, 151)
r[,4] <- c(503, 75, 187, 240)
r[,5] <- c(275, 61, 287, 373)
N <- apply(r, MARGIN=2, FUN=sum); Comunidades = 5


# Parámetros de inferencia por simulación
#
n.chains = 4
n.adapt = 5000
n.burn = 5000
n.iter = 20000
thin = 4


observable = list(
    "Comunidades" = Comunidades,
    "r" = r,
    "N" =N
)


inits = list()
for (c in seq(n.chains )){

    inits[[c]] = list(
        "s" = runif(length(sensitivity), 0.5, 1.0),
        "x" = runif(length(specificity), 0.5, 1.0),
        "p" = runif(length(prevalence), 0.0, 1.0)
    )
}


model.especification = 'HuiWalter.rjags'

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
    variable.names = c('s','x','p'),
    n.iter = n.iter,
    thin = thin
)


# Análisis de cadenas
#
estadisticos <- function(chains){
    variables_names = colnames(chains[[1]])

    # Estructura de la salida
    #
    res = matrix(nrow=length(variables_names), ncol=5)
    colnames(res) = c("2.5%", "Median", "97.5%", "psrf.point", "psrf.upper")
    rownames(res) = variables_names

    # Gelman-Rubin diagnostico para revisar si hay convergencia en las cadenas MCMC
    #
    res[,c("psrf.point", "psrf.upper")] = gelman.diag(chains)[[1]]

    # Intervalo de credibilidad
    #
    colapsed = rbind(chains[[1]],chains[[2]])
    for (vn in variables_names ){
        res[vn,c("2.5%", "Median", "97.5%")] = quantile(colapsed[,vn], p=c(0.025, 0.5,0.975))
    }

    return(res)
}


resumen = estadisticos(chains)

plot_estimates <- function(real,resumen,ylim,name){

    xlim = c(0.75, length(real)+0.25)
    plot(0,0,col=rgb(1,1,1,0),  ylim=ylim, xlim=xlim, axes=F, xlab="", ylab="")
    for (is in seq(length(real))){
        points(is, real[is], pch=19)
        nombre = paste0(name,"[",is,"]")
        print(nombre)
        text(is, ylim[2],nombre, cex=1.5, col=rgb(0,0,0,0.6))
        segments(x0=is, x1=is, y0=resumen[nombre,"2.5%"], y1=resumen[nombre,"97.5%"],lwd=1)
        segments(x0=is-0.05, x1=is+0.05, y0=resumen[nombre,"Median"], y1=resumen[nombre,"Median"])
    }

    abline(v=3.5)
    grilla = seq(ylim[1],ylim[2],by=0.05)
    axis(side=2, at= grilla ,labels=NA,cex.axis=0.6,tck=0.015)
    axis(lwd=0,side=2,at= grilla, cex.axis=1.5,line=-0.45)
    abline(h=grilla, col=rgb(0,0,0,0.1))
}


plot_estimates(real=sensitivity,resumen=resumen,ylim=c(0.5,1), name="s")
plot_estimates(real=specificity,resumen=resumen,ylim=c(0.5,1), name="x")

