par(mar=c(3.75,4.25,0.5,0.5))
set.seed(999)

Q_c = 3    # Pagos por cara
Q_s = 1.2  # Pagos por sello
p = 0.5    # Probabilidad de la moneda
b = 0.5    # La apuesta

esperanza = p*b*Q_c + (1-p)*(1-b)*Q_s
print(paste("Esperanza", esperanza))

# ################################
# Riqueza promedio de una población

# Pasos temporales
#
t = seq(0,10)

# Esperanza en el tiempo
#
r = esperanza^t

# Plot
#
plot(t,log(r),lwd=2,axes = F,ann = F, col=rgb(0.8,0.2,0.2,1),pch=19, cex=1.5, ylim=c(-max(log(r)),max(log(r))))

# Itera por tamaño de población
#
for (n in c(1,10,100,1000,10000)){

    # Inicializa en 0 riqueza total de la población
    #
    total = numeric(11)

    # Itera por persona al interior de la población
    #
    for (j in seq(n)){

        # Tira las 10 monedas (una por paso temporal)
        #
        m = rbinom(0.5,n=10,size=1)

        # Suma la riqueza obtenida por el individuo a la riqueza total
        #
        total = total + c(1,cumprod(1.5*m + 0.6*(1-m)))
    }

    # Grafica promedio de riqueza de la población
    #
    lines(t,log(total/n) ,lwd=3, col=rgb(0,0,0,(log(n,10)+1)/6 ) )
}
# Detalles
axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
axis(lwd=0,side=1,  cex.axis=1.5,line=-0.3)
axis(lwd=0,side=2,cex.axis=1.5,line=-0.3)
mtext("Tiempo", side=1, line=2.5,cex = 2)
mtext("log Riqueza", side=2, line=2.5,cex = 2)
legend(0,0.5,pch=19,legend = c(expression(10^0), expression(10^1), expression(10^2), expression(10^3), expression(10^4)), col=c(rgb(0,0,0,1/6),rgb(0,0,0,2/6),rgb(0,0,0,3/6),rgb(0,0,0,4/6),rgb(0,0,0,5/6)),bty = "n",cex = 1.5)


# #######################
# Riqueza de los individuos en el tiempo

set.seed(999)

# Pasos temporales
#
t = seq(0,1000)

# Esperanza en el tiempo
#
r = 1.05^t

# Grafica la esperanza de riqueza en el tiempo
#
plot(t,log(r),lwd=2,axes = F,ann = F, col=rgb(0.8,0.2,0.2,1),pch=19, cex=1.5, ylim=c(-max(log(r))-0.5,max(log(r))))

# Itera por individuo
#
for (i in seq(33)){

    # Tira todas las monedas que obtiene un individuo en el tiempo
    #
    m = c(1,rbinom(0.5,n=1000,size=1))

    # Grafica la riqueza del individuo en el tiempo }
    #
    lines(t,log(cumprod(1.5*m + 0.6*(1-m))) ,lwd=2, col=rgb(runif(1),runif(1),runif(1)) )
}
# Detalles
axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
axis(lwd=0,side=1,  cex.axis=1.5,line=-0.3)
axis(lwd=0,side=2,cex.axis=1.5,line=-0.3)
mtext("Tiempo", side=1, line=2.5,cex = 2)
mtext("log Riqueza", side=2, line=2.5,cex = 2)
text(900,-16,expression("?"),srt=0, cex=5)


# Vuelvo a graficar los pero a más largo plazo
#
set.seed(9)
t = seq(0,10000)
r = 1.05^t
plot(t,log(r),lwd=2,axes = F,ann = F, col=rgb(0.8,0.2,0.2,1),pch=19, cex=1.5, ylim=c(-max(log(r))-0.5,max(log(r))))
for (i in seq(33)){
    m = c(1,rbinom(0.5,n=10000,size=1))
    lines(t,log(cumprod(1.5*m + 0.6*(1-m))) ,lwd=2, col=rgb(runif(1),runif(1),runif(1)) )
}

axis(side=2, labels=NA,cex.axis=0.6,tck=0.015)
axis(side=1, labels=NA,cex.axis=0.6,tck=0.015)
axis(lwd=0,side=1,  cex.axis=1.5,line=-0.3)
axis(lwd=0,side=2,cex.axis=1.5,line=-0.3)
mtext("Tiempo", side=1, line=2.5,cex = 2)
mtext("log Riqueza", side=2, line=2.5,cex = 2)
#segments(x0=0,x1=10000, y0=0, y1=log(0.6^(1/2)*1.5^(1/2))*10000,lwd=3)
#text(900,16,expression("5%"),srt=0, cex=2)
#text(900,-16,expression("-5%"),srt=0, cex=2)
abline(h=0,lty=3)

