require(rjags)
source("./sampler.R")

p1_X_p2_1 <- function(p1, p2_1){
    res = matrix(0,2,2)
    res[1,] = c(p1[1]*p2_1[1,1], p1[1]*p2_1[1,2] )
    res[2,] = c(p1[2]*p2_1[2,1], p1[2]*p2_1[2,2] )
    return(res)
}

N = rep(400,4)


# ########################
# Defining probabilities distributions to create joint distributions

# P(s1)
s1 = c(0.01,0.99)
# P(s2|s1)
s2_1 = matrix(ncol=2,nrow=2)
s2_1[1,] = c(0.999,0.001)
s2_1[2,] = c(0.001,0.999)
# P(s3) = P(s3|s2,s1)
s3 = c(0.001,0.999)
# P(s4) = P(s4|s3,s2,s1)
s4 = c(0.1,0.9) # 0,1
# P(s5|s4) = P(s5|s4,s3,s2,s1)
s5_4 = matrix(ncol=2,nrow=2)
s5_4[1,] = c(0.9,0.1)
s5_4[2,] = c(0.15,0.85)
# P(s6|s4) = P(s6|s5,s4,s3,s2,s1)
s6_4 = matrix(ncol=2,nrow=2)
s6_4[1,] = c(0.85,0.15)
s6_4[2,] = c(0.1,0.9)
# P(s7) = P(s7|s6,s5,s4,s3,s2,s1)
s7 = c(0.2,0.8) # 0,1
# P(s8|s7) = P(s8|s7,s6,s5,s4,s3,s2,s1)
s8_7 = matrix(ncol=2,nrow=2)
s8_7[1,] = c(0.9,0.1)
s8_7[2,] = c(0.15,0.85)
# P(s9|s7) = P(s9|s8,s7,s6,s5,s4,s3,s2,s1)
s9_7 = matrix(ncol=2,nrow=2)
s9_7[1,] = c(0.85,0.15)
s9_7[2,] = c(0.1,0.9)

# ##############################
# The marginal distributions we

# Two dimentional joint distribution
s12 = p1_X_p2_1(s1,s2_1)
s45 = p1_X_p2_1(s4,s5_4)
s46 = p1_X_p2_1(s4,s6_4)
s78 = p1_X_p2_1(s7,s8_7)
s79 = p1_X_p2_1(s7,s9_7)

# Marginals
s2 = c(1-sum(s12[,2]),sum(s12[,2]))
s5 = c(1-sum(s45[,2]),sum(s45[,2]))
s6 = c(1-sum(s46[,2]),sum(s46[,2]))
s8 = c(1-sum(s78[,2]),sum(s78[,2]))
s9 = c(1-sum(s79[,2]),sum(s79[,2]))

# All marginals
S = c(s1[2],s2[2],s3[2],s4[2],s5[2],s6[2],s7[2],s8[2],s9[2])
# 0.8500 0.8225 0.7725 0.9000 0.7750 0.8250 0.8000 0.7000 0.7500
# 0.99000 0.98902 0.99900 0.90000 0.77500 0.82500 0.80000 0.70000 0.75000


# ############################
# Joint distribution
Sjoint = array(NA, rep(2,9))
for (i1 in seq(2)){ for (i2 in seq(2)){ for (i3 in seq(2)){
for (i4 in seq(2)){ for (i5 in seq(2)){ for (i6 in seq(2)){
for (i7 in seq(2)){ for (i8 in seq(2)){ for (i9 in seq(2)){
    Sjoint[i1,i2,i3,i4,i5,i6,i7,i8,i9] = s1[i1]*s2_1[i1,i2]*s3[i3]*s4[i4]*s5_4[i4,i5]*s6_4[i4,i6]*s7[i7]*s8_7[i7,i8]*s9_7[i7,i9]
} } } } } } } } }


x1 = rev(s1)
x2_1 = matrix(rev(s2_1), 2,2)
x3 = rev(s3)
x4 = rev(s4)
x5_4 = matrix(rev(s5_4), 2,2)
x6_4 = matrix(rev(s6_4), 2,2)
x7 = rev(s7)
x8_7 = matrix(rev(s8_7), 2,2)
x9_7 = matrix(rev(s9_7), 2,2)


Xjoint = array(NA, rep(2,9))
for (i1 in seq(2)){ for (i2 in seq(2)){ for (i3 in seq(2)){
for (i4 in seq(2)){ for (i5 in seq(2)){ for (i6 in seq(2)){
for (i7 in seq(2)){ for (i8 in seq(2)){ for (i9 in seq(2)){
    Xjoint[i1,i2,i3,i4,i5,i6,i7,i8,i9] = x1[i1]*x2_1[i1,i2]*x3[i3]*x4[i4]*x5_4[i4,i5]*x6_4[i4,i6]*x7[i7]*x8_7[i7,i8]*x9_7[i7,i9]
} } } } } } } } }



# Verifying the joint distribution
# stopifnot(equal_float( s12[1,1], sum(Sjoint[1,1,,,,,,,]) ))
# stopifnot(equal_float( s12[1,2], sum(Sjoint[1,2,,,,,,,]) ))
# stopifnot(equal_float( s12[2,1], sum(Sjoint[2,1,,,,,,,]) ))
# stopifnot(equal_float( s12[2,2], sum(Sjoint[2,2,,,,,,,]) ))
# stopifnot(equal_float( s13[1,1], sum(Sjoint[1,,1,,,,,,]) ))
# stopifnot(equal_float( s13[1,2], sum(Sjoint[1,,2,,,,,,]) ))
# stopifnot(equal_float( s13[2,1], sum(Sjoint[2,,1,,,,,,]) ))
# stopifnot(equal_float( s13[2,2], sum(Sjoint[2,,2,,,,,,]) ))

# #################################
# Covariance
cov = matrix(0,9,9)

# The covariance between blocks is 0
s14 = matrix(NA,2,2)
s14[1,] = c(sum(Sjoint[1,,,1,,,,,]), sum(Sjoint[1,,,2,,,,,]) )
s14[2,] = c(sum(Sjoint[2,,,1,,,,,]), sum(Sjoint[2,,,2,,,,,]) )
s24 = matrix(NA,2,2)
s24[1,] = c(sum(Sjoint[,1,,1,,,,,]), sum(Sjoint[,1,,2,,,,,]) )
s24[2,] = c(sum(Sjoint[,2,,1,,,,,]), sum(Sjoint[,2,,2,,,,,]) )
s25 = matrix(NA,2,2)
s25[1,] = c(sum(Sjoint[,1,,,1,,,,]), sum(Sjoint[,1,,,2,,,,]) )
s25[2,] = c(sum(Sjoint[,2,,,1,,,,]), sum(Sjoint[,2,,,2,,,,]) )
cov14 = s14[2,2] - s1[2]*s4[2]
cov24 = s24[2,2] - s2[2]*s4[2]
cov25 = s25[2,2] - s2[2]*s5[2]
# stopifnot(equal_float(cov14,0) )
# stopifnot(equal_float(cov24,0) )
# stopifnot(equal_float(cov25,0) )

# So we compute covariance within blocks
cov[1,2] = s12[2,2] - s1[2]*s2[2]
s56 = matrix(NA,2,2)
s56[1,] = c(sum(Sjoint[,,,,1,1,,,]), sum(Sjoint[,,,,1,2,,,]) )
s56[2,] = c(sum(Sjoint[,,,,2,1,,,]), sum(Sjoint[,,,,2,2,,,]) )
cov[4,5] = s45[2,2] - s4[2]*s5[2]
cov[4,6] = s46[2,2] - s4[2]*s6[2]
cov[5,6] = s56[2,2] - s5[2]*s6[2]
s89 = matrix(NA,2,2)
s89[1,] = c(sum(Sjoint[,,,,,,,1,1]), sum(Sjoint[,,,,,,,1,2]) )
s89[2,] = c(sum(Sjoint[,,,,,,,2,1]), sum(Sjoint[,,,,,,,2,2]) )
cov[7,8] = s78[2,2] - s7[2]*s8[2]
cov[7,9] = s79[2,2] - s7[2]*s9[2]
cov[8,9] = s89[2,2] - s8[2]*s9[2]
#       [,1]      [,2] [,3] [,4]   [,5]     [,6] [,7] [,8] [,9]
#  [1,]    0 0.0098802    0    0 0.0000 0.000000    0 0.00 0.00
#  [2,]    0 0.0000000    0    0 0.0000 0.000000    0 0.00 0.00
#  [3,]    0 0.0000000    0    0 0.0000 0.000000    0 0.00 0.00
#  [4,]    0 0.0000000    0    0 0.0675 0.067500    0 0.00 0.00
#  [5,]    0 0.0000000    0    0 0.0000 0.050625    0 0.00 0.00
#  [6,]    0 0.0000000    0    0 0.0000 0.000000    0 0.00 0.00
#  [7,]    0 0.0000000    0    0 0.0000 0.000000    0 0.12 0.12
#  [8,]    0 0.0000000    0    0 0.0000 0.000000    0 0.00 0.09
#  [9,]    0 0.0000000    0    0 0.0000 0.000000    0 0.00 0.00


# ########################
# Data


simlated_data <- sampler_from_joint(Sjoint=Sjoint,
                                      Xjoint=Xjoint,
                                      P=c(0.5,0.5,0.5,0.5),
                                      n_individuals=1000)


# Verificando que la generación de datos sea correcta.
#
# data89 = simlated_data[simlated_data[,"State"]==1, c("Test8","Test9")]
# contingency_table_89 = matrix(0, nrow=2,ncol=2)
# for ( i in seq(dim(data89)[1])){
#     i8 = data89[i,1]
#     i9 = data89[i,2]
#     contingency_table_89[i8+1, i9+1] = contingency_table_89[i8+1, i9+1] + 1
# }
# contingency_table_89 = contingency_table_89/dim(data89)[1]
# sum(abs(s89 - contingency_table_89) < 0.01) == 4


save(S,cov, file="real.RData")

write.csv(simlated_data, "Covariance_Simulated_Data.csv")


