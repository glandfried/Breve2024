# Gustavo Landfried: github.com/glandfried

base2 = function(number, size){
  res = numeric(size)
  j = 1
  while (number > 0){
    res[j] = number %% 2
    number = number %/% 2
    j = j + 1
  }
  return(res)
}


sampler_from_joint <- function(Sjoint=c(0.17, 0.32, 0.27, 0.72),          # Sensibilidad
                               Xjoint=c(0.1275, 0.2025, 0.3025, 0.7275),  # Especificidad
                               P=c(0.17, 0.15, 0.2, 0.125),               # Prevalencias
                               n_individuals=300){                        # Tamaño de muestras
    n_tests = log(length(Sjoint),2)

    res = matrix(nrow=(n_individuals*length(P)),  ncol=1+n_tests+1)
    columns = rep("", n_tests+2)
    columns[1] = "Population"; columns[n_tests+2] = "State"
    for (i in seq(2,n_tests+1)){columns[i] = paste0("Test",i-1)}
    colnames(res) = columns

    # Indice de resultado de las muestras
    #
    ir = 1

    # Itera por país.
    #
    for (ip in seq(length(P))){ # ip = 1

        # Generar los infectados
        #
        n_positives = rbinom(n=1, prob=P[ip], size=n_individuals )

        # Genera los diagnósticos de infectados y no infectados
        #
        joint_diagnosis_infected = rmultinom(n=1, prob=Sjoint, size=n_positives)
        joint_diagnosis_non_infected = rmultinom(n=1, prob=Xjoint, size=n_individuals-n_positives)

        print(sum(joint_diagnosis_infected )+sum(joint_diagnosis_non_infected ))


        for (id in seq(length(Sjoint)) ){ # id=5
            diagnosis = base2(id-1,n_tests)

            # Agrega a diagnósticos a res
            #
            j=1; while (j <= joint_diagnosis_infected[id]){
                res[ir,] = c(ip, diagnosis, 1)
                ir = ir + 1
                j=j+1
            }
            j=1; while (j <= joint_diagnosis_non_infected[id]){
                res[ir,] = c(ip, diagnosis, 0)
                ir = ir + 1
                j=j+1
            }

        }
    }
    return(res)
}








