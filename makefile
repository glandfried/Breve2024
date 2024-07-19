all:
	echo "Leer el MAKEFILE!"

submodule:
	make -C auxiliar


RELEASE/2023.1/0-programa.pdf:
	make -C RELEASE/2023.1/

source:
	git remote set-url origin git@git.exactas.uba.ar:bayes/seminario.git

mirror:
	git remote set-url --add origin git@github.com:BayesDeLasProvinciasUnidasDelSur/curso.git

fork:
	echo "git@github.com:glandfried/Breve2024.git"
