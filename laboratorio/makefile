requirements.txt:
	source venv-notbooks/bin/activate
	pip install requirements.txt

install:
	source venv-notbooks/bin/activate
	pip install git+https://github.com/asherbender/bayesian-linear-model
	pip install scipy
	pip install numpy
	pip install pandas
	pip install plotnine
	pip freeze > requirements.txt

activate: venv-notbooks
	source venv-notbooks/bin/activate
	## Para que aparezca entre los kernels de visual studio:
	# pip install ipykernel
	# ipython kernel install --user --name=venv-notbooks
	## Actualizar visual studio

venv-notbooks:
	python3 -m venv venv-notbooks

