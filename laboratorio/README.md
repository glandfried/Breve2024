# Notas sobre como editar y publicar

## Instalación

### En un browser

Para correr los notebooks debe estar instalado jupyter:

```
pip3 install jupyter
```

Y luego ejecutar de consola:

jupyter notebook

### Desde una IDE

Usualmente las IDEs como VSCode o IntelijIDEA viene con soporte interno para notebooks.


## Redacción de notebooks

La confección de notebook se puede realizar escribiendo y codeando como si fuese para uno, de manera completa.
Antes de publicarlo se le pasa un "compilador" que hace varias cosas:

  - Limpia las salidas.
  - Limpia los números de ejecución
  - Ocultar celdas con el tag "ocultar" (ver detalle más abajo)
  - Transforma las celdas de tipo "raw" a "code", util para dejar código que no compile y aun así podér hacer "Run all"

Se generará un notebook con el mismo nombre agregandole `-published`.

Para compilar simplemente correr `python3 build.py`.

### Agregar tags

En Jupyter Notebook para editar los tags y agragar "ocultar", se recomienda además agregar un comentario al inicio de la celda del estilo "#esta celda será ocultada"
Para habilitar la edicción de tags  `View -> Cell Toolbar -> Tags`.


### Limpiar el archivo

Con el siguiente comando se puede limpiar el archivo, requiere instalar (TODO completar con lo necesario para tener el nbconvert)

El comando es similar a este:
```
jupyter nbconvert \
	--TagRemovePreprocessor.enabled=True \
	--TagRemovePreprocessor.remove_cell_tags ocultar \
	--ClearOutputPreprocessor.enabled=True \
	--ClearMetadataPreprocessor.enabled=True \
	--no-prompt \
	--to notebook \
	--output output.ipynb \
	pruebas.ipynb
```

### Extras

Si uno quisiera tener celdas que permitan tener código incompleto, pero que esto no choque con la ejecución de la notebook se puede poner en formato "raw NBconvert" y luego correr un script para que cambie la celda a formato código.

Script (archivo raw_to_code.py)

```
import json
import sys


in_file = sys.argv[1]

with open(in_file) as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'raw':
        cell['cell_type'] = 'code'
        # cell['source'] = ''
        cell['outputs'] = []
        cell['execution_count'] = None

out_file = in_file[:-6] + "-clean.ipynb"

with open(out_file, 'w') as f:
    json.dump(nb, f)
```


## Cambio de cuatrimestre:

Fuente: https://stackoverflow.com/questions/13716658/how-to-delete-all-commit-history-in-github

1. Checkout

  ```
  git checkout --orphan latest_branch
  ```

1. Add all the files

  ```
	git add -A
  ```

1. Commit the changes

  ```
  git commit -am "commit message"
  ```

1. Delete the branch

  ```
  git branch -D main
  ```

1. Rename the current branch to main

  ```
  git branch -m main
  ```

1. Finally, force update your repository

  ```
  git push -f origin main
  ```
