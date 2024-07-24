import subprocess
import json
import sys

list_of_notebooks = ["00-previa/notebook_00_herramientas.ipynb", "01-modelos/notebook_01_ejercicios.ipynb"]

for name_notebook in list_of_notebooks: # name_notebook = "01-modelos/notebook_01_ejercicios.ipynb"
    # Comando para limpiar el notebook
    #cmd = f"""jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags ocultar --ClearOutputPreprocessor.enabled=True  --no-prompt --to notebook --output ../temp.ipynb {name_notebook}"""
    cmd = f"""jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags ocultar --no-prompt --to notebook --output ../temp.ipynb {name_notebook}"""
    print(subprocess.run(cmd, shell=True))

    # Abre el archivo temporario de salida, limpio.
    with open("temp.ipynb") as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb['cells']:

        # Convierte las celdas de tipo 'raw' en 'code'
        if cell['cell_type'] == 'raw':
            cell['cell_type'] = 'code'
            if 'vscode' in cell['metadata']:
                cell['metadata']['vscode']['languageId'] = 'Python'
            # cell['source'] = ''
            cell['outputs'] = []
            cell['execution_count'] = None
        new_cells.append(cell)

    nb["cells"] = new_cells
    out_file = name_notebook[:-6] + "-published.ipynb"

    with open(out_file, 'w') as f:
        json.dump(nb, f)

