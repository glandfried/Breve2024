import subprocess
import json
import sys

list_of_notebooks = ["00/notebook_00_herramientas.ipynb"]

for name_notebook in list_of_notebooks:
    # Comando para limpiar el notebook
    cmd = f"""jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags ocultar --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --no-prompt --to notebook --output ../temp.ipynb {name_notebook}"""
    print(subprocess.run(cmd, shell=True))

    # Abre el archivo temporario de salida, limpio.
    with open("temp.ipynb") as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb['cells']:

        # Si tiene tag 'ocultar', no ejecuta el procedimiento
        # (En qu'e caso hace falta revisar esto? Ya debería estar borrada, no?)
        if "tags" in cell["metadata"] and "ocultar" in cell["metadata"]["tags"]:
            continue

        # Convierte las celdas de tipo 'raw' en 'code'
        if cell['cell_type'] == 'raw':
            cell['cell_type'] = 'code'
            # cell['source'] = ''
            cell['outputs'] = []
            cell['execution_count'] = None
        new_cells.append(cell)

    nb["cells"] = new_cells
    out_file = name_notebook[:-6] + "-published.ipynb"

    with open(out_file, 'w') as f:
        json.dump(nb, f)

