import os
from pathlib import Path
print("Directorio actual:", os.getcwd())

current_dir = Path.cwd()
project_root = current_dir
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
    os.chdir("..")

print(f"Raíz del proyecto: {project_root}")
print("CWD cambiado a raíz del proyecto")


