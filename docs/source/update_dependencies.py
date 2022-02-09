import yaml
import os
import shutil
import os
import subprocess

log_stream = subprocess.check_output(
    ['python3', '../scripts/firedrake-install', '--show-dependencies'], stderr=subprocess.STDOUT, env=os.environ).decode()

rows = log_stream.split('\n')
start_index = rows.index('START LOG DATA DOCUMENTATION') + 1
end_index = rows.index('END LOG DATA DOCUMENTATION')

log_yaml = "\n".join(rows[start_index:end_index])

data = yaml.safe_load(log_yaml)
print(type(data))

# creating new folder containing the .rst files with the dependencies
try:
    os.mkdir('source/dependencies')
except FileExistsError:
    shutil.rmtree('source/dependencies')
    os.mkdir('source/dependencies')

# dealing with default dependencies (system dependencies, wheel_blacklist, parallel_packages)
for key, val in data.items():

    with open(f'source/dependencies/{key}.rst', 'w') as out_file:

        for requirement in val:
            out_file.write(f"* {requirement} \n")

# dealing with requirements-git
with open('../requirements-git.txt') as requirements_file:
    lines = requirements_file.readlines()

    with open('source/dependencies/firedrake_dependencies.rst', 'w') as out_file:

        # parse the information
        for line in lines:
            library = line[line.find('=') + 1:-1]
            link = line[line.find('+') + 1: -1]

            # the petsc doesn't need to appear
            if (library != 'petsc'):
                out_file.write(f"* `{library} <{link}>`_ \n")
