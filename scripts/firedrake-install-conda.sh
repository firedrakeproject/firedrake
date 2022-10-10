#!/usr/bin/env bash

# Parse command-line arguments
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        name="${1/--/}"
        declare "$name"="$2"
        shift
    fi
    shift
done

# For some odd reason, on MacOS we need to tell it where to find `ar`,
# otherwise it might try to find an `llvm-ar` that doesn't necessarily exist.
if [[ $(uname -s) == Darwin ]]; then
    export AR=/usr/bin/ar
fi

# Set the directory where we will build Firedrake.
if [[ -z $prefix ]]; then
    if [[ $(command -v pyenv) ]]; then
        declare prefix=$(pyenv root)/versions/$(pyenv version-name)/envs/firedrake
    else
        declare prefix=${HOME}/.conda/envs/firedrake
    fi
fi
echo "Install location: $prefix"

# Make a conda environment that excludes HDF5 & mpich (PETSc will build these)
conda list --explicit | grep -v mpich | grep -v hdf5 | grep -v h5py | grep -v asn1crypto > spec-file.txt
conda create --prefix $prefix --file spec-file.txt

# Install Firedrake into a subdirectory of the conda env
cd $prefix
if [[ -z $script ]]; then
    curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    declare script=${PWD}/firedrake-install
fi
source activate firedrake
python $script --no-package-manager --venv-name=firedrake-venv

# Make an activate script
mkdir -p etc/conda/activate.d
cp firedrake-venv/bin/activate etc/conda/activate.d/01_firedrake-venv.sh

# Make a custom deactivate script
mkdir -p etc/conda/deactivate.d
echo deactivate > etc/conda/deactivate.d/01_firedrake-venv.sh
