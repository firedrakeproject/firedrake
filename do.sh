sed -i 's|http://archive.ubuntu.com/ubuntu|http://www.mirrorservice.org/sites/archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources
apt-get update
apt-get -y install git python3 gnupg2 curl ca-certificates
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | apt-key add -
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" > /etc/apt/sources.list.d/cuda.list
apt-get update
apt-get -y install $(python3 ./firedrake-configure --arch default --gpu-arch cuda --show-system-packages)
apt-get -y install fonts-dejavu graphviz graphviz-dev parallel poppler-utils python3-venv

cd /opt
git clone --depth 1 https://gitlab.com/petsc/petsc.git
cd petsc
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
python3 ../firedrake-repo/scripts/firedrake-configure --arch default --gpu-arch cuda --show-petsc-configure-options | xargs -L1 ./configure --with-make-np=8 --download-slepc
make PETSC_DIR=/opt/petsc PETSC_ARCH=arch-firedrake-default-cuda
make PETSC_DIR=/opt/petsc PETSC_ARCH=arch-firedrake-default-cuda check
export PETSC_DIR="/opt/petsc"
export PETSC_ARCH="arch-firedrake-default-cuda"
export SLEPC_DIR="/opt/petsc/arch-firedrake-default-cuda"

cd /opt
export $(python3 ./firedrake-configure --arch default --gpu-arch cuda --show-env)
python3 -m venv venv
. venv/bin/activate
pip cache purge
pip install "$PETSC_DIR"/src/binding/petsc4py
pip install -r ./firedrake/requirements-build.txt
pip install --no-build-isolation --no-deps "$PETSC_DIR"/"$PETSC_ARCH"/externalpackages/git.slepc/src/binding/slepc4py
pip install --no-deps git+https://github.com/NGSolve/ngsPETSc.git netgen-mesher netgen-occt
pip install --verbose --no-build-isolation --no-binary h5py './firedrake-repo[check]'

