# Builds a Docker image with the necessary libraries for compiling
# Firedrake.  The image is at:
#
#     https://registry.hub.docker.com/u/firedrakeproject/
#
# Authors:
# Florian Rathgeber <florian.rathgeber@gmail.com>
#
# Based on the fenics:dev-env image (https://github.com/FEniCS/docker/)

FROM phusion/baseimage:0.9.17
MAINTAINER Firedrake project <firedrake@firedrakeproject.org>

# Install build environment
RUN apt-get -qq update && \
    apt-get -y install \
      build-essential \
      cmake \
      gfortran \
      git-core \
      gmsh \
      libopenblas-dev \
      libhdf5-openmpi-dev \
      liblapack-dev \
      libopenmpi-dev \
      libpcre3-dev \
      mercurial \
      openmpi-bin \
      python-dev \
      python-pip \
      wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install PETSc from source
RUN cd /tmp && \
    wget -nc https://bitbucket.org/mapdes/petsc/get/firedrake.tar.bz2 && \
    tar -xjf firedrake.tar.bz2 && \
    cd mapdes-petsc-* && \
    ./configure --COPTFLAGS="-O3" \
                --CXXOPTFLAGS="-O3" \
                --FOPTFLAGS="-O3" \
                --with-blas-lib=/usr/lib/libopenblas.a \
                --with-lapack-lib=/usr/lib/liblapack.a \
                --with-c-support \
                --with-debugging=0 \
                --with-shared-libraries \
                --download-chaco \
                --download-ctetgen \
                --download-hypre \
                --download-triangle \
                --prefix=/usr/local && \
     make && \
     make install && \
     rm -rf /tmp/*

# Install swig from source
ENV SWIG_VERSION 3.0.6
RUN cd /tmp && \
    wget -nc http://downloads.sourceforge.net/swig/swig-${SWIG_VERSION}.tar.gz && \
    tar xf swig-${SWIG_VERSION}.tar.gz && \
    cd swig-${SWIG_VERSION} && \
    ./configure && \
    make && \
    make install && \
    rm -rf /tmp/*

# Install Python dependencies
ENV PETSC_DIR /usr/local
RUN cd /tmp && \
    pip install \
      https://bitbucket.org/mapdes/petsc4py/get/firedrake.tar.bz2 \
      cachetools \
      Cython \
      decorator \
      mpi4py \
      networkx \
      numpy \
      psutil \
      pytest \
      six \
      sympy

# Set up user so that we do not run as root
RUN useradd -m -s /bin/bash -G sudo,docker_env firedrake && \
    echo "firedrake:docker" | chpasswd && \
    echo "firedrake ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# See https://github.com/phusion/baseimage-docker/issues/186
RUN touch /etc/service/syslog-forwarder/down

ENV HOME /home/firedrake

# Our helper script
COPY firedrake.conf $HOME/firedrake.conf
# and the README
COPY WELCOME $HOME/WELCOME
RUN chown firedrake:firedrake $HOME/WELCOME && \
    chown firedrake:firedrake $HOME/firedrake.conf

# OpenBLAS threads should be 1 to ensure performance
RUN echo "export OPENBLAS_NUM_THREADS=1" >> $HOME/.bashrc && \
    echo "export OPENBLAS_VERBOSE=0" >> $HOME/.bashrc

# This makes sure we launch with ENTRYPOINT /bin/bash into the home directory
RUN echo ". $HOME/firedrake.conf" >> $HOME/.bashrc && \
    echo "cat $HOME/WELCOME" >> $HOME/.bashrc

WORKDIR $HOME
ENTRYPOINT ["/sbin/my_init","--quiet","--","/sbin/setuser","firedrake","/bin/sh","-c"]
CMD ["/bin/bash"]
