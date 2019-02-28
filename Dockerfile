# DockerFile for a firedrake

FROM ubuntu:18.04

# This DockerFile is looked after by
MAINTAINER David Ham <david.ham@imperial.ac.uk>

# Update and install required packages for Firedrake
USER root
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata \
    && apt-get -y install curl vim \
                 openssh-client build-essential autoconf automake \
                 cmake gfortran git libblas-dev liblapack-dev \
                 libmpich-dev libtool mercurial mpich\
                 python3-dev python3-pip python3-tk python3-venv \
                 zlib1g-dev libboost-dev \
    && rm -rf /var/lib/apt/lists/*


# Set up user so that we do not run as root
RUN useradd -m -s /bin/bash -G sudo firedrake && \
    echo "firedrake:docker" | chpasswd && \
    echo "firedrake ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    ldconfig

USER firedrake
WORKDIR /home/firedrake

# Now install firedrake
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
RUN bash -c "python3 firedrake-install --no-package-manager --disable-ssh --venv-name=firedrake"
