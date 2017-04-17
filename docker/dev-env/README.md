# Firedrake development environment image

This image provides a development environment for building Firedrake
and its dependencies. It does not provide an installation of Firedrake
but is intended for users who want to build their own version of
Firedrake. It also serves as a base image for
<https://registry.hub.docker.com/u/firedrakeproject/dev>, which does
provide the development version of Firedrake.

To launch the container:

    docker run -t -i firedrakeproject/dev-env:latest

We do provide a helper script (firedrake.conf) which is sourced in this
container to compile Firedrake automatically:

    update_firedrake

If you want to have access to the source code and build files in the container
on the host machine then run:

    docker run -v $(pwd)/build:/home/firedrake/build -t -i firedrakeproject/dev-env:latest
    
If you would like to have another directory on the host shared into the
container then run:

    docker run -v $(pwd)/shared:/home/firedrake/shared -t -i firedrakeproject/dev-env:latest
