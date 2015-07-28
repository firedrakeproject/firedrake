# Builds a Docker image with the Firedrake development version built
# from git sources. The image is at:
#
#     https://registry.hub.docker.com/u/firedrakeproject/
#
# Authors:
# Florian Rathgeber <florian.rathgeber@gmail.com>
#
# Based on the fenics:dev image (https://github.com/FEniCS/docker/)

FROM firedrakeproject/dev-env:latest
MAINTAINER Firedrake Project <firedrake@firedrakeproject.org>

USER firedrake
ENV HOME /home/firedrake

RUN /bin/bash -c ". $HOME/firedrake.conf && \
                  update_firedrake"

COPY WELCOME $HOME/WELCOME
USER root
ENTRYPOINT ["/sbin/my_init","--quiet","--","/sbin/setuser","firedrake","/bin/sh","-c"]
CMD ["/bin/bash"]
