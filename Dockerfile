# Builds a Docker image with Firedrake and its dependencies:
#
#   http://firedrakeproject.org
#
# This image is hosted on Docker Hub:
#
#   https://registry.hub.docker.com/u/firedrakeproject/
#
# Author: Florian Rathgeber <florian.rathgeber@gmail.com>

FROM phusion/baseimage:0.9.18
MAINTAINER Florian Rathgeber <florian.rathgeber@gmail.com>

# Install required packages (we cannot use the firedrake-install script for
# this since we have no Python yet)
RUN apt-get -qq update && \
    apt-get -y install python

# Copy Firedrake install script
COPY scripts/firedrake-install /usr/bin/firedrake-install

# Install Firedrake and dependencies
RUN firedrake-install --global --unattended --disable-ssh --log && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# See https://github.com/phusion/baseimage-docker/issues/186
RUN touch /etc/service/syslog-forwarder/down

CMD ["/bin/bash"]
