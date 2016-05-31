# Dpckerfile to build Firedrake via firedrake-install
FROM ubuntu:xenial
MAINTAINER Tim Greaves <tim.greaves@imperial.ac.uk>

# Tweak user information to match your runtime environment, if desired
ENV FIREDRAKE_UID 1000
ENV FIREDRAKE_GID 1000
ENV FIREDRAKE_HOME /firedrake

# Jenkins checks out the relevant files for testing and makes them the 
# context for this container build; we want them all in the working
# directory of the container.
RUN mkdir /firedrake
COPY scripts/firedrake-install /firedrake
WORKDIR /firedrake

# Firedrake install script doesn't use apt-get -y ; fix at system level
RUN echo "APT::Get::Assume-Yes \"True\";" >> /etc/apt/apt.conf.d/50assumeyes

# Update the image, and install required base packages for firedrake-install
RUN apt-get update
RUN apt-get dist-upgrade
RUN apt-get install sudo python-minimal

# Add the Firedrake user for the build, with sudo access and ownership of the
# added repo contents which make up the build source
RUN groupadd -g $FIREDRAKE_GID firedrake
RUN useradd -d $FIREDRAKE_HOME -u $FIREDRAKE_UID -g $FIREDRAKE_GID firedrake
RUN echo "firedrake ALL=NOPASSWD: ALL" >> /etc/sudoers
RUN chown -R firedrake.firedrake /firedrake

# Switch to the new 'firedrake' user for building
USER firedrake

# Build using firedrake install; this is run twice, the second following an
# exit with error after virtualenv has been installed
RUN /firedrake/firedrake-install --minimal-petsc --disable-ssh || /firedrake/firedrake-install --minimal-petsc --disable-ssh

# Set environment as it would be after activating the virtualenv
ENV PATH "/firedrake/firedrake/bin:$PATH"
ENV VIRTUAL_ENV "/firedrake/firedrake"
