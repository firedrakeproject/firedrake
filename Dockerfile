# Dpckerfile to build Firedrake via firedrake-install

FROM ubuntu:xenial
MAINTAINER Tim Greaves <tim.greaves@imperial.ac.uk>

# Tweak user information to match your runtime environment, if desired
ARG FIREDRAKE_UID='1001'
ARG FIREDRAKE_GID='1001'
ARG FIREDRAKE_HOME='/firedrake'

# Offer the option of overriding where firedrake-install is picked up from
ARG FIREDRAKE_INSTALL_SCRIPT_DIR='scripts/'

# Optionally, pass in flags to be passed to firedrake-install
ARG FIREDRAKE_INSTALL_FLAGS
# and extra packages to be installed in the container
ARG EXTRA_PACKAGES

# Jenkins checks out the relevant files for testing and makes them the 
# context for this container build; we want them all in the working
# directory of the container.
RUN mkdir $FIREDRAKE_HOME
COPY $FIREDRAKE_INSTALL_SCRIPT_DIR/firedrake-install $FIREDRAKE_HOME
WORKDIR $FIREDRAKE_HOME

# Update the image, and install required base packages for firedrake-install
RUN apt-get -y update
RUN apt-get -y dist-upgrade
RUN apt-get -y install sudo python-minimal $EXTRA_PACKAGES

# Add the Firedrake user for the build, with sudo access and ownership of the
# added repo contents which make up the build source
RUN groupadd -g $FIREDRAKE_GID firedrake
RUN useradd -d $FIREDRAKE_HOME -u $FIREDRAKE_UID -g $FIREDRAKE_GID firedrake
RUN echo "firedrake ALL=NOPASSWD: ALL" >> /etc/sudoers
RUN chown -R firedrake.firedrake $FIREDRAKE_HOME

# Switch to the new 'firedrake' user for building
USER firedrake

# Build using firedrake install; this is run twice, the second following an
# exit with error after virtualenv has been installed
RUN $FIREDRAKE_HOME/firedrake-install $FIREDRAKE_INSTALL_FLAGS || $FIREDRAKE_HOME/firedrake-install $FIREDRAKE_INSTALL_FLAGS

# Set environment as it would be after activating the virtualenv
ENV PATH "$FIREDRAKE_HOME/firedrake/bin:$PATH"
ENV VIRTUAL_ENV "$FIREDRAKE_HOME/firedrake"
