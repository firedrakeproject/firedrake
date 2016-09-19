========================================
Configuring Parameters via web interface
========================================

Firedrake's parameters :class:`~firedrake.parameters.Parameters` can be
configured either programmatically in python, or through a graphical interface
in the web browser. Configuring Parameters via the web interface is enabled by
the ``firedrake-config`` script. To use the script, the parameters to be
configured must be of instance :class:`~firedrake.parameters.Parameters`
and named ``parameters`` inside the module.

Launching the configuration GUI
===============================

To launch the configuration GUI, execute this in shell:

.. code-block:: bash

    firedrake-config MODULE_NAME

This command will run a server at ``http://0.0.0.0:5000`` and generates a
graphical interface for users to configure the parameters.

Once running, the configuration can be edited in the browser. Validation of the
inputs is performed after input has been changed, with visual feedback of
border colouring of green for success and red for failure. In addition, a set
of parameters could only be saved if all validates pass, so that it is not
possible to create an invalid set of configuration parameters.

User may wish to see advanced setting and configure them. This can be done by
clicking on ``Show more options`` button. Similarly, advanced setting can be
hidden using the ``Show fewer options`` button.

For loading existing configurations, simply click on the ``Load`` button. An
input would be displayed prompting the user to upload a configuration file to
the server. The file would be loaded after clicking on ``Submit`` button. Note
that the file does not load automatically when selected, clicking on the
``Submit`` button is necessary.

Saving Parameters
=================

By clicking on ``Save``, parameters will be saved into ``parameters.json``, to save
the file to other paths, there are two options.

Option 1: Specify a config file name from the command line. This is done by
adding an extra argument ``--config_file`` when executing ``firedrake-config``,
i.e.

.. code-block:: bash

    firedrake-config MODULE_NAME --config_file FILE_PATH

This will enable the server to save parameters to the specified path when
clicking on ``Save``

Option 2: Use ``Save as`` button on the web page. After clicking on ``Save as``,
there will be a link on the web page for Downloading the configuration file.
The user can then save the file to any path they wish and store the file.

Loading Parameters to Firedrake after configuration
===================================================

Once the user has successfully configured the parameters through the web
interface and saved the configuration file, the configuration needs to be
loaded to Firedrake when using firedrake. To load a parameter file to a
Parameters instance, use
:meth:`~firedrake.parameters.Parameters.load` and pass the filename as
the parameter. For example

.. code-block:: python

    parameters.load("parameters.json")

To save the parameters instance to a file, please use
:meth:`~firedrake.parameters.Parameters.dump`

Branding
========

The header page and footer page is custom-configurable. By default, it contains
the Firedrake logo and copyright information for Firedrake.

To change the header and footer, there are two options.

Option 1: Edit the ``header.html`` and ``footer.html`` in the ``templates``
folder.

After installation, the templates files for the web interface will be placed
into the ``templates`` folder. To change the header and footer, simply replace
``header.html`` and ``footer.html`` with your custom page with the same file
name.

Option 2: Specify a path for ``header.html`` and ``footer.html``

Name the header file as ``header.html``, the footer file as ``footer.html``
(case-sensitive). In the module containing the parameters instance, add
attributes ``header_path`` and ``footer_path`` to the module. The web interface
will then include the paths and render the page using the header and footer
files in the path specified by the user.
