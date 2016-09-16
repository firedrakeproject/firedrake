=====================================
Configuring Parameters via web server
=====================================

The ``firedrake-config`` script allows users to configure parameters via a 
web server. The parameters to be configured must be of instance
:class:`firedrake.parameters.Parameters` and named ``parameters`` inside the
module. 

Running the server
==================

To run the web server, execute this in shell:

.. code-block:: bash

    firedrake-config MODULE_NAME

This command will run a server at ``http://0.0.0.0:5000`` and generates a
graphical interface for users to configure the parameters.

Users can edit the parameters in the browser. Once a parameter has been
changed, it would be validated using the metadata stored in the
:class:`firedrake.parameters.Parameters` instance. If the input is valid, a
the border of the input would become green, otherwise it would be red.

Users can also choose to show more or fewer options if the visible levels of
parameters have been stored beforehand. This is easily done by clicking on
buttons ``Show more options`` and ``Show fewer options``

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



