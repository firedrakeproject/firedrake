==========================================
For Developers: Branding the web interface
==========================================

The header page and footer page is custom-configurable. By default, it contains
the Firedrake logo and copyright information for Firedrake.

To change the header and footer, specify a path for ``header.html``
and ``footer.html``

Name the header file as ``header.html``, the footer file as ``footer.html``
(case-sensitive). In the module containing the parameters instance, add
attributes ``header_path`` and ``footer_path`` to the module. The web interface
will then include the paths and render the page using the header and footer
files in the path specified by the user.
