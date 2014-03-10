"""Launch a livereload server serving up the html documention. Watch the
sphinx source directory for changes and rebuild the html documentation. Watch
the pyop2 package directory for changes and rebuild the API documentation.

Requires livereload_ ::

  pip install git+https://github.com/lepture/python-livereload

.. _livereload: https://github.com/lepture/python-livereload"""

from livereload import Server

server = Server()
server.watch('source', 'make html')
server.watch('../../pyop2', 'make apidoc')
server.serve(root='build/html', open_url=True)
