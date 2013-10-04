from distutils.core import setup
import os
import os.path
import glob

try:
  destdir = os.environ["DESTDIR"]
except KeyError:
  destdir = ""

setup(
      name='dxdiff',
      version='1.0',
      description="An XML aware diff tool.",
      author = "The ICOM team",
      author_email = "fraser.waters08@imperial.ac.uk",
      url = "http://amcg.ese.ic.ac.uk",
      packages = ['dxdiff'],
      scripts=["dxdiff/dxdiff"],
     )

