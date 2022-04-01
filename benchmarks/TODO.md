## Connor remarks

So `asv` assumes that it can always install a package in a new (conda or virtualenv) environment with a differently checked out commit. This will not work for us because installing Firedrake is *rather more complicated* than just `pip install firedrake`. We also can't check a range of commits because we would need to check out the corresponding commits in TSFC, PyOP2 etc and this won't work.

The approach that I'm pursuing at the moment is to run `VIRTUALENV_SYSTEM_SITE_PACKAGES=1 asv run` since we can then avoid reinstalling things into the new environment. This does not seem to work though when we are inside a Firedrake virtual environment (`spack` to the rescue?).

### Outstanding tasks

- Once we have a working spack Firedrake install try this again, making sure to only checkout `HEAD`.
