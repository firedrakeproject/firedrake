# Many preconditioners will want to stash something the first time
# they are set up (e.g. constructing a sub-KSP or PC context, assemble
# a constant matrix and do something different at each subsequent
# iteration.
#
# This creates a base class from which we can inherit, just providing
# the two alternate setUp routines and the apply method.  It makes
# it a bit cleaner to write down new PCs that follow this pattern.


class InitializedPC(object):
    @property
    def initialized(self):
        return hasattr(self, "_initialized") and self._initialized

    @initialized.setter
    def initialized(self, value):
        self._initialized = value

    def initialSetUp(self, pc):
        pass

    def subsequentSetUp(self, pc):
        pass

    def setUp(self, pc):
        if self.initialized:
            self.subsequentSetUp(pc)
        else:
            self.initialSetUp(pc)
            self.initialized = True
