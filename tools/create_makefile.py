#!/usr/bin/python
import re
import os
import sys
import glob

# List of dependencies we don't want in makefiles. Dependencies starting
# with these strings will be dropped.
dep_exclusions = [
    # Because switching on and off mba changes the module files loaded, these
    # are special-cased in the Makefiles.
    "../include/mba2d_module.mod",
    "../include/mba3d_mba_nodal.mod",
    # Remove dependencies on confdefs.h because it causes
    # lots of spurious rebuilds.
    "../include/confdefs.h",
    # Get rid of absolute paths.
    # We are only interested in dependencies from within the
    # Fluidity tree  so we dump the ones from outside.
    "/"
]


class dependency_list(object):

    '''Class to store and process the dependencies of a single .o and its
    associated .mod(s) if any.'''

    def __init__(self, obj, source, dep_strings):
        self.obj = obj  # .o target of rule.
        self.source = source  # .F90 file

        self.targets = set()  # remaining targets (.mod)
        self.deps = set()  # dependencies (.mod, .h and included .F90s)

        intargets = True  # True while we are processing targets. False once we
                       # proceed to dependencies.

        for dep in dep_strings:
            for d in dep.split():
                # Drop continuaton characters.
                if d == "\\":
                    continue

                if intargets:
                    if d[-1] == ":":
                        self.targets.add(d[:-1])
                        intargets = False
                    else:
                        self.targets.add(d)

                else:
                    self.deps.add(d)
        # Treat the .o and the .F90 specially.
        self.targets.remove(obj)
        self.deps.remove(source)

        # Gfortran produces spurious circular dependencies if the .F90
        # contains both a module and routines which use that module.
        self.deps.difference_update(self.targets)

    def remove_dep_by_rule(self, f):
        '''remove_dep_by_rule(self, f)

        Remove and dependency d for which f(d) is true.'''
        discards = set()
        for dep in self.deps:
            if f(dep):
                discards.add(dep)
        self.deps.difference_update(discards)

    def as_strings(self):
        '''as_strings(self)

        produce the actual makefile rules in string form'''

        out = []

        # Special rule to fake .mod dependency on .o.
        for t in self.targets:
            if t.endswith(".mod"):
                out += wrap(t + ": " + self.obj) + ["\n"]
                out += ["\t@true\n", "\n"]

        # Main rule.
        out += wrap(self.obj + " "
                    + " ".join(self.targets) + ": "
                    + self.source + " "
                    + " ".join(sorted(self.deps))) + ["\n", "\n"]

        return out


def wrap(string):
    """wrap(string)

    Linewrap dependencies string according to makefile conventions"""
    linelen = 78

    lines = []
    line = ""

    for substring in string.split():
        # Always put one thing in a line:
        if len(line) == 0:
            line = substring
        elif len(line) + len(substring) + 1 <= linelen:
            line += " " + substring
        else:
            lines.append(line + " \\\n")
            line = "   " + substring
    # Put the last line in.
    lines.append(line)

    return lines


def trysystem(command):
    '''trysystem(command)

    Wrapped version of the os.system command which causes an OSError if
    the command fails.'''
    if not(os.system(command) == 0):
        raise OSError


def create_refcounts():
    '''create_refcounts()

    Produce all the generated reference counting Fortran source. This
    currently only has effect in the femtools directory.
    '''

    refcounts_raw = os.popen("grep include.*Ref *.F90").read()

    refcounts = re.findall(r'"Reference_count.*?"', refcounts_raw)

    if len(refcounts) > 0:
        trysystem("make " + " ".join(refcounts))


def generate_dependencies(fortran):
    '''generate_dependencies(fortran)

    Given a list of Fortran source files, generate the actual list of
    makefile dependencies.
    '''
    import os.path

    setsize = len(fortran) + 1  # Make sure loop executes once.

    dependencies = {}
    # Loop as long as we are making progress.
    while len(fortran) < setsize and len(fortran) > 0:
        print("Generating dependencies, %d to go." % len(fortran))
        setsize = len(fortran)

        discards = set()
        for f in fortran:
            print "  " + f
            obj = os.path.splitext(f)[0] + ".o"
            os.system("rm " + obj + " 2>/dev/null || true")

            pipe = os.popen(
                'make GENFLAGS="-cpp -M -MF ' + obj + '_dependencies" ' + obj)
            pipe.readlines()
            if pipe.close() is None:

                this_deps = dependency_list(
                    obj,
                    f,
                    file(obj + "_dependencies", "r").readlines())

                # Remove unwanted dependencies
                for dep in dep_exclusions:
                    this_deps.remove_dep_by_rule(lambda x:
                                                 x.startswith(dep))
                dependencies[f] = this_deps
                # split_module_dependency(
                #    strip_absolute_paths(this_deps))+["\n"]
                discards.add(f)
            os.system("rm " + obj + "_dependencies 2>/dev/null || true")
        fortran.difference_update(discards)

    if len(fortran) > 0:
        print "Failed to generate all dependencies. Failures:"
        print str(fortran)
        raise OSError

    dep_strings = [
        "# Dependencies generated by create_makefile.py. DO NOT EDIT\n"]

    # Sort the output by filename.
    files = dependencies.keys()
    files.sort()

    for f in files:
        dep_strings += dependencies[f].as_strings()

    return dep_strings


def handle_options():
    from optparse import OptionParser
    optparser = OptionParser(usage='usage: %prog [options] <filename>',
                             add_help_option=True,
                             description="""Use gfortran (>=4.5) to automatically
                             generate makefile module dependencies.""")

    optparser.add_option("--exclude",
                         help="list of .F90 files to exclude from consideration.",
                         action="store", type="string", dest="exclude", default="")

    optparser.add_option("--test",
                         help="Cause a failure if the dependencies would" +
                         " change. This is used to detect user failure to" +
                         " run make makefiles", action="store_true",
                         dest="test", default=False)

    (options, argv) = optparser.parse_args()

    return options

if __name__ == '__main__':
    options = handle_options()

    sys.stderr.write("Clobbering previous dependencies\n")
    os.system("mv Makefile.dependencies Makefile.dependencies.old")
    try:
        trysystem("touch Makefile.dependencies")

        sys.stderr.write("Making clean\n")
        trysystem("make clean")

        sys.stderr.write("Listing F90 files\n")
        fortran = set(glob.glob("*.[fF]90"))\
            .difference(set(options.exclude.split()))

        sys.stderr.write("Creating reference counts\n")
        create_refcounts()

        dependencies = generate_dependencies(fortran)

        file("Makefile.dependencies", 'w').writelines(dependencies)

        if options.test:
            if os.path.isfile("Makefile.dependencies"):
                # Check that nothing has changed.
                trysystem(
                    "diff -q Makefile.dependencies.old Makefile.dependencies")
            elif os.path.isfile("Makefile.dependencies.old"):
                raise OSError("Dependencies have disappeared!")

    except:
        # If anything fails, move the previous makefiled dependencies back.
        os.system("mv Makefile.dependencies.old Makefile.dependencies")

        if options.test:
            print "**********************************************************"
            print "Testing make makefiles failed.\n" +\
                "This may indicate that make makefiles should have been\n" +\
                "run and the resulting Makefile.dependencies committed"
            print "**********************************************************"

        # Now re-raise whatever the exception was.
        raise

    # On success, remove the old file.
    os.system("rm Makefile.dependencies.old")
