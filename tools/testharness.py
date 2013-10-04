#!/usr/bin/env python

import sys
import os
import os.path
import glob
import time
import fluidity.regressiontest as regressiontest
import traceback
import threading
import xml.parsers.expat
import string


sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]),
                                os.pardir, "python"))
try:
    import xml.etree.ElementTree as etree
except ImportError:
    import elementtree.ElementTree as etree


class TestHarness:

    def __init__(self, length="any", parallel=False, exclude_tags=None,
                 tags=None, file="", from_file=None, verbose=True,
                 justtest=False, valgrind=False, backend=None, no_pbs=False):
        self.tests = []
        self.verbose = verbose
        self.length = length
        self.parallel = parallel
        self.passcount = 0
        self.failcount = 0
        self.warncount = 0
        self.teststatus = []
        self.completed_tests = []
        self.justtest = justtest
        self.valgrind = valgrind
        self.no_pbs = no_pbs

        if file == "":
            print "Test criteria:"
            print "-" * 80
            print "length: ", length
            print "parallel: ", parallel
            print "tags to include: ", tags
            print "tags to exclude: ", exclude_tags
            print "-" * 80
            print

        # step 1. form a list of all the xml files to be considered.

        xml_files = []
        rootdir = os.path.abspath(
            os.path.join(os.path.dirname(sys.argv[0]), os.pardir))
        dirnames = []
        testpaths = ["examples", "tests", "longtests"]
        for directory in testpaths:
            if os.path.exists(os.path.join(rootdir, directory)):
                dirnames.append(directory)
        testdirs = [os.path.join(rootdir, x) for x in dirnames]
        for directory in testdirs:
            subdirs = [os.path.join(directory, x)
                       for x in os.listdir(directory)]
            for subdir in subdirs:
                g = glob.glob1(subdir, "*.xml")
                for xml_file in g:
                    try:
                        p = etree.parse(os.path.join(subdir, xml_file))
                        x = p.getroot()
                        if x.tag == "testproblem":
                            xml_files.append(os.path.join(subdir, xml_file))
                    except xml.parsers.expat.ExpatError:
                        print "Warning: %s mal-formed" % xml_file
                        traceback.print_exc()

        # step 2. if the user has specified a particular file, let's use that.

        if file != "":
            files = [file]
        elif from_file:
            try:
                f = open(from_file, 'r')
                files = [line[:-1] for line in f.readlines()]
            except IOError as e:
                sys.stderr.write(
                    "Unable to read tests from file %s: %s" % (from_file, e))
                sys.exit(1)
            f.close()
        else:
            files = None

        if files:
            for (subdir, xml_file) in [os.path.split(x) for x in xml_files]:
                if xml_file == file:
                    p = etree.parse(os.path.join(subdir, xml_file))
                    prob_defn = p.findall("problem_definition")[0]
                    prob_nprocs = int(prob_defn.attrib["nprocs"])
                    testprob = regressiontest.TestProblem(
                        filename=os.path.join(subdir, xml_file),
                        verbose=self.verbose, replace=self.modify_command_line(
                            prob_nprocs),
                        backend=backend, no_pbs=no_pbs)
                    self.tests.append((subdir, testprob))
                    files.remove(xml_file)
            if files != []:
                print "Could not find the following specified test files:"
                for f in files:
                    print f
                sys.exit(1)
            return

        # step 3. form a cut-down list of the xml files matching the correct
        # length and the correct parallelism.
        working_set = []
        for xml_file in xml_files:
            p = etree.parse(xml_file)
            prob_defn = p.findall("problem_definition")[0]
            prob_length = prob_defn.attrib["length"]
            prob_nprocs = int(prob_defn.attrib["nprocs"])
            if prob_length == length or (length == "any" and prob_length not in ["special", "long"]):
                if self.parallel is True:
                    if prob_nprocs > 1:
                        working_set.append(xml_file)
                else:
                    if prob_nprocs == 1:
                        working_set.append(xml_file)

        def get_xml_file_tags(xml_file):
            p = etree.parse(xml_file)
            p_tags = p.findall("tags")
            if len(p_tags) > 0 and not p_tags[0].text is None:
                xml_tags = p_tags[0].text.split()
            else:
                xml_tags = []

            return xml_tags

        # step 4. if there are any excluded tags, let's exclude tests that have
        # them
        if exclude_tags is not None:
            to_remove = []
            for xml_file in working_set:
                p_tags = get_xml_file_tags(xml_file)
                include = True
                for tag in exclude_tags:
                    if tag in p_tags:
                        include = False
                        break
                if not include:
                    to_remove.append(xml_file)
            for xml_file in to_remove:
                working_set.remove(xml_file)

        # step 5. if there are any tags, let's use them
        if tags is not None:
            tagged_set = []
            for xml_file in working_set:
                p_tags = get_xml_file_tags(xml_file)

                include = True
                for tag in tags:
                    if tag not in p_tags:
                        include = False

                if include is True:
                    tagged_set.append(xml_file)
        else:
            tagged_set = working_set

        for (subdir, xml_file) in [os.path.split(x) for x in tagged_set]:
            # need to grab nprocs here to pass through to modify_command_line
            p = etree.parse(os.path.join(subdir, xml_file))
            prob_defn = p.findall("problem_definition")[0]
            prob_nprocs = int(prob_defn.attrib["nprocs"])
            testprob = regressiontest.TestProblem(
                filename=os.path.join(subdir, xml_file),
                verbose=self.verbose, replace=self.modify_command_line(
                    prob_nprocs),
                backend=backend)
            self.tests.append((subdir, testprob))

        if len(self.tests) == 0:
            print "Warning: no matching tests."

    def length_matches(self, filelength):
        if self.length == filelength:
            return True
        if self.length == "medium" and filelength == "short":
            return True
        return False

    def decide_fluidity_command(self):
        bindir = os.environ["PATH"].split(':')[0]

        for binaryBase in ["dfluidity", "fluidity"]:
            binary = binaryBase
            debugBinary = binaryBase + "-debug"
            try:
                fluidity_mtime = os.stat(os.path.join(bindir, binary))[-2]
                have_fluidity = True
            except OSError:
                fluidity_mtime = 1e30
                have_fluidity = False

            try:
                debug_mtime = os.stat(os.path.join(bindir, debugBinary))[-2]
                have_debug = True
            except OSError:
                debug_mtime = 1e30
                have_debug = False

            if have_fluidity is True or have_debug is True:
                if have_fluidity is False and have_debug is True:
                    flucmd = debugBinary

                elif have_fluidity is True and have_debug is False:
                    flucmd = binary

                elif fluidity_mtime > debug_mtime:
                    flucmd = binary
                else:
                    flucmd = debugBinary

                # No longer valid since debugging doesn't change the name - any
                # suitable alternative tests?
                # if self.valgrind is True:
                #  if flucmd != debugBinary:
                #     print "Error: you really should compile with debugging for use with valgrind!"
                #     sys.exit(1)

                return flucmd

        return None

    def modify_command_line(self, nprocs):
        flucmd = self.decide_fluidity_command()
        print flucmd

        def f(s):
            if not flucmd in [None, "fluidity"]:
                s = s.replace('fluidity ', flucmd + ' ')

            if self.valgrind:
                s = "valgrind --tool=memcheck --leak-check=full -v" + \
                    " --show-reachable=yes --num-callers=8 " + \
                    "--error-limit=no --log-file=test.log " + s

            print s
            if (self.no_pbs):
                # check for mpiexec and the correct number of cores
                if (string.find(s, 'mpiexec') == -1):
                    s = s.replace(flucmd + " ", "mpiexec " + flucmd + " ")
                    print s

                if (string.find(s, '-n') == -1):
                    s = s.replace(
                        'mpiexec ', 'mpiexec -n ' + str(nprocs) + ' ')
                    print s

            return s

        return f

    def log(self, str):
        if self.verbose:
            print str

    def clean(self):
        self.log(" ")
        for t in self.tests:
            os.chdir(t[0])
            t[1].clean()

        return

    def run(self):
        self.log(" ")
        if not self.justtest:
            threadlist = []
            self.threadtests = regressiontest.ThreadIterator(self.tests)
            for i in range(options.thread_count):
                threadlist.append(threading.Thread(target=self.threadrun))
                threadlist[-1].start()
            for t in threadlist:
                '''Wait until all threads finish'''
                t.join()

            count = len(self.tests)
            while True:
                for t in self.tests:
                    if t is None:
                        continue
                    test = t[1]
                    os.chdir(t[0])
                    if test.is_finished():
                        if test.length == "long":
                            test.fl_logs(nLogLines=20)
                        else:
                            test.fl_logs(nLogLines=0)
                        try:
                            self.teststatus += test.test()
                        except:
                            self.log(
                                "Error: %s raised an exception while testing:" % test.filename)
                            lines = traceback.format_exception(
                                sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
                            for line in lines:
                                self.log(line)
                            self.teststatus += ['F']
                            test.pass_status = ['F']
                        self.completed_tests += [test]
                        t = None
                        count -= 1

                if count == 0:
                    break
                time.sleep(60)
        else:
            for t in self.tests:
                test = t[1]
                os.chdir(t[0])
                if self.length == "long":
                    test.fl_logs(nLogLines=20)
                else:
                    test.fl_logs(nLogLines=0)
                self.teststatus += test.test()
                self.completed_tests += [test]

        self.passcount = self.teststatus.count('P')
        self.failcount = self.teststatus.count('F')
        self.warncount = self.teststatus.count('W')

        if self.failcount + self.warncount > 0:
            print
            print "Summary of test problems with failures or warnings:"
            for t in self.completed_tests:
                if t.pass_status.count('F') + t.warn_status.count('W') > 0:
                    print t.filename + ':', ''.join(t.pass_status + t.warn_status)
            print

        if self.passcount + self.failcount + self.warncount > 0:
            print "Passes:   %d" % self.passcount
            print "Failures: %d" % self.failcount
            print "Warnings: %d" % self.warncount

        if self.failcount > 0:
            print "Exiting with error since at least one failure..."
            sys.exit(1)

    def threadrun(self):
        '''This is the portion of the loop which actually runs the
        tests. This is split out so that it can be threaded'''

        for (dir, test) in self.threadtests:
            try:
                runtime = test.run(dir)
                if self.length == "short" and runtime > 30.0:
                    self.log("Warning: short test ran for %f seconds which" +
                             " is longer than the permitted 30s run time" % runtime)
                    self.teststatus += ['W']
                    test.pass_status = ['W']

            except:
                self.log("Error: %s raised an exception while running:" %
                         test.filename)
                lines = traceback.format_exception(
                    sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])
                for line in lines:
                    self.log(line)
                self.tests.remove((dir, test))
                self.teststatus += ['F']
                test.pass_status = ['F']
                self.completed_tests += [test]

    def list(self):
        for (subdir, test) in self.tests:
            print os.path.join(subdir, test.filename)


if __name__ == "__main__":
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-l", "--length", dest="length",
                      help="length of problem (default=short)", default="any")
    parser.add_option("-p", "--parallelism", dest="parallel", default="serial",
                      help="parallelism of problem (default=serial)")
    parser.add_option("-e", "--exclude-tags", dest="exclude_tags",
                      default=[], action="append",
                      help="run only tests that do not have specific tags (takes precidence over -t)")
    parser.add_option("-t", "--tags", dest="tags",
                      help="run tests with specific tags", default=[], action="append")
    parser.add_option("-f", "--file", dest="file",
                      help="specific test case to run (by filename)", default="")
    parser.add_option("--from-file", dest="from_file", default=None,
                      help="run tests listed in FROM_FILE (one test per line)")
    parser.add_option("-n", "--threads", dest="thread_count", type="int",
                      help="number of tests to run at the same time", default=1)
    parser.add_option("-v", "--valgrind", action="store_true", dest="valgrind")
    parser.add_option("-c", "--clean", action="store_true", dest="clean",
                      default=False)
    parser.add_option("--just-test", action="store_true", dest="justtest")
    parser.add_option("--just-list", action="store_true", dest="justlist")
    parser.add_option("--backend")
    parser.add_option("--no_pbs", action="store_true", dest="no_pbs")
    (options, args) = parser.parse_args()

    if len(args) > 0:
        parser.error("Too many arguments.")

    if options.parallel == "serial":
        para = False
    elif options.parallel == "parallel":
        para = True
    else:
        parser.error("Specify either serial or parallel.")

    os.environ["PATH"] = os.path.abspath(
        os.path.join(os.path.dirname(sys.argv[0]), "..", "bin")) + ":" + os.environ["PATH"]
    try:
        os.environ["PYTHONPATH"] = os.path.abspath(
            os.path.join(os.path.dirname(sys.argv[0]), "..", "python")) + ":" + os.environ["PYTHONPATH"]
    except KeyError:
        os.putenv("PYTHONPATH", os.path.abspath(
            os.path.join(os.path.dirname(sys.argv[0]), "..", "python")))
    try:
        os.environ["LD_LIBRARY_PATH"] = os.path.abspath(
            os.path.join(os.path.dirname(sys.argv[0]), "..", "lib")) + ":" + os.environ["LD_LIBRARY_PATH"]
    except KeyError:
        os.putenv("LD_LIBRARY_PATH", os.path.abspath(
            os.path.join(os.path.dirname(sys.argv[0]), "..", "lib")))

    try:
        os.mkdir(os.environ["HOME"] + os.sep + "lock")
    except OSError:
        pass

    if len(options.exclude_tags) == 0:
        exclude_tags = None
    else:
        exclude_tags = options.exclude_tags

    if len(options.tags) == 0:
        tags = None
    else:
        tags = options.tags

    testharness = TestHarness(length=options.length, parallel=para,
                              exclude_tags=exclude_tags, tags=tags,
                              file=options.file, verbose=True,
                              justtest=options.justtest,
                              valgrind=options.valgrind,
                              from_file=options.from_file,
                              backend=options.backend,
                              no_pbs=options.no_pbs)

    if options.justlist:
        testharness.list()
    elif options.clean:
        testharness.clean()
    else:
        print "-" * 80
        which = os.popen("which %s" %
                         testharness.decide_fluidity_command()).read()
        if len(which) > 0:
            print "which %s: %s" % ("fluidity", which),
        versio = os.popen("%s -V" %
                          testharness.decide_fluidity_command()).read()
        if len(versio) > 0:
            print versio
        print "-" * 80

        if options.valgrind is True:
            print "-" * 80
            print "I see you are using valgrind!"
            print "A couple of points to remember."
            print "a) The log file will be produced in the directory containing the tests."
            print "b) Valgrind typically takes O(100) times as long. I hope your test is short."
            print "-" * 80

        testharness.run()
