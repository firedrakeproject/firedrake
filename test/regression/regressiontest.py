#!/usr/bin/env python
import sys
import os
import copy
import random
import xml.dom.minidom
import traceback
import time
import glob
import threading


class TestProblem:

    """A test records input information as well as tests for the output."""

    def __init__(self, filename, verbose=False, replace=None, pbs=False):
        """Read a regression test from filename and record its details."""
        self.name = ""
        self.command = replace
        self.command_line = ""
        self.length = ""
        self.nprocs = 1
        self.verbose = verbose
        self.variables = []
        self.pass_tests = []
        self.warn_tests = []
        self.pass_status = []
        self.warn_status = []
        self.filename = filename.split('/')[-1]
        self.pbs = pbs
        # add dir to import path
        sys.path.insert(0, os.path.dirname(filename))

        dom = xml.dom.minidom.parse(filename)

        probtag = dom.getElementsByTagName("testproblem")[0]

        for child in probtag.childNodes:
            try:
                tag = child.tagName
            except AttributeError:
                continue

            if tag == "name":
                self.name = child.childNodes[0].nodeValue
            elif tag == "problem_definition":
                self.length = child.getAttribute("length")
                self.nprocs = int(child.getAttribute("nprocs"))
                cmd = child.getElementsByTagName("command_line")[0]
                if cmd.hasChildNodes():
                    xmlcmd = cmd.childNodes[0].nodeValue
                    if self.command is not None:
                        self.command_line = self.command(xmlcmd)
            elif tag == "variables":
                for var in child.childNodes:
                    try:
                        self.variables.append(
                            Variable(name=var.getAttribute("name"),
                                     language=var.getAttribute("language"),
                                     code=var.childNodes[0].nodeValue.strip()))
                    except AttributeError:
                        continue
            elif tag == "pass_tests":
                for test in child.childNodes:
                    try:
                        self.pass_tests.append(
                            Test(name=test.getAttribute("name"),
                                 language=test.getAttribute("language"),
                                 code=test.childNodes[0].nodeValue.strip()))
                    except AttributeError:
                        continue
            elif tag == "warn_tests":
                for test in child.childNodes:
                    try:
                        self.warn_tests.append(
                            Test(name=test.getAttribute("name"),
                                 language=test.getAttribute("language"),
                                 code=test.childNodes[0].nodeValue.strip()))
                    except AttributeError:
                        continue

        self.random_string()

    def log(self, str):
        if self.verbose:
            print self.filename[:-4] + ": " + str

    def random_string(self):
        letters = "abcdefghijklmnopqrstuvwxyz"
        letters += letters.upper() + "0123456789"

        str = self.filename[:-4]
        for i in range(10):
            str += random.choice(letters)

        self.random = str

    def call_genpbs(self, dir):
        cmd = 'genpbs "%s" "%s" "%s" "%s"' % (self.filename[:-4],
                                              self.command_line,
                                              self.nprocs, self.random)
        self.log("cd " + dir + "; " + cmd)
        ret = os.system("cd " + dir + "; " + cmd)

        if ret != 0:
            self.log("Calling genpbs failed.")
            raise Exception

    def is_finished(self):
        if self.pbs and self.nprocs > 1 or self.length == "long":
            file = os.environ["HOME"] + "/lock/" + self.random
            try:
                os.remove(file)
                return True
            except OSError:
                return False
        else:
            return True

    def clean(self):
        self.log("Cleaning")

        try:
            os.stat("Makefile")
            self.log("Calling 'make clean':")
            ret = os.system("make clean")
            if not ret == 0:
                self.log("No clean target")
        except OSError:
            self.log("No Makefile, not calling make")

    def run(self, dir):
        self.log("Running")

        run_time = 0.0

        try:
            os.stat(dir + "/Makefile")
            self.log("Calling 'make input':")
            ret = os.system("cd " + dir + "; make input")
            assert ret == 0
        except OSError:
            self.log("No Makefile, not calling make")

        if (self.pbs) and self.nprocs > 1 or self.length == "long":
            ret = self.call_genpbs(dir)
            self.log("cd " + dir + "; qsub " + self.filename[
                     :-4] + ".pbs: " + self.command_line)
            os.system("cd " + dir + "; qsub " + self.filename[:-4] + ".pbs")
        else:
            self.log(self.command_line)
            start_time = time.clock()
            os.system("cd " + dir + "; " + self.command_line)
            run_time = time.clock() - start_time

        return run_time

    def fl_logs(self, nLogLines=None):
        logs = glob.glob("fluidity.log*")
        errLogs = glob.glob("fluidity.err*")

        if nLogLines is None or nLogLines > 0:
            for filename in logs:
                log = open(filename, "r").read().split("\n")
                if not nLogLines is None:
                    log = log[-nLogLines:]
                self.log("Log: " + filename)
                for line in log:
                    self.log(line)

        for filename in errLogs:
            self.log("Log: " + filename)
            log = open(filename, "r").read().split("\n")
            for line in log:
                self.log(line)

        return

    def test(self):
        def Trim(string):
            if len(string) > 4096:
                return string[:4096] + " ..."
            else:
                return string

        varsdict = {}
        self.log("Assigning variables:")
        for var in self.variables:
            tmpdict = {}
            try:
                var.run(tmpdict)
            except:
                self.log("failure.")
                self.pass_status.append('F')
                return self.pass_status

            varsdict[var.name] = tmpdict[var.name]
            self.log("Assigning %s = %s" %
                     (str(var.name), Trim(str(varsdict[var.name]))))

        if len(self.pass_tests) != 0:
            self.log("Running failure tests: ")
            for test in self.pass_tests:
                self.log("Running %s:" % test.name)
                status = test.run(varsdict)
                if status is True:
                    self.log("success.")
                    self.pass_status.append('P')
                elif status is False:
                    self.log("failure.")
                    self.pass_status.append('F')
                else:
                    self.log("failure (info == %s)." % status)
                    self.pass_status.append('F')

        if len(self.warn_tests) != 0:
            self.log("Running warning tests: ")
            for test in self.warn_tests:
                self.log("Running %s:" % test.name)
                status = test.run(varsdict)
                if status is True:
                    self.log("success.")
                    self.warn_status.append('P')
                elif status is False:
                    self.log("warning.")
                    self.warn_status.append('W')
                else:
                    self.log("warning (info == %s)." % status)
                    self.warn_status.append('W')

        self.log(''.join(self.pass_status + self.warn_status))
        return self.pass_status + self.warn_status


class TestOrVariable:

    """Tests and variables have a lot in common. This code unifies the
    commonalities."""

    def __init__(self, name, language, code):
        self.name = name
        self.language = language
        self.code = code

    def run(self, varsdict):
        func = getattr(self, "run_" + self.language)
        return func(varsdict)


class Test(TestOrVariable):

    """A test for the model output"""

    def run_bash(self, varsdict):

        varstr = ""
        for var in varsdict.keys():
            varstr = varstr + ("export %s=\"%s\"; " % (var, varsdict[var]))

        retcode = os.system(varstr + self.code)
        if retcode == 0:
            return True
        else:
            return False

    def run_python(self, varsdict):
        tmpdict = copy.copy(varsdict)
        try:
            exec self.code in tmpdict
            return True
        except AssertionError:
            # in case of an AssertionError, we assume the test has just failed
            return False
        except:
            # tell us what else went wrong:
            traceback.print_exc()
            return False


class Variable(TestOrVariable):

    """A variable definition for use in tests"""

    def run_bash(self, varsdict):
        cmd = "bash -c \"%s\"" % self.code
        fd = os.popen(cmd, "r")
        exec self.name + "=" + fd.read() in varsdict
        if self.name not in varsdict.keys():
            raise Exception

    def run_python(self, varsdict):
        try:
            exec self.code in varsdict
        except:
            print "Variable computation raised an exception"
            print "-" * 80
            for (lineno, line) in enumerate(self.code.split('\n')):
                print "%3d  %s" % (lineno + 1, line)
            print "-" * 80
            traceback.print_exc()
            print "-" * 80
            raise Exception

        if self.name not in varsdict.keys():
            print "self.name == ", self.name
            print "varsdict.keys() == ", varsdict.keys()
            print "self.name not found: does the variable define the right name?"
            raise Exception


class ThreadIterator(list):

    '''A thread-safe iterator over a list.'''

    def __init__(self, seq):
        self.list = list(seq)

        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):

        if len(self.list) == 0:
            raise StopIteration

        self.lock.acquire()
        ans = self.list.pop()
        self.lock.release()

        return ans


if __name__ == "__main__":
    prob = TestProblem(filename=sys.argv[1], verbose=True)
    prob.run()
    while not prob.is_finished():
        time.sleep(60)
    print prob.test()
