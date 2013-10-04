#!/usr/bin/env python

import os
import os.path
import glob


class UnitTest:

    def __init__(self, exe):
        self.exe = os.curdir + os.sep + exe.split(os.sep)[-1]
        self.verbose = True
        self.dir = os.sep.join((os.curdir + os.sep + exe).split(os.sep)[:-1])
        self.cwd = os.getcwd()

    def log(self, msg):
        if self.verbose and msg != '':
            print "    %s: %s" % (self.exe.split('/')[-1], msg)

    def run(self):
        os.chdir(self.dir)
#        self.log("chdir " + self.dir)
        self.log("Running")
        f = os.popen(self.exe)
        self.output = f.read()
        os.chdir(self.cwd)

        exitStatus = f.close()
        if exitStatus is None:
            return 0
        else:
            return exitStatus

    def parse(self):
        passcount = 0
        warncount = 0
        failcount = 0

        for line in self.output.split('\n'):
            line = line.lstrip()
            self.log(line)
            if line.startswith("Pass"):
                passcount = passcount + 1
            if line.startswith("Warn"):
                warncount = warncount + 1
            if line.startswith("Fail"):
                failcount = failcount + 1

        return (passcount, warncount, failcount)


class UnitTestHarness:

    def __init__(self, dir):
        self.tests = []
        if dir[-1] == '/':
            dir = dir + "*"
        else:
            dir = dir + "/*"

        files = glob.glob(dir)
        for file in files:
            if not os.path.isdir(file):
                self.tests.append(UnitTest(file))

    def run(self):
        passcount = 0
        warncount = 0
        failcount = 0

        warntests = []
        failtests = []

        for test in self.tests:
            exitStatus = test.run()

            (P, W, F) = test.parse()

            if (P, W, F) == (0, 0, 0):
                print "    WARNING: no output from test"
                warncount += 1
                warntests.append(test.exe)

            if W > 0:
                warntests.append(test.exe)

            if F > 0:
                failtests.append(test.exe)

            if not exitStatus == 0:
                print "    ERROR: non-zero exit code from test"
                failcount += 1
                if not test.exe in failtests:
                    failtests.append(test.exe)

            passcount += P
            warncount += W
            failcount += F

        print "RESULTS"
        print "    Passes:   %d" % passcount
        if len(warntests) == 0:
            print "    Warnings: %d" % warncount
        else:
            print "    Warnings: %d; tests = %s" % (warncount, warntests)
        if len(failtests) == 0:
            print "    Failures: %d" % failcount
        else:
            print "    Failures: %d; tests = %s" % (failcount, failtests)

if __name__ == "__main__":
    import sys

    try:
        os.environ["PYTHONPATH"] = os.path.abspath(
            os.path.join(os.getcwd(), "python")) + ":" + os.environ["PYTHONPATH"]
    except KeyError:
        os.putenv("PYTHONPATH", os.path.abspath(
            os.path.join(os.getcwd(), "python")))

    try:
        os.environ["LD_LIBRARY_PATH"] = os.getcwd() + os.sep + sys.argv[
            1] + os.sep + "lib:" + os.environ["LD_LIBRARY_PATH"]
    except KeyError:
        os.putenv("LD_LIBRARY_PATH", os.getcwd()
                  + os.sep + sys.argv[1] + os.sep + "lib")

    if "--electricfence" in sys.argv:
        os.putenv("LD_PRELOAD", "/usr/lib/libefence.so.0.0")
        #os.putenv("EF_DISABLE_BANNER", "1")

    TestHarness = UnitTestHarness(sys.argv[-1])
    TestHarness.run()
