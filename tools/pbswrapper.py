#!/usr/bin/env python

import os
import time
import sys


def is_job_running(jobid):
    outp = os.popen("qstat -an | grep %s" % jobid)
    if outp.read() == '':
        return False
    else:
        return True

pbsfile = sys.argv[1]

outp = os.popen("qsub %s" % pbsfile)
jobid = outp.read().split(".")[0]
outp.close()

while is_job_running(jobid):
    time.sleep(15)

file = open("compile.log")
data = file.read()
for line in data.split('\n'):
    print line + '\n',

if "BUILD COMPLETE" in data:
    sys.exit(0)
else:
    sys.exit(1)
