import os
import sys


def check_env_var(var):
    try:
        accessed = os.environ[var]
    except:
        print "Check that env var %s is correctly set. Exiting..."
        sys.exit(0)
    return accessed


pyop2_cache = check_env_var('PYOP2_CACHE_DIR')
ffc_cache = check_env_var('FIREDRAKE_FFC_KERNEL_CACHE_DIR')
tmpdir = check_env_var('TMPDIR')

with open(os.environ['PBS_NODEFILE'], 'r') as f:
    nodes = f.readlines()
    for n in nodes:
        for cache in [pyop2_cache, ffc_cache]:
            print 'scp -q -r %s %s:%s' % (cache, n.strip(), tmpdir)
            os.system('scp -q -r %s %s:%s' % (cache, n.strip(), tmpdir))
