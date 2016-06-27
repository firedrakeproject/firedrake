import os
import re
import argparse


def tprint(max_fp, min_fp, ntiles, threshold):
    RED = '\033[1;37;31m%s\033[0m'
    msg = '    footprint (ntiles=%d): MAX %d KB/tile; MIN %d KB/tile' % (ntiles, max_fp, min_fp)
    if max_fp > threshold:
        print RED % msg
    else:
        print msg


def get_log_info(logdir):
    log = logdir.split('/')[-1]
    p, em, ts = re.findall(r'\d+', log)
    return int(p), int(em), int(ts)


# Parse input
p = argparse.ArgumentParser(description='Analyze log files produced by the PyOP2 tiling backend')
p.add_argument('--em', type=int, default=-1,
               help='the fusion scheme adopted (-1, the default, selects them all)')
p.add_argument('--ts', type=int, default=-1,
               help='initial average tile size (-1, the default, selects them all)')
p.add_argument('--p', type=int, default=-1,
               help='the method\'s order in space (-1, the default, selects them all')
p.add_argument('--thr', type=int, default=1800,
               help='the memory footprint threshold in KB (e.g., a fraction of L3)')
args = p.parse_args()

explicit_mode = args.em
tile_size = args.ts
poly_order = args.p
threshold = args.thr

# Log directory
logdirname = 'all-logs'
if "FIREDRAKE_DIR" in os.environ:
    parent_logdir = os.path.join(os.environ["FIREDRAKE_DIR"], 'demos', 'tiling', logdirname)
else:
    parent_logdir = logdirname

# Get requested log info
requested_logs = []
for logdir, childrendirs, _ in os.walk(parent_logdir):
    if childrendirs:
        continue
    p, em, ts = get_log_info(logdir)

    if poly_order in [-1, p] and explicit_mode in [-1, em] and tile_size in [-1, ts]:
        requested_logs.append(logdir)

# Navigate the requested logs, and produce a suitable output
last_p = -1
for logdir in requested_logs:
    p, em, ts = get_log_info(logdir)
    filenames = list(os.walk(logdir))[0][2]
    filenames = [i for i in filenames if i != 'summary.txt']
    filenames = sorted(filenames, key=lambda i: int(re.findall(r'\d+', i)[1]))

    if p != last_p:
        print "-"*49
        last_p = p

    GREEN = '\033[1;37;32m%s\033[0m'
    print GREEN % "Poly order: %d, Fusion mode: %d, Seed tile size: %d" % (p, em, ts)
    for i, loopchainlog in enumerate(filenames):
        with open(os.path.join(logdir, loopchainlog), 'r') as f:
            ntiles, ts_wc, ts_bc = [l for l in f.readlines() if 'Estimated' in l or 'KB/tile:' in l]
            ntiles = int(re.findall(r'\d+', ntiles)[0])
            max_footprint = int(re.findall(r'\d+', ts_wc)[0])
            min_footprint = int(re.findall(r'\d+', ts_bc)[0])
            tprint(max_footprint, min_footprint, ntiles, threshold)
print "-"*49
