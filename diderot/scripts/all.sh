#scripts/firedrake-clean
sh diderot/scripts/d2s_gather.sh
scripts/firedrake-clean
rm diderot/tmp/*.nrrd
rm diderot/tmp/*.png
py.test -v d2s.py