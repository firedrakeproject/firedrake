#scripts/firedrake-clean
sh diderot/gather.sh
scripts/firedrake-clean
rm tmp/*.nrrd
rm tmp/*.png
py.test -v tests/regression/test_vis_diderot.py