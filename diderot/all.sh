scripts/firedrake-clean
sh diderot/gather.sh
scripts/firedrake-clean
rm *.nrrd
py.test -v tests/regression/test_vis_diderot.py
sh diderot/getImg.sh