py.test -v tests/regression/test_visualise.py
unu quantize -b 8 -i x.nrrd -o tmpsine.png
eog tmpsine.png
