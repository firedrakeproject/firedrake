python generate.py
make clean
make 
#cd ${HOME}/Documents/diderot-git/examples/simple_d2s
#make clean
#make 
#cd /Users/chariseechiw/firedrake/vedat
python simple2.py
unu quantize -b 8 -i x.nrrd -o tmpsine.png
eog tmpsine.png
