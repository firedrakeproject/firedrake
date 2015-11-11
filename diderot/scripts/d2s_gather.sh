rm store/*.o
cd /Users/chariseechiw/diderot/fem/examples/basic_d2s_sample
make clean
make
make basic_d2s_sample_init.o
mv /Users/chariseechiw/diderot/fem/examples/basic_d2s_sample/basic_d2s_sample.o /Users/chariseechiw/fire/firedrake/diderot/store/basic_d2s_sample.o
mv /Users/chariseechiw/diderot/fem/examples/basic_d2s_sample/basic_d2s_sample_init.o /Users/chariseechiw/fire/firedrake/diderot/store/basic_d2s_sample_init.o
cd /Users/chariseechiw/diderot/fem/examples/basic_d2s_lerp
make clean
make
make basic_d2s_lerp_init.o
mv /Users/chariseechiw/diderot/fem/examples/basic_d2s_lerp/basic_d2s_lerp.o /Users/chariseechiw/fire/firedrake/diderot/store/basic_d2s_lerp.o
mv /Users/chariseechiw/diderot/fem/examples/basic_d2s_lerp/basic_d2s_lerp_init.o /Users/chariseechiw/fire/firedrake/diderot/store/basic_d2s_lerp_init.o
cd /Users/chariseechiw/fire/firedrake/
scripts/firedrake-clean