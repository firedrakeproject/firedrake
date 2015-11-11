rm diderot/*.o
cd /Users/chariseechiw/diderot/fem/examples/d2s/simple_lerp
make clean
make
make simple_lerp_init.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/simple_lerp/simple_lerp.o /Users/chariseechiw/fire/firedrake/diderot/simple_lerp.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/simple_lerp/simple_lerp_init.o /Users/chariseechiw/fire/firedrake/diderot/simple_lerp_init.o
cd /Users/chariseechiw/diderot/fem/examples/d2s/fox
make clean
make
make fox_init.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/fox/fox.o /Users/chariseechiw/fire/firedrake/diderot/fox.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/fox/fox_init.o /Users/chariseechiw/fire/firedrake/diderot/fox_init.o
cd /Users/chariseechiw/diderot/fem/examples/d3s_simple_lerp
make clean
make
make d3s_simple_lerp_init.o
mv /Users/chariseechiw/diderot/fem/examples/d3s_simple_lerp/d3s_simple_lerp.o /Users/chariseechiw/fire/firedrake/diderot/d3s_simple_lerp.o
mv /Users/chariseechiw/diderot/fem/examples/d3s_simple_lerp/d3s_simple_lerp_init.o /Users/chariseechiw/fire/firedrake/diderot/d3s_simple_lerp_init.o
cd /Users/chariseechiw/diderot/fem/examples/d2s/simple_sample
make clean
make
make simple_sample_init.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/simple_sample/simple_sample.o /Users/chariseechiw/fire/firedrake/diderot/simple_sample.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/simple_sample/simple_sample_init.o /Users/chariseechiw/fire/firedrake/diderot/simple_sample_init.o
cd /Users/chariseechiw/fire/firedrake/
cd /Users/chariseechiw/diderot/fem/examples/d2s/iso_sample
make clean
make
make iso_sample_init.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/iso_sample/iso_sample.o /Users/chariseechiw/fire/firedrake/diderot/iso_sample.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/iso_sample/iso_sample_init.o /Users/chariseechiw/fire/firedrake/diderot/iso_sample_init.o
cd /Users/chariseechiw/fire/firedrake/
make iso_lerp_init.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/iso_lerp/iso_lerp.o /Users/chariseechiw/fire/firedrake/diderot/iso_lerp.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/iso_lerp/iso_lerp_init.o /Users/chariseechiw/fire/firedrake/diderot/iso_lerp_init.o
cd /Users/chariseechiw/diderot/fem/examples/d2s/simple_lerp_color
make clean
make
make simple_lerp_color_init.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/simple_lerp_color/simple_lerp_color.o /Users/chariseechiw/fire/firedrake/diderot/simple_lerp_color.o
mv /Users/chariseechiw/diderot/fem/examples/d2s/simple_lerp_color/simple_lerp_color_init.o /Users/chariseechiw/fire/firedrake/diderot/simple_lerp_color_init.o
cd /Users/chariseechiw/diderot/fem/examples/d3s/mip
make clean
make
make mip_init.o
mv /Users/chariseechiw/diderot/fem/examples/d3s/mip/mip.o /Users/chariseechiw/fire/firedrake/diderot/mip.o
mv /Users/chariseechiw/diderot/fem/examples/d3s/mip/mip_init.o /Users/chariseechiw/fire/firedrake/diderot/mip_init.o
cd /Users/chariseechiw/diderot/fem/examples/d3s/simple
make clean
make
make simple_init.o
mv /Users/chariseechiw/diderot/fem/examples/d3s/simple/simple.o /Users/chariseechiw/fire/firedrake/diderot/simple.o
mv /Users/chariseechiw/diderot/fem/examples/d3s/simple/simple_init.o /Users/chariseechiw/fire/firedrake/diderot/simple_init.o
cd /Users/chariseechiw/diderot/fem/examples/d3s/simple_lerpd3
make clean
make
make simple_lerpd3_init.o
mv /Users/chariseechiw/diderot/fem/examples/d3s/simple_lerpd3/simple_lerpd3.o /Users/chariseechiw/fire/firedrake/diderot/simple_lerpd3.o
mv /Users/chariseechiw/diderot/fem/examples/d3s/simple_lerpd3/simple_lerpd3_init.o /Users/chariseechiw/fire/firedrake/diderot/simple_lerpd3_init.o
cd /Users/chariseechiw/fire/firedrake/
scripts/firedrake-clean
