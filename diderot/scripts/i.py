from os.path import abspath, dirname
import pytest
import os

m=['basic_d2s_lerp']
path='/Users/chariseechiw/diderot/fem/examples/'

for i in m:
    path0=path+i
    print path0
    os.system('cd '+ path0)
    os.system('make clean')
    os.system('make')