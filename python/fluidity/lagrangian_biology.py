import os
import sys


def read_chl_h42(filename):
    """This function reads the chlorophyll concentration from file in
    Hydrolight standard format (10 header lines, followed by depth/chl pairs).
    """

    try:
        os.stat(filename)
    except:
        print "No such file: " + str(filename)
        sys.exit(1)
    f = open(filename, 'r')
    for i in range(0, 10):
        # skip header lines
        line = f.readline()

    depths = []
    chl = []
    while line:
        line = f.readline().strip()
        if line == "":
            break
        depth = line.split()[0]
        if depth < 0.0:
            break
        depths.append(float(depth))
        chl.append(float(line.split()[1]))

    f.close()
    return (depths, chl)


def derive_PAR_irradiance(state):
    """Solar irradiance in PAR is derived from 36 individual wavebands as
    modelled by Hyperlight. The spectral bands are in Wm-2nm-1, and the PAR
    total is in mumol phot m-2s-1."""

    planck = 6.626E-34
    speed = 2.998E8
    fnmtom = 1.0E-9
    fmoltoumol = 1.0E6
    avagadro = 6.023E23
    factor = fnmtom * fmoltoumol / (planck * speed * avagadro)

    par_irrad = state.scalar_fields['IrradiancePAR']
    for n in range(par_irrad.node_count):
        irrad_sum = 0.0
        for l in range(0, 35):
            wavelength = 350 + l * 10
            field_name = 'Irradiance_' + str(wavelength)
            spectral_irrad = state.scalar_fields[field_name]
            irrad_sum = irrad_sum + \
                spectral_irrad.node_val(n) * factor * wavelength * 10.0

        par_irrad.set(n, irrad_sum)
    return 0.0
