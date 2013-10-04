#!/usr/bin/env python

# This script generates pvd files for time series in paraview
import vtktools as vtk
import glob
import re
import sys


def generate_pvd(pvdfilename, list_vtu_filenames, list_time):
    """ This function writes a simple xml pvd file that can be
        loaded in paraview to display time series correctly
        with irregular timesteps
    """
    if (len(list_vtu_filenames) != len(list_time)):
        sys.stderr.write(
            "Error, list of filenames and time are of unequal length.\n")
        raise SystemExit("Something went rather badly.")
    pvdfile = open(pvdfilename, "w")
    # Write header:
    pvdfile.write("""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
<Collection>
""")
    # Now write the information for the time series:
    for i in range(len(list_time)):
        pvdfile.write('    <DataSet timestep="' + str(list_time[i]) +
                      '" group="" part="0" file="' + list_vtu_filenames[i] + '"/>\n')
    # Write closing statements in the file and close it:
    pvdfile.write('</Collection>\n</VTKFile>')
    pvdfile.close()


# Function taken from: http://stackoverflow.com/a/2669120/396967
def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Get basename of the simulation from command line or assemble it on your own:
try:
    simulation_basename = sys.argv[1]
except:
    errmsg = "ERROR: You have to give 'genpvd' the basename of the considered \
            vtu files.\nProgram will exit...\n"
    sys.stderr.write(errmsg)
    raise SystemExit("Simulation basename is required.")

# Find all vtu/pvtu files for fluid vtus in this folder:
fluid_vtus = []
for file in sorted_nicely(glob.glob(simulation_basename + '_[0-9]*vtu')):
    if (not ('checkpoint' in file)):
        fluid_vtus.append(file)

# Give an error if the list of vtu files is empty:
if (len(fluid_vtus) == 0):
    errmsg = "ERROR: No vtu files with basename " + \
        simulation_basename + " were found.\nIs the basename correct?\n"
    sys.stderr.write(errmsg)
    raise SystemExit("No vtu files found. Exit")

# Loop over all the fluid vtus found and collect time data from them to
# assemble the pvd file:
time = []
for filename in fluid_vtus:
    sys.stdout.write("Processing file: " + filename + "\n")
    # Drastically reducing the computational effort by only opening the first
    # partition of the mesh (if it ran in parallel):
    if (filename.endswith('.pvtu')):
        simbasename = filename.replace('.pvtu', '')
        filename = simbasename + '/' + simbasename + '_0.vtu'
    # Get the unstructured mesh:
    data = vtk.vtu(filename)
    # Only process the first node in the mesh, as the time is constant over
    # the whole mesh:
    n0 = data.ugrid.GetCell(0).GetPointId(0)
    t = data.ugrid.GetPointData().GetArray("Time").GetTuple(n0)
    time.append(t[0])

# Generate fluid pvd file:
generate_pvd(simulation_basename + '.pvd', fluid_vtus, time)

sys.stdout.write("=============================================\n")
sys.stdout.write("Program exited without errors.\n")
sys.stdout.write("Open the file\n")
sys.stdout.write("  " + simulation_basename + ".pvd\n")
sys.stdout.write("in Paraview.\n")
sys.stdout.write("=============================================\n")
