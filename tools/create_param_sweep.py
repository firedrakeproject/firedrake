#!/usr/bin/env python

# Generate a number of FLML files from a base directory
# manipulating various parameters to explore parameter
# space

# Ouput structure is:
# output_dir/
#   template (copied in if not already here)
#   runs/
#      1/
#        run.flml
#        other defined files
#      2
#      3
#      directory_listing.csv

import shutil
import sys
import os
import itertools
import glob
import libspud
import argparse


def main():

    parser = argparse.ArgumentParser(
        description="""This script produces the FLML and input files required
        for parameter sweeps. The user supplies a template file, output
        location and a text file that contains the option paths and parameter
        sets that are to be used."""
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help="Verbose output: mainly progress reports",
        default=False
    )
#    parser.add_argument(
#            "-m",
#            "--move-files",
#            help="""A file to move along with the flml, for example forcing
#                 NetCDF, initialisation files, etc.
#                 Files will need to be in the template directory.
#                 You do not need to move the mesh files.
#                 Add as many -m flags as requried""",
#            action="append",
#            dest="extras",
#            )

    # positional args:
    parser.add_argument(
        'template_dir',
        help="A directory containing the meshes, FLML and any associated files"
    )
    parser.add_argument(
        'output_dir',
        help="""A directory where output will be stored. Will be created if it
        doesn't exist"""
    )
    parser.add_argument(
        'param_space_file',
        help="""A text file containing a human-readable name; option path;
        comma-seperated list of values"""
    )

    args = parser.parse_args()
    output_dir = str(args.output_dir)
    template_dir = str(args.template_dir)
    param_space_file = str(args.param_space_file)
    verbose = args.verbose

    # check template dir exists
    if (not os.path.exists(template_dir)):
        print "Your template directory does not exist or you don't have \
               permissions to read it"
        sys.exit(-1)

    # check it contains an FLML
    if (len(glob.glob(os.path.join(template_dir, '*.flml'))) == 0):
        print "Your template directory does not contain an FLML file. Can't \
               do much without it."
        sys.exit(-1)
    elif (len(glob.glob(template_dir + '*.flml')) > 1):
        print "Warning: your template directory contains >1 FLML. We'll be \
               using: " + glob.glob(template_dir / +'*.flml')[0]
    # get the name of the template dir, discarding the path
    direc = template_dir.rsplit('/')[0]

    # then we have the output directory
    # We'll create a dir called "runs" and dump output in there, with a
    # directory listing file strip of any trailing /
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    # check it exists
    if (not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        if (verbose):
            print "Warning: Creating output directory: " + output_dir

    # if the template dir is not already there, copy it in
    if (not os.path.exists(os.path.join(output_dir, direc))):
        shutil.copytree(template_dir, os.path.join(output_dir, direc))
        if (verbose):
            print "Copying template directory, into output folder"

    # reset template_dir variable to point to the new one instead
    template_dir = os.path.join(output_dir, direc)

    # create "runs" directory
    if (not os.path.exists(os.path.join(output_dir, "runs"))):
        os.mkdir(os.path.join(output_dir, "runs"))
        if (verbose):
            print "Creating runs folder"

    # third arg is the param space file
    # Plain text file with the following format:
    # Name; spud_path; value_1, value_2, value3, etc
    # Name; spud_path; value_1, value_2
    # check file exists

    # read in the param space file
    if (verbose):
        print "Reading in parameter space"
    param_space, paths, names = read_param_space(param_space_file)

    # generate all the combinations
    params = gen_param_combinations(param_space)

    # make the FLMLs
    gen_flmls(params, template_dir, output_dir, paths, names, verbose)

    if (verbose):
        print "Done generating files"

    return 0


# set up parameter space
def read_param_space(param_space_file):

    f = open(param_space_file, 'r')

    param_space = []
    paths = []
    names = []
    for l in f:
        line = l.strip()
        data = line.split(';')
        name = data[0]
        path = data[1]
        values = data[2].strip().split(':')
        param_space.append(values)
        paths.append(path)
        names.append(name)

    return param_space, paths, names


def gen_param_combinations(param_space):

    # thought this was going to be difficult, but it's one line...
    return list(itertools.product(*param_space))


def gen_flmls(params, template_dir, output_dir, paths, names, verbose):

    # get flml file from template_dir - first one we come across
    # If you have more in there, tough.
    full_flml = glob.glob(os.path.join(template_dir, '*.flml'))[0]
    # strip to the filename only
    flml = os.path.basename(full_flml)

    f = open(os.path.join(output_dir, 'runs', "directory_listing.csv"), "w")
    # append data to directory listing file
    line = "Directory number"
    for n in names:
        line = line + "," + n

    line = line + "\n"
    f.write(line)

    # loop over paramas
    # create a new directory, with a unqiue number, starting from 1
    # This makes it easier to use in an array job on CX1
    dir_num = 1
    for p_set in params:
        if (verbose):
            print "Processing " + str(dir_num)

        # copy contents from template folder to directory number
        dirname = os.path.join(output_dir, 'runs', str(dir_num))
        if (os.path.exists(os.path.join(dirname))):
            shutil.rmtree(dirname)
        shutil.copytree(template_dir, dirname)

        # open FLML file
        output_file = os.path.join(dirname, flml)
        # This load the data into the memory of the libspud library
        libspud.load_options(output_file)

        i = 0
        for path in paths:
            path = path.strip()
            # get type
            path_type = libspud.get_option_type(path)
            path_rank = libspud.get_option_rank(path)
            if (path_type is float and path_rank == 0):
                libspud.set_option(path, float(p_set[i]))
            elif (path_rank == 1 and path_type is float):
                value = eval(p_set[i])
                val = list(map(float, value))
                libspud.set_option(path, val)
            elif (path_rank == 2 and path_type is float):
                value = eval(p_set[i])
                val = []
                for row in value:
                    val.append(list(map(float, row)))
                libspud.set_option(path, val)

            i = i + 1

        # save file
        libspud.write_options(output_file)

        # append data to directory listing file
        line = str(dir_num)
        for p in p_set:
            # quoting the params so csv parsers can get the columns right
            line = line + "," + '"' + str(p) + '"'
        line = line + "\n"
        f.write(line)

        dir_num += 1

    f.close()

if __name__ == "__main__":
    main()
