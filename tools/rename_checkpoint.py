#!/usr/bin/env python

import argparse
import glob
import shutil
import os


def main():

    parser = argparse.ArgumentParser(
        prog="rename_checkpoint",
        description="""This takes a list of vtu files in the working directory
        produced from a serial checkpointed flml file with names
        base_filename_checkpoint_i.vtu for all i and renames them as
        base_filename_index+i.vtu.

        Can additionally take a list of vtu and pvtu files in the current
        directory produced from a checkpointed parallel flml file with names
        base_filename_checkpoint_i_j.vtu and base_filename_checkpoint_i.pvtu
        for all i (index) and j (processor number) and renames them as
        base_filename_index+i_j.vtu and base_filename_index+i.pvtu.

        WARNING: This may overwrite files if the backup filenames being written
        to exist already!""")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print filenames being moved",
        action="store_true",
        dest="verbose",
        default=False
    )
    parser.add_argument(
        "-s",
        "--stat",
        help="""Also process the stat files associated with the basename output
        and basename_checkpoint to produce a single stat file""",
        action="store_true",
        dest="stat",
        default=False
    )
    parser.add_argument(
        'basename',
        metavar='basename',
        help="Basename of output (without .pvtu or .vtu)",
    )
    parser.add_argument(
        'final_index',
        type=int,
        metavar='final_index',
        help="""Final index of the non-checkpoint run. checkpoint_0 will then
        match this index""",
    )

    args = parser.parse_args()
    verbose = args.verbose
    basename = args.basename
    index = args.final_index
    process_stat = args.stat

    if basename[-4:] == ".vtu":
        base_filename = os.path.basename(basename[:-4])
    elif basename[-5:] == ".pvtu":
        base_filename = os.path.basename(basename[:-5])
    else:
        base_filename = os.path.basename(basename)

    filelist = glob.glob(base_filename + "_checkpoint_*[0-9].vtu") + glob.glob(
        base_filename + "_checkpoint_*[0-9]/" + base_filename + "_checkpoint_*[0-9].vtu") + glob.glob(base_filename + "_checkpoint_*[0-9].pvtu")

    rename_vtu(base_filename, filelist, index, verbose=verbose)

    if (process_stat):
        concat_statfile(base_filename)


def rename_vtu(base_filename, filelist, index, verbose=False):
    """Rename the VTU or PVTU files based on base_filename and "checkpoint" """

    for i in range(len(filelist)):
        if filelist[i][-4:] == ".vtu":
            filesplit = filelist[i].split(".vtu")[0].split(
                base_filename + "_checkpoint_")[-1].split("_")
            # serial vtus
            if(len(filesplit) == 1):
                newindex = index + int(filesplit[0])
                newfilename = base_filename + "_" + str(newindex) + ".vtu"
                if(os.path.exists(newfilename)):
                    if(verbose):
                        print "backing up", newfilename, "to", newfilename + ".bak"
                    shutil.move(newfilename, newfilename + ".bak")
                if(verbose):
                    print "moving", filelist[i], "to", newfilename
                shutil.move(filelist[i], newfilename)
            # parallel vtus
            elif(len(filesplit) == 2):
                newindex = index + int(filesplit[0])
                checkpoint_directory = filelist[i].split('/')
                if len(checkpoint_directory) > 1:
                    checkpoint_directory = checkpoint_directory[0] + '/'
                else:
                    checkpoint_directory = './'
                newfilename = checkpoint_directory + base_filename + \
                    "_" + str(newindex) + "_" + filesplit[1] + ".vtu"
                if(os.path.exists(newfilename)):
                    if(verbose):
                        print "backing up", newfilename, "to", newfilename + ".bak"
                    shutil.move(newfilename, newfilename + ".bak")
                if(verbose):
                    print "moving", filelist[i], "to", newfilename
                shutil.move(filelist[i], newfilename)
        elif filelist[i][-5:] == ".pvtu":
            filesplit = filelist[i].split(".pvtu")[0].split(
                base_filename + "_checkpoint_")[-1].split("_")
            # parallel pvtus
            if(len(filesplit) == 1):
                newindex = index + int(filesplit[0])
                newfilename = base_filename + "_" + str(newindex) + ".pvtu"
                # If in format where all vtus live in a directory assocaited
                # with a pvtu move that directory also set directory name so
                # when rewriting pvtu can add correct directory name
                checkpoint_directory = '.'
                if(os.path.exists(filelist[i].split('.pvtu')[0])):
                    if(os.path.exists(base_filename + "_" + str(newindex))):
                        if(verbose):
                            print "backing up", base_filename + "_" + \
                                str(newindex) + "/*.vtu", "to", base_filename \
                                + "_" + str(newindex) + "/*.vtu.bak"
                        for f in glob.glob(base_filename + "_" + str(newindex) + "/*.vtu"):
                            shutil.move(f, f + ".bak")
                        if(verbose):
                            print "moving", filelist[i].split('.pvtu')[0] + \
                                "/*.vtu", "to", base_filename + "_" + \
                                str(newindex) + "/*.vtu"
                        for f in glob.glob(filelist[i].split('.pvtu')[0] + "/*.vtu"):
                            shutil.move(
                                f, base_filename + "_" + str(newindex) + '/' + f.split('/')[1])
                    else:
                        os.mkdir(base_filename + "_" + str(newindex))
                        if(verbose):
                            print "moving", filelist[i].split('.pvtu')[0] + \
                                "/*.vtu", "to", base_filename + "_" + \
                                str(newindex) + "/*.vtu"
                        for f in glob.glob(filelist[i].split('.pvtu')[0] + "/*.vtu"):
                            shutil.move(
                                f, base_filename + "_" + str(newindex) + '/' + f.split('/')[1])
                    try:
                        if(verbose):
                            print "removing directory", filelist[i].split('.pvtu')[0]
                        os.rmdir(filelist[i].split('.pvtu')[0])
                    except OSError:
                        if(verbose):
                            print filelist[i].split('.pvtu')[0], "not removed, directory may not be empty"
                    checkpoint_directory = base_filename + "_" + str(newindex)
                if(os.path.exists(newfilename)):
                    if(verbose):
                        print "backing up", newfilename, "to", newfilename + ".bak"
                    shutil.move(newfilename, newfilename + ".bak")
                if(verbose):
                    print "moving", filelist[i], "to", newfilename
                shutil.move(filelist[i], newfilename)
                # must also adjust content of pvtu so it points at moved
                # parallel vtus
                pvtufile = file(newfilename, 'r')
                lines = pvtufile.readlines()
                pvtufile.close()
                for line in range(len(lines)):
                    lineindex = lines[line].find(
                        base_filename + "_checkpoint_" + filesplit[0])
                    if(lineindex != -1):
                        processorsplit = lines[line][lineindex:].split(".vtu")[
                            0].split(base_filename + "_checkpoint_")[-1].split("_")
                        if(len(processorsplit) == 2):
                            newline = lines[line][:lineindex] + checkpoint_directory + '/' + base_filename + "_" + str(
                                newindex) + "_" + str(processorsplit[-1]) + ".vtu" + lines[line][lineindex:].split(".vtu")[-1]
                            lines[line] = newline
                pvtufile = file(newfilename, 'w')
                pvtufile.writelines(lines)
                pvtufile.close()


def concat_statfile(base_filename):

    statfile_check = base_filename + "_checkpoint.stat"
    statfile = base_filename + ".stat"

    # open the basename stat in read mode first
    sf = open(statfile, 'r')
    # open the checkpointed statfile in read mode
    sfc = open(statfile_check, 'r')

    # find the last time in the basename stat
    for line in sf:
        line = str.strip(line)
        time = line.split(" ")[0]

    sf.close()
    sf = open(statfile, 'a')
    # for each line the checkpointed statfile, if it contians data, append it
    # to the basename stat file
    for line in sfc:
        if (not line.startswith('<')):
            line = str.strip(line)
            e_time = line.split(" ")[0]
            if (e_time > time):
                sf.write(line + "\n")

    # mv the checkpointed stat file to basename_checkpoint.stat.old
    shutil.move(statfile_check, statfile_check + ".bak")


if __name__ == "__main__":
    main()
