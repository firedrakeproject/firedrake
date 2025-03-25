#! /usr/bin/env python3
from six import iteritems

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pprint import pformat
import logging
import json
import sys
import os
import subprocess
from collections import OrderedDict, defaultdict


def check_output(args, env=None):
    return subprocess.check_output(args, stderr=subprocess.STDOUT, env=env)


def quit(log, message):
    log.error(message)
    sys.exit(1)


def main():
    parser = ArgumentParser(description="""Provide information on the currently downloaded version of Firedrake and its configuration.
    This is particularly useful information to include when reporting bugs.""",
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--log", action='store_true',
                        help="Log the output of the script to firedrake-status.log as well as to the console.")

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-6s %(message)s',
                            filename='firedrake-status.log',
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(level=logging.WARNING,
                            format='%(message)s')
    log = logging.getLogger()

    try:
        firedrake_env = os.environ["VIRTUAL_ENV"]
    except KeyError:
        quit(log, "Unable to retrieve virtualenv name from the environment.\n Please ensure the virtualenv is active before running firedrake-update.")

    try:
        with open(os.path.join(os.environ["VIRTUAL_ENV"],
                               ".configuration.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = defaultdict(dict)

    try:
        config["system"] = check_output(["uname", "-a"])
    except subprocess.CalledProcessError:
        log.error("Failed to retrieve system information.")

    print("Firedrake Configuration:")
    if not config:
        print("No configuration information found.")
    else:
        for key, val in iteritems(config["options"]):
            print("    {}: {}".format(key, val))

        print("Additions:")
        if config["additions"]:
            for a in config["additions"]:
                print("   " + a)
        else:
            print("    None")

    for var in ["PYTHONPATH", "PETSC_ARCH", "PETSC_DIR"]:
        config["environment"][var] = os.environ.get(var, None)

    print("Environment:")
    for key, val in iteritems(config["environment"]):
        print("    {}: {}".format(key, val))

    status = OrderedDict()
    for dir in sorted(os.listdir(firedrake_env + "/src")):
        try:
            os.chdir(firedrake_env + "/src/" + dir)
        except OSError as e:
            if e.errno == 20:
                # Not a directory
                continue
            else:
                raise
        try:
            revision = check_output(["git", "rev-parse", "--short", "HEAD"]).decode('ascii').strip()
            branch = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('ascii').strip()
        except subprocess.CalledProcessError:
            log.error("Unable to retrieve git information from %s." % dir)
        else:
            try:
                _ = check_output(["git", "diff-index", "--quiet", "HEAD"])
                modified = False
            except subprocess.CalledProcessError:
                modified = True

            status[dir] = {"branch": branch,
                           "revision": revision,
                           "modified": modified}

    status_string = ""
    status_string += "Status of components:\n"
    componentformat = "|{:20}|{:30}|{:10}|{!s:10}|\n"
    header = componentformat.format("Package", "Branch", "Revision", "Modified")
    line = "-" * (len(header) - 1) + "\n"
    status_string += line + header + line
    for dir, d in iteritems(status):
        status_string += componentformat.format(dir, d["branch"], d["revision"], d["modified"])
    status_string += line

    print(status_string)
    log.info("\n" + status_string)

    log.info("Firedrake configuration: ")
    log.info(pformat(config))
    log.debug("\nPip packages installed:")
    try:
        log.debug(check_output(["pip", "freeze"]))
    except subprocess.CalledProcessError:
        log.error("""Failed to retrieve list of pip installed packages. Try running:

        pip freeze.

    """)
    log.debug("\n Full environment:")
    try:
        log.debug(check_output(["env"]))
    except subprocess.CalledProcessError:
        log.error("""Shell command env failed.""")


if __name__ == "__main__":
    main()
