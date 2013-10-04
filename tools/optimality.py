#!/usr/bin/python
#    Copyright (C) 2006 Imperial College London and others.
#
#    Please see the AUTHORS file in the main source directory for a full list
#    of copyright holders.
#
#    Prof. C Pain
#    Applied Modelling and Computation Group
#    Department of Earth Science and Engineering
#    Imperial College London
#
#    amcgsoftware@imperial.ac.uk
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation,
#    version 2.1 of the License.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
#    USA

import os.path
import numpy
import argparse
import shlex
from subprocess import Popen, PIPE
import scipy.optimize
import string
import libspud
from fluidity_tools import stat_parser
from fluidity_tools import stat_creator
import time
import pickle
import glob
import math
import shutil

verbose = False
debug = False

# Hack for libspud to be able to read an option from a different files.
# A better solution would be to fix libspud or use an alternative
# implementation like
# https://github.com/gmarkall/manycore_form_compiler/blob/master/mcfc/optionfile.py


def superspud(filename, cmd):
    libspud.load_options(filename)
    r = None
    if hasattr(cmd, '__iter__'):
        for c in cmd:
            exec "try: r = " + c + "\nexcept libspud.SpudNewKeyWarning: pass"
    else:
        exec "try: r = " + cmd + "\nexcept libspud.SpudNewKeyWarning: pass"
    libspud.clear_options()
    return r

# Executes the model specified in the optimality option tree
# The model stdout is printed to stdout.


def run_model(m, opt_options, model_options):
    update_custom_controls(m, opt_options)
    if (superspud(model_options, "libspud.have_option('/adjoint/controls/load_controls')")):
        # If the model is loading the default controls, we need to make suer
        # the control files are up to date:
        update_default_controls(m, opt_options, model_options)
    command_line = superspud(
        opt_options, "libspud.get_option('/model/command_line')")
    option_file = superspud(
        opt_options, "libspud.get_option('/model/option_file')")
    args = shlex.split(command_line)
    args.append(option_file)
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    out = string.join(p.stdout.readlines())
    outerr = string.join(p.stderr.readlines())
    if p.wait() != 0:
        print "Model execution failed."
        print "The error was:"
        print outerr
        exit()
    if verbose:
        print "Model output: "
        print out

# Intialises the custom controls using the supplied python code.


def get_custom_controls(opt_options):
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    m = {}
    for i in range(nb_controls):
        cname = superspud(opt_options,
                          "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(opt_options,
                          "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        # With the custom type, the user specifies python function to
        # initialise the controls.
        if ctype == 'custom':
            initial_control_code = superspud(
                opt_options, "libspud.get_option('/control_io/control[%d]/type::custom/initial_control')" % i)
            d = {}
            exec initial_control_code in d
            m[cname] = d['initial_control']()
    return m

# Initialse the default controls by reading in the control files.
# This assumes that the model has been run without the
# "/adjoint/load_controls" option (which produced the initial control
# files).


def read_default_controls(opt_options, model_options):
    simulation_name = superspud(
        model_options, "libspud.get_option('/simulation_name')")
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    m = {}
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        if ctype == 'default':
            act_flag = False  # Check that at least one control file exists
            for ctrl_file in glob.iglob('control_' + simulation_name + '_' + cname + '_[0-9]*.pkl'):
                try:
                    timestep = int(
                        ctrl_file.strip()[len('control_' + simulation_name + '_' + cname + '_'):len(ctrl_file) - 4])
                except:
                    print """Error while reading the control files.
                    The control file %s does not conform the standard naming
                    conventions for control files.""" % ctrl_file
                    exit()
                f = open(ctrl_file, 'rb')
                m[(cname, timestep)] = pickle.load(f)
                f.close()
                act_flag = True
            if not act_flag:
                print "Warning: Found no control file for control ", cname, "."
    return m

# Initialse the default controli bounds by reading in the control bound files.
# This assumes that the model has been run without the
# "/adjoint/load_controls" option (which produced the initial control
# bound files).


def read_default_control_bounds(opt_options, model_options):
    simulation_name = superspud(
        model_options, "libspud.get_option('/simulation_name')")
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    m_bounds = {"lower_bound": {}, "upper_bound": {}}
    # Loop over controls
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        if ctype != 'default':
            continue
        have_bound = {}
        # Loop over lower and upper bound
        for k in m_bounds.keys():
            have_bound[k] = superspud(model_options, "libspud.have_option('/adjoint/controls/control[" + str(i) + "/bounds/" + k + "')")
            if not have_bound[k]:
                continue
            # Check that at least one control bound file exists
            act_flag = False
            for ctrl_file in glob.iglob('control_' + simulation_name + '_' + cname + '_' + k + '_[0-9]*.pkl'):
                try:
                    timestep = int(
                        ctrl_file.strip()[len('control_' + simulation_name + '_' + cname + '_' + k + '_'):len(ctrl_file) - 4])
                except:
                    print """Error while reading the control bound files.
                    The control bound file %s does not conform the standard
                    naming conventions for control files.""" % ctrl_file
                    exit()
                f = open(ctrl_file, 'rb')
                m_bounds[k][(cname, timestep)] = pickle.load(f)
                f.close()
                act_flag = True
            if not act_flag:
                print "Warning: Found no control bound file for control ", cname, "."
    return m_bounds

# Completes the control bounds by adding the missing controls and filling
# them with nan's


def complete_default_control_bounds(m, m_bounds):
    bound_types = {"lower_bound": {}, "upper_bound": {}}
    for bound_type in bound_types:
        for control in m.keys():
            if control in m_bounds[bound_type]:
                continue
            # We need objects as dtype because we want to keep the Nones for
            # later
            m_bounds[bound_type][control] = numpy.empty(
                shape=m[control].shape, dtype=object)
            m_bounds[bound_type][control].fill(None)
    return m_bounds


# Returns the control derivatives for both the custom and the default controls.
def read_control_derivatives(opt_options, model_options):
    simulation_name = superspud(
        model_options, "libspud.get_option('/simulation_name')")
    functional_name = superspud(
        opt_options, "libspud.get_option('/functional/name')")
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    derivs = {}
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        if ctype == 'default':
            act_flag = False  # Check that at least one control file exists
            for ctrl_file in glob.iglob('control_' + simulation_name + '_adjoint_' + functional_name + '_' + cname + '_TotalDerivative_[0-9]*.pkl'):
                try:
                    # The naming convenction is
                    # control+simulation_name+control_name+TotalDerivative, but
                    # do not forget that the derivatives where produced during
                    # the adjoint run in which the simulation name is
                    # simulation_name+functional_name
                    timestep = int(
                        ctrl_file.strip()[len('control_' + simulation_name + '_adjoint_' + functional_name + '_' + cname + '_TotalDerivative_'):len(ctrl_file) - 4])
                except:
                    print """Error while reading the control derivative files.
                    The control file %s does not conform the standard naming
                    conventions for control files.""" % ctrl_file
                    exit()
                f = open(ctrl_file, 'rb')
                derivs[(cname, timestep)] = pickle.load(f)
                f.close()
                act_flag = True
            if not act_flag:
                print "Warning: Found no control derivative file for control ", cname, "."
        elif ctype == 'custom':
            control_derivative_code = superspud(
                opt_options, "libspud.get_option('/control_io/control[%d]/type::custom/control_derivative')" % i)
            d = {}
            exec control_derivative_code in d
            derivs[cname] = d['control_derivative']()
        else:
            print "Unknown control type " + ctype + "."
            exit()
    return derivs

# Writes the custom controls onto disk


def update_custom_controls(m, opt_options):
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        # With the custom type, the user specifies a python function to update
        # the controls.
        if ctype == 'custom':
            update_control_code = superspud(
                opt_options, "libspud.get_option('/control_io/control[%d]/type::custom/update_control')" % i)
            d = {}
            exec update_control_code in d
            d['update_control'](m[cname])

# Writes the default controls onto disk


def update_default_controls(m, opt_options, model_options):
    global debug
    simulation_name = superspud(
        model_options, "libspud.get_option('/simulation_name')")
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    # Loop over default controls
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        if ctype == 'default':
            # Loop over controls
            for k in m.keys():
                # Check if that is a control we are looking for
                if k[0] == cname:
                    timestep = k[1]
                    file_name = 'control_' + simulation_name + \
                        '_' + cname + '_' + str(timestep) + '.pkl'
                    if not os.path.isfile(file_name):
                        print "Error: writing control file ", file_name, " which did not exist before."
                        exit()
                    if debug:
                        # Check that the file we are writing has the same shape
                        # than the one we are writing
                        f = open(file_name, 'rb')
                        m_old = pickle.load(f)
                        if m[k].shape != m_old.shape:
                            print "Error: The shape of the control in ", file_name, " changed."
                            exit()
                        f.close()
                    f = open(file_name, 'wb')
                    pickle.dump(m[k], f)
                    f.close()

# Check the consistency of model and option file


def check_option_consistency(opt_options, model_options):
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        ctype = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/type/name')" % i)
        # Check that the default controls exist in the model
        # and that custom controls not.
        if ctype == 'custom':
            if superspud(model_options, "libspud.have_option('/adjoint/controls/control::" + cname + "')"):
                print "The custom control " + cname + " is a default control in the model option tree."
                exit()
        elif ctype == 'default':
            if not superspud(model_options, "libspud.have_option('/adjoint/controls/control::" + cname + "')"):
                print "The default control " + cname + " was not found in the model option tree."
                exit()
        else:
            print "Unknown control type " + ctype + "."
            exit()

# Check that the the controls in dJdm are consistent with the ones in m
# If m_bounds is present, it also checks the consistency of the bounds


def check_control_consistency(m, djdm, m_bounds=None):
    djdm_keys = djdm.keys()
    m_keys = m.keys()
    djdm_keys.sort()
    m_keys.sort()
    if m_keys != djdm_keys:
        print "Error: The controls are not consistent with the controls derivatives."
        print "The controls are:", m_keys
        print "The control derivatives are:", djdm_keys
        print "Check the consistency of the control definition in the model and the optimality configuration."
        exit()
    for k, v in sorted(m.items()):
        if m[k].shape != djdm[k].shape:
            print "The control ", k, " has shape ", m[k].shape, " but dJd(", k, ") has shape ", djdm[k].shape
            exit()
    # Check the bounds
    if m_bounds is not None:
        bound_types = ("lower_bound", "upper_bound")
        for bound_type in bound_types:
            m_bounds_keys = m_bounds[bound_type].keys()
            m_bounds_keys.sort()
            if m_keys != m_bounds_keys:
                print "Error: The controls are not consistent with the control ", bound_type, "."
                print "The controls are:", m_keys
                print "The control ", bound_type, "s are:", m_bounds_keys
                exit()
            for k, v in sorted(m.items()):
                if m[k].shape != m_bounds[bound_type][k].shape:
                    print "The control ", k, " has shape ", m[k].shape, " but the ", bound_type, " has shape ", m_bounds[bound_type][k].shape
                    exit()


def delete_temporary_files(model_options):
    # remove any control files
    pkl_files = glob.glob('control_*.pkl')
    for f in pkl_files:
        os.remove(f)
    # remove any stat files from the model
    simulation_name = superspud(
        model_options, "libspud.get_option('/simulation_name')")
    stat_files = glob.glob(simulation_name + '*.stat')
    for f in stat_files:
        os.remove(f)

# Returns true if bounds are specified for one of the controls


def have_bounds(opt_options, model_options):
    nb_controls = superspud(
        opt_options, "libspud.option_count('/control_io/control')")
    have_bounds = False
    for i in range(nb_controls):
        cname = superspud(
            opt_options, "libspud.get_option('/control_io/control[%d]/name')" % i)
        have_bounds = have_bounds or superspud(
            model_options, "libspud.have_option('/adjoint/controls/control[" + cname + "]/bounds')")
    return have_bounds

#
# Optimisation loop ###################
#


def optimisation_loop(opt_options, model_options):

    # Implement a memoization function to avoid duplicated functional
    # (derivative) evaluations
    class MemoizeMutable:

        def __init__(self, fn):
            self.fn = fn
            self.memo = {}

        def __call__(self, *args, **kwds):
            import cPickle
            str = cPickle.dumps(args, 1) + cPickle.dumps(kwds, 1)
            if not str in self.memo:
                self.memo[str] = self.fn(*args, **kwds)
            return self.memo[str]

        def has_cache(self, *args, **kwds):
            import cPickle
            str = cPickle.dumps(args, 1) + cPickle.dumps(kwds, 1)
            return str in self.memo

        # Insert a function value into the cache manually.
        def __add__(self, value, *args, **kwds):
            import cPickle
            str = cPickle.dumps(args, 1) + cPickle.dumps(kwds, 1)
            self.memo[str] = value

    # Small test code for the un/serialiser
    def test_serialise():
        x = {'a': numpy.random.rand(3, 2), 'b': numpy.random.rand(
            3, 2, 4, 5), 'c': numpy.random.rand(1)}
        [m_serial, m_shape] = serialise(x)
        x_re = unserialise(m_serial, m_shape)
        return (x['a'] == x_re['a']).all() and (x['b'] == x_re['b']).all() and (x['c'] == x_re['c']).all()

    # This function takes in a dictionary m with numpy.array as entries.
    # From that it creates one serialised numpy.array with all the data.
    # In addition it creates m_shape, a dictionary which is used in
    # unserialise.
    def serialise(m):
        m_serial = numpy.array([])
        m_shape = {}
        for k, v in sorted(m.items()):
            m_serial = numpy.append(m_serial, v.flatten())
            m_shape[k] = v.shape
        return [m_serial, m_shape]

    # Reconstructs the original dictionary of numpy.array's from the
    # serialised version and the shape.
    def unserialise(m_serial, m_shape):
        m = {}
        start_index = 0
        for k, s in sorted(m_shape.items()):
            offset = 1
            for d in s:
                offset = offset * d
            end_index = start_index + offset
            m[k] = numpy.reshape(m_serial[start_index:end_index], s)
            start_index = end_index
        return m

    # Returns the functional value with the current controls
    def J(m_serial, m_shape, write_stat=True):
        has_cache = mem_pure_J.has_cache(m_serial, m_shape)
        if has_cache:
            cache_str = "(cache hit)"
        else:
            cache_str = ""

        J = mem_pure_J(m_serial, m_shape)
        print "J = %s %s" % (J, cache_str)

        if write_stat:
            # Update the functional value in the optimisation stat file
            stat_writer[(functional_name, 'value')] = J
        return J

    # A pure version of the computation of J
    def pure_J(m_serial, m_shape):
        if verbose:
            print "Running forward model for functional evaluation (<function pure_J>)"
        m = unserialise(m_serial, m_shape)
        run_model(m, opt_options, model_options)
        simulation_name = superspud(model_options, "libspud.get_option('/simulation_name')")
        stat_file = simulation_name + ".stat"
        s = stat_parser(stat_file)
        if not functional_name in s:
            print "The functional '", functional_name, "' does not exist in the stat file."
            print "Check your model configuration"
            exit()
        J = s[functional_name]["value"][-1]
        return J

    # Returns the functional derivative with respect to the controls.
    def dJdm(m_serial, m_shape, write_stat=True):
        return mem_pure_dJdm(m_serial, m_shape)

    # A pure version of the computation of J
    def pure_dJdm(m_serial, m_shape):
        if verbose:
            print "Running forward/adjoint model for functional derivative evaluation (<function pure_dJdm>)"
        m = unserialise(m_serial, m_shape)
        run_model(m, opt_options, model_options)
        # While computing dJdm we run the forward/adjoint model and in
        # particular we compute the functional values. In order to not compute
        # the functional values again when calling J, we manually add write it
        # into the memoize cache.
        simulation_name = superspud(
            model_options, "libspud.get_option('/simulation_name')")
        stat_file = simulation_name + ".stat"
        J = stat_parser(stat_file)[functional_name]["value"][-1]
        # Add the functional value the memJ's cache
        mem_pure_J.__add__(J, m_serial, m_shape)
        # Now get the functional derivative information
        djdm = read_control_derivatives(opt_options, model_options)
        check_control_consistency(m, djdm, m_bounds)
        # Serialise djdm in the same order than m_serial
        djdm_serial = []
        for k, v in sorted(m_shape.items()):
            djdm_serial = numpy.append(djdm_serial, djdm[k])
        return djdm_serial

    # Check the gradient using the Taylor expansion
    def check_gradient(m_serial, m_shape):
        print '-' * 80
        print ' Entering gradient verification '
        print '-' * 80

        fd_errors = []
        fd_conv = []
        grad_errors = []
        grad_conv = []

        nb_tests = 4
        perturbation = 2e-4
        perturbation_vec = numpy.random.rand(len(m_serial))

        j_unpert = J(m_serial, m_shape)
        djdm_unpert = dJdm(m_serial, m_shape)

        for i in range(nb_tests):
            perturbation = perturbation / 2
            m_pert = m_serial + perturbation * perturbation_vec
            fd_errors.append(abs(j_unpert - J(m_pert, m_shape)))
            grad_errors.append(
                abs(j_unpert + numpy.dot(djdm_unpert, perturbation_vec * perturbation) - J(m_pert, m_shape)))

        print "Error in Taylor expansion of order 0: ", fd_errors
        print "Error in Taylor expansion of order 1: ", grad_errors

        for i in range(nb_tests - 1):
            if fd_errors[i + 1] == 0.0 or fd_errors[i] == 0.0:
                fd_conv.append(1.0)
            else:
                fd_conv.append(math.log(fd_errors[i] / fd_errors[i + 1], 2))
            if grad_errors[i + 1] == 0.0 or grad_errors[i] == 0.0:
                grad_conv.append(2.0)
            else:
                grad_conv.append(
                    math.log(grad_errors[i] / grad_errors[i + 1], 2))

        print "Convergence of Taylor expansion of order 0 (should be 1.0): ", fd_conv
        print "Convergence of Taylor expansion of order 1 (should be 2.0): ", grad_conv

        stat_writer[(functional_name, "iteration")] = 0
        stat_writer[
            (functional_name + "_gradient_error", "convergence")] = min(grad_conv)
        stat_writer.write()

    # This function gets called after each optimisation iteration.
    # It is currently used to write statistics and copy model output files
    # into a subdirectory
    def callback(m_serial, m_shape):
        global iteration
        iteration = iteration + 1
        stat_writer[(functional_name, "iteration")] = iteration
        stat_writer.write()

        if superspud(opt_options, "libspud.have_option('/debug/save_model_output')"):
            save_model_results()

        print '-' * 80
        print ' Finished optimisation iteration', iteration
        print '-' * 80

    def save_model_results():
        global iteration
        # Copy any model output files in a subdirectory
        simulation_name = superspud(
            model_options, "libspud.get_option('/simulation_name')")
        Popen(
            ["mkdir", "opt_" + str(iteration) + "_" + simulation_name.strip()])
        Popen("cp " + simulation_name.strip() + "* " + "opt_" +
              str(iteration) + "_" + simulation_name.strip(), shell=True)

    #

    #
    print '-' * 80
    print ' Beginning of optimisation loop'
    print '-' * 80
    #
    # Initialisation of optimisation loop ###
    global iteration
    iteration = 0
    # Initialise stat file
    if verbose:
        print "Initialise stat file"
    stat_writer = stat_creator(
        superspud(opt_options, "libspud.get_option('/name')").strip() + '.stat')
    # Get the optimisation settings
    if verbose:
        print "Read oml settings"
    algo = superspud(
        opt_options, "libspud.get_option('optimisation_options/optimisation_algorithm[0]/name')")
    have_bound = have_bounds(opt_options, model_options)
    # Create the memoized version of the functional (derivative) evaluation
    # functions
    mem_pure_dJdm = MemoizeMutable(pure_dJdm)
    mem_pure_J = MemoizeMutable(pure_J)

    # Get initial controls ###
    # The initial controls are retrieved in several steps.
    # 1) get custom controls by running the user specified python code and save the associated pkl files
    # 2) run the forward/adjoint model without the "load_control" flag. The model will save the initial default controls as pkl files.
    # 3) Finally load these initial default controls files

    # First we initialise the custom controls
    # This has to be done first since the next step
    # involves running the model and therefore
    # will need the custom controls to be set.
    if verbose:
        print "Get initial custom controls"
    custom_m = get_custom_controls(opt_options)

    # Next run the forward/adjoint model without the option
    # /adjoint/controls/load_controls
    if verbose:
        print "Get initial default controls"
    model_file = superspud(
        opt_options, "libspud.get_option('/model/option_file')")
    if (superspud(model_options, "libspud.have_option('/adjoint/controls/load_controls')")):
        superspud(
            model_options, ["libspud.delete_option('/adjoint/controls/load_controls')", "libspud.write_options('" + model_file + "')"])

    # Run the forward model including adjoint.
    functional_name = superspud(
        opt_options, "libspud.get_option('/functional/name')")
    if superspud(opt_options, "libspud.have_option('/adjoint/functional::" + functional_name + "/disable_adjoint_run')"):
        superspud(
            opt_options, "libspud.delete_option('/adjoint/functional::" + functional_name + "/disable_adjoint_run')")
    [custom_m_serial, custom_m_shape] = serialise(custom_m)
    mem_pure_J(custom_m_serial, custom_m_shape)
    if superspud(opt_options, "libspud.have_option('/debug/save_model_output')"):
        save_model_results()
    # This should have created all the default initial controls and we can now
    # activate the load_controls flag.
    superspud(model_options,
              ["libspud.add_option('/adjoint/controls/load_controls')", "libspud.write_options('" + model_file + "')"])

    # Finally, load the default controls
    m = read_default_controls(opt_options, model_options)
    m_bounds = read_default_control_bounds(opt_options, model_options)
    nb_controls = len(m) + len(custom_m)
    # And merge them
    m.update(custom_m)
    if (nb_controls != len(m)):
        print "Error: Two controls with the same name defined."
        print "The controls must have all unique names."
        print "Your controls are: ", m.keys()
        exit()
    djdm = read_control_derivatives(opt_options, model_options)
    # Now complete the bounds arrays where the user did not specify any bounds
    m_bounds = complete_default_control_bounds(m, m_bounds)
    # Since now all the controls and derivatives are defined, we can check the
    # consistency of the control variables
    check_control_consistency(m, djdm, m_bounds)

    # Serialise the controls for the optimisation routine
    [m_serial, m_shape] = serialise(m)
    [m_lb_serial, m_lb_shape] = serialise(m_bounds["lower_bound"])
    [m_ub_serial, m_ub_shape] = serialise(m_bounds["upper_bound"])
    assert(m_ub_shape == m_shape)
    assert(m_lb_shape == m_shape)
    # zip the lower and upper bound to a list of tuples
    m_bounds_serial = zip(m_lb_serial, m_ub_serial)

    # Check gradient
    if superspud(opt_options, "libspud.have_option('/debug/check_gradient')"):
        check_gradient(m_serial, m_shape)

    #
    if algo != 'NULL':
        print '-' * 80
        print ' Entering %s optimisation algorithm ' % algo
        print '-' * 80
    #

    #
    # BFGS ###############
    #
    if algo == 'BFGS':
        if have_bound:
            print "BFGS does not support bounds."
            exit()
        tol = superspud(
            opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::BFGS/tolerance')")
        maxiter = None
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::BFGS/iterations')"):
            maxiter = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::BFGS/iterations')")
        res = scipy.optimize.fmin_bfgs(
            J, m_serial, dJdm, gtol=tol, full_output=1,
            maxiter=maxiter, args=(m_shape, ), callback = lambda m: callback(m, m_shape))
        print "Functional value J(m): ", res[1]
        print "Control state m: ", res[0]

    #
    # NCG ################
    #
    elif algo == 'NCG':
        if have_bound:
            print "NCG does not support bounds."
            exit()
        tol = superspud(
            opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::NCG/tolerance')")
        maxiter = None
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::NCG/iterations')"):
            maxiter = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::NCG/iterations')")
        res = scipy.optimize.fmin_ncg(
            J, m_serial, dJdm, avextol=tol, full_output=1,
            maxiter=maxiter, args=(m_shape, ), callback = lambda m: callback(m, m_shape))
        print "Functional value J(m): ", res[1]
        print "Control state m: ", res[0]

    #
    # L-BFGS-B ###########
    #
    elif algo == 'L-BFGS-B':
        opt_args = dict(func=J, x0=m_serial, fprime=dJdm, args=(m_shape,))
        if have_bound:
            opt_args['bounds'] = m_bounds_serial
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/tolerance')"):
            pgtol = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/tolerance')")
            opt_args['pgtol'] = pgtol
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/factr')"):
            factr = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/factr')")
            opt_args['factr'] = factr
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/memory_limit')"):
            memory_limit = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/memory_limit')")
            opt_args['m'] = memory_limit
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/maximal_functional_evaluations')"):
            maxfun = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/maximal_functional_evaluations')")
            opt_args['maxfun'] = maxfun
        if superspud(opt_options, "libspud.have_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/verbosity')"):
            iprint = superspud(
                opt_options, "libspud.get_option('/optimisation_options/optimisation_algorithm::L-BFGS-B/verbosity')")
            opt_args['iprint'] = iprint

        res = scipy.optimize.fmin_l_bfgs_b(**opt_args)
        print res
    #
    # NULL ##############
    #
    elif algo == 'NULL':
        exit()
    else:
        print "Unknown optimisation algorithm in option path."
        exit()


# main()  ###################
def main():
    global verbose
    global debug

    parser = argparse.ArgumentParser(
        description='Optimisation program for fluidity.')
    parser.add_argument('filename', metavar='FILE', help="the .oml file")
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='verbose mode')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='the debug mode runs additional internal tests.')
    args = parser.parse_args()
    verbose = args.verbose
    debug = args.debug

    if not os.path.isfile(args.filename):
        print "File", args.filename, "not found."
        exit()

    # Initial spud environments for the optimality and model options.
    opt_file = args.filename
    if not superspud(opt_file, "libspud.have_option('/optimisation_options')"):
        print "File", args.filename, "is not a valid .oml file."
        exit()
    model_file = superspud(
        opt_file, "libspud.get_option('/model/option_file')")
    if not os.path.isfile(model_file):
        print "Could not find ", model_file, " as specified in /model/option_file"
        exit()

    # Create a copy of the option files so that we don't touch the original
    def rename_file(fn):
        fn_basename, fn_extension = os.path.splitext(fn)
        shutil.copy(fn, fn_basename + '_tmp' + fn_extension)
        fn = fn_basename + '_tmp' + fn_extension
        return fn

    model_file = rename_file(model_file)
    opt_file = rename_file(opt_file)
    superspud(
        opt_file, ["libspud.set_option('/model/option_file', '" + model_file + "')", "libspud.write_options('" + opt_file + "')"])

    # Check consistency of the option files
    check_option_consistency(opt_file, model_file)

    # Start the optimisation loop
    optimisation_loop(opt_file, model_file)

    # Tidy up
    os.remove(opt_file)
    os.remove(model_file)

# __main__ ########################
if '__main__' == __name__:
    start_time = time.time()
    main()
    print "Optimisation finished in ", time.time() - start_time, "seconds"
