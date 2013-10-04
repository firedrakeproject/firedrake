#!/usr/bin/env python

__author__ = "Patrick Farrell"
__license__ = "LGPL"
__credits__ = ["Patrick Farrell", "David Ham"]
__blame__ = ["Patrick Farrell"]

import re


def parse(code):
    """Given the string containing code for a functional or its derivative,
    work out what dependencies on which state it has. E.g., if the code was

    from math import sin, pi
    coord  = states[n]["Fluid"].vector_fields["Coordinate"]
    u = states[n+1]["Fluid"].scalar_fields["Velocity"]
    du = states[n-1]["Fluid"].scalar_fields["VelocityDerivative"]

    for i in range(du.node_count):
      x = coord.node_val[i][0]
        du.set(i, 0.01125*pi**2*sin(3.0/20*(x + 10)*pi)

    then this routine should return the list
    [-1, 0, +1]."""

    # My beautiful regex, made with the help of http://re.dabase.com/
    regex = re.compile('''states\[(?P<n>[n0-9+-]*)\]''')

    return sorted(set(map(eval, re.findall(regex, code))))

if __name__ == "__main__":
    code = '''
  from math import sin, pi
  coord  = states[n]["Fluid"].vector_fields["Coordinate"]
  u = states[n+1]["Fluid"].scalar_fields["Velocity"]
  du = states[n-1]["Fluid"].scalar_fields["VelocityDerivative"]

  for i in range(du.node_count):
    x = coord.node_val[i][0]
      du.set(i, 0.01125*pi**2*sin(3.0/20*(x + 10)*pi)
  '''

    print parse(code)


def make_adj_variables(d):
    """ d is a dict like
    {"Fluid::Velocity": [0, 3, 4],
     "Fluid::Pressure": [2, 3, 5]}
    Take this, and make a list of things we can easily convert into
    adj_variables."""

    varlist = []
    for key in d:
        for timestep in d[key]:
            if timestep < 0:
                print "Warning: dependencies function returned a variable with timestep %d." % timestep
            else:
                newd = {}
                newd['name'] = key
                newd['timestep'] = timestep
                varlist.append(newd)

    return varlist
