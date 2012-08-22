# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""PyOP2 configuration module.

The PyOP2 configuration module exposes itself as a dictionary object holding
configuration options.

Example::

    from pyop2 import configuration as cfg

    # should be called once by the backend selector logic.
    # configuration values can be overiden upon calling 'configure'
    cfg.configure(backend='opencl', debug=6)
    # or using a specific yaml configuration file
    cfg.configure(opconfig='./conf-alt.yaml')

    # configuration value access:
    cfg['backend'] :> 'opencl'
    # attribute accessor also supported
    cfg.backend :> 'opencl'

Configuration option lookup order:

    1. Named parameters specified at configuration.
    2. From `opconfig` configuration file if specified
    3. From user configuration `./pyop2.yaml` (relative to working directory)
       if present and no `opconfig` specified
    4. From default value defined by pyop2 (`assets/default.yaml`)
    5. KeyError

Reserved option names:
    - configure, reset, __*__
"""

import types
import sys
import yaml
import pkg_resources
import warnings
import UserDict

class ConfigModule(types.ModuleType):
    """Dictionary impersonating a module allowing direct access to attributes."""

    OP_CONFIG_KEY = 'config'
    DEFAULT_CONFIG = 'assets/default.yaml'
    DEFAULT_USER_CONFIG = 'pyop2.yaml'

    def configure(self, **kargs):
        self._config = UserDict.UserDict()

        entries = list()
        entries += yaml.load(pkg_resources.resource_stream('pyop2', ConfigModule.DEFAULT_CONFIG)).items()

        alt_user_config = False
        if kargs.has_key(ConfigModule.OP_CONFIG_KEY):
            alt_user_config = True
            try:
                from_file = yaml.load(kargs[ConfigModule.OP_CONFIG_KEY])
                entries += from_file.items() if from_file else []
            except IOError:
                pass

        if not alt_user_config:
            try:
                from_file = yaml.load(file(ConfigModule.DEFAULT_USER_CONFIG))
                entries += from_file.items() if from_file else []
            except IOError as e:
                pass

        entries += kargs.items()
        self._config = UserDict.UserDict(entries)

    def reset(self):
        """Reset all configuration entries."""
        self._config = None

    def __getitem__(self, key):
        if not self._config:
            raise KeyError
        return self._config[key]

    def __getattr__(self, name):
        if not self._config:
            raise AttributeError
        return self._config[name]

_original_module = sys.modules[__name__]
_fake = ConfigModule(__name__)
_fake.__dict__.update({
    '__file__': __file__,
    '__package': 'pyop2',
    #'__path__': __path__, #__path__ not defined ?
    '__doc__': __doc__,
    #'__version__': __version__, #__version__ not defined ?
    '__all__': (),
    '__docformat__': 'restructuredtext en'
})
sys.modules[__name__] = _fake
